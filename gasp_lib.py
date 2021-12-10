#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marc-Antoine Martinod

GASP: GLINT As Simulated in Python

Simulation of a nulling interferometer based on the following features:
    - Atmospheric turbulence and wavefront correction
    - Integrated-optics remapper and combiner
    - Spectral dispersion
    - Detector

This is the library of GASP where all the functions are stored.
The primary development aims at simulating GLINT Mark II (4T chip as
described in Martinod et al. (2021)).
However, the ambition is to create interface allowing to imprement
any kind of combiner, apertures, spectral dispersions and detector.

Some conventions to respect:
    - Spectral information is always on axis 0 of array as matrix product\
        will iterate along this axis
    - If photometric outputs are designed, they must be put in the last rows
"""

import numpy as np
import h5py
import xaosim as xs
import matplotlib.pyplot as plt
import hcipy as hp
from astropy.io import fits
from scipy.ndimage import rotate

# =============================================================================
# Generic functions
# =============================================================================
def bin_array(arr, binning, axis=0, avg=True):
    """
    Bin array along the specified axis.
    
    :param arr: Array containing data to bin
    :type arr: nd-array
    :param binning: Number of elements to bin. If `arr.shape[axis]%binning!=0`,\
        remaining rows along this axis will be dropped.
    :type binning: int
    :param axis: axis along which the frames are, defaults to 0
    :type axis: int, optional
    :param avg: If ``True``, it returns the average of the binned frame.\
        Otherwise, return its sum, defaults to False
    :type avg: bool, optional
    :return: binned array
    :rtype: array

    """
    if binning is None:
        binning = arr.shape[axis]

    shape = arr.shape
    # Number of frames which can be binned respect to the input value
    crop = shape[axis]//binning*binning
    arr = np.take(arr, np.arange(crop), axis=axis)
    shape = arr.shape
    if axis < 0:
        axis += arr.ndim
    shape = shape[:axis] + (-1, binning) + shape[axis+1:]
    arr = arr.reshape(shape)
    if not avg:
        arr = arr.sum(axis=axis+1)
    else:
        arr = arr.mean(axis=axis+1)

    return arr


def oversample_wavelength(wl, oversampling_factor):
    """
    Oversample the wavelength axis.
    Useful to create temporal incoherence by binning the oversampled
    spectrally dispersed interferometric signal.
    The oversampled wavelength should go through the whole simulation.
    
    Note: the wavenlength axis **must be equally** spaced.
    
    :param wl: Dispersed wavelength to oversample.
    :type wl: 1D-array
    :param oversampling_factor: number of elements to add between each\
        intervall [wl[i], wl[i+1][.
    :type oversampling_factor: int
    :return: Oversampled wavelength array.
    :rtype: 1D-array
    """
    oversampled_wl = []
    dwl = np.diff(wl)[0]
    for i in range(len(wl)):
        oversample = np.linspace(wl[i]-dwl/2, wl[i]+dwl/2, oversampling_factor, endpoint=True)
        oversampled_wl.append(oversample)
        
    oversampled_wl = np.array(oversampled_wl)
    
    return oversampled_wl.flatten()

# =============================================================================
# Combiner part
# =============================================================================
"""
Functions used to create the photonic combiner.
The model is made on a matrix frame.
By convention, one follows the organisation of the combination as
(1-2, 1-3,... 1-N then 2-3, 2-4,...,2-N etc.) and the photometric outputs
are set at the end.
"""

"""
List of the baselines in that order: 12, 13, 14, 23, 24, 34 to model the
photonic chip of GLINT Mark II as a diagonal block matrix.
"""
LIST_BASELINES = ['null1', 'null5', 'null3', 'null2', 'null6', 'null4']
LIST_PHOTO = ['p1', 'p2', 'p3', 'p4']

def load_zeta_file(zeta_path):
    """
    Load the zeta coefficient from a HDF5 file.
    
    :param zeta_path: path to the file to load.
    :type zeta_path: str
    :return: Dictionary containing the spectral values of the zeta coefficients.
    :rtype: dict

    """
    zeta_file = h5py.File(zeta_path, 'r')
    zeta_dic = {}
    for key, value in zeta_file.items():
        zeta_dic[key] = np.array(value)
    zeta_file.close()
    return zeta_dic


def zeta_to_splitcoupler_coeffs(zeta_file, wl, baseline):
    """Calculate chromatic coefficients for directional coupler from\
        real data.

    :param zeta_file: zeta coefficients
    :type zeta_file: dict
    :param wl: wavelength to which interpolate the zeta coefficients, in metre
    :type wl: 1D-array
    :return: splitting coefficients of beams A and B, coupling coefficients
        of the coupler
    :rtype: 4-tuple, each element is a 1D-array of size same as `wl`.

    """
    null_table = {'null1': [1, 2], 'null2': [2, 3], 'null3': [1, 4],
                  'null4': [3, 4], 'null5': [1, 3], 'null6': [2, 4]}

    beams_table = {1: ['b1null1', 'b1null7', 'b1null3', 'b1null9', 'b1null5', 'b1null11'],
                   2: ['b2null1', 'b2null7', 'b2null2', 'b2null8', 'b2null6', 'b2null12'],
                   3: ['b3null2', 'b3null8', 'b3null4', 'b3null10', 'b3null5', 'b3null11'],
                   4: ['b4null3', 'b4null9', 'b4null4', 'b4null10', 'b4null6', 'b4null12']}

    wl_scale = zeta_file['wl_scale'] * 1e-9  # wavelength scale

    # null/antinull outputs for beams 1 and 2 (zeta coefficients)
    idA, idB = null_table[baseline]  # Id of corresponding beams to baseline
    zetas_bA = np.array([zeta_file[elt] for elt in beams_table[idA]])
    zetas_bB = np.array([zeta_file[elt] for elt in beams_table[idB]])

    zeta_bA_nX = zeta_file['b'+str(idA)+'null'+baseline[-1]]
    zeta_bA_anX = zeta_file['b'+str(idA)+'null'+str(int(baseline[-1])+6)]
    zeta_bB_nX = zeta_file['b'+str(idB)+'null'+baseline[-1]]
    zeta_bB_anX = zeta_file['b'+str(idB)+'null'+str(int(baseline[-1])+6)]

    # central wavelength 1550 +/- 100 nm
    # within = ((wl_scale <= 1650) & (wl_scale >= 1350))
    # wl_scale = np.array(wl_scale[within])
    # zeta_bA_nX = zeta_bA_nX[within]
    # zeta_bA_anX = zeta_bA_anX[within]
    # zeta_bB_nX = zeta_bB_nX[within]
    # zeta_bB_anX = zeta_bB_anX[within]
    # zetas_bA = zetas_bA[:, within]
    # zetas_bB = zetas_bB[:, within]

    zetas_bA = np.array([np.interp(wl, wl_scale[::-1], elt[::-1]) for elt in zetas_bA])
    zetas_bB = np.array([np.interp(wl, wl_scale[::-1], elt[::-1]) for elt in zetas_bB])
    zeta_bA_nX = np.interp(wl, wl_scale[::-1], zeta_bA_nX[::-1])
    zeta_bA_anX = np.interp(wl, wl_scale[::-1], zeta_bA_anX[::-1])
    zeta_bB_nX = np.interp(wl, wl_scale[::-1], zeta_bB_nX[::-1])
    zeta_bB_anX = np.interp(wl, wl_scale[::-1], zeta_bB_anX[::-1])

    # splitting ratio for beams A and B
    splitting_bA_to_interf = (zeta_bA_nX + zeta_bA_anX) / \
                              (1 + np.sum(zetas_bA, 0))
    splitting_bB_to_interf = (zeta_bB_nX + zeta_bB_anX) / \
                              (1 + np.sum(zetas_bB, 0))
    splitting_bA_to_photo = 1 / (1 + np.sum(zetas_bA, 0))
    splitting_bB_to_photo = 1 / (1 + np.sum(zetas_bB, 0))

    # Coupling coefficients inside the coupler 1
    kappa_AB = (zeta_bA_anX / zeta_bA_nX) / (1 + (zeta_bA_anX / zeta_bA_nX))
    kappa_BA = (zeta_bB_nX / zeta_bB_anX) / (1 + (zeta_bB_nX / zeta_bB_anX))

    """
    Wavelength scale; note we cut off the highest and lowest wavelengths
    as zeta coeffs become messy there
    """

    return splitting_bA_to_interf, splitting_bB_to_interf,\
        kappa_AB, kappa_BA,\
            splitting_bA_to_photo, splitting_bB_to_photo


def create_directional_coupler(kappa_AB, kappa_BA):
    """Get transfer matrix of a directional coupler.

    Transfer matrix of a directional coupler.

    Structure:
        1st row = null output
        2nd row = bright output
        3rd row = photometric output A
        4th row = photometric output B
        
    :param kappa_AB: coupling coefficients from waveguide A to waveguide B
    :type kappa_AB: 1D-array of M spectral values
    :param kappa_BA: coupling coefficients from waveguide B to waveguide A
    :type kappa_BA: 1D-array of M spectral values
    :return: matrix modelling the coupling of the incoming beams
    :rtype: (M, Nout, Nin) array with M the number of spectral element,\
        Nout output waveguide and Nin input waveguides.
    """
    z = np.zeros(kappa_AB.shape)
    ones = np.ones(kappa_AB.shape)
    directional_coupler = np.array([[(1-kappa_AB)**0.5,
                            kappa_BA**0.5 * np.exp(-1j * np.pi/2), z, z],
                          [kappa_AB**0.5 *
                            np.exp(-1j * np.pi/2), (1-kappa_BA)**0.5, z, z],
                          [z, z, ones, z],
                          [z, z, z, ones]])
    
    # Put wavelength first
    directional_coupler = np.transpose(directional_coupler, axes=(2, 0, 1))
    return directional_coupler


def get_glint_chip_markII(wl, zeta_dict=None):
    """
    Model of the combiner in GLINT Mark II (Martinod et al. (2021)) which consits
    of 6 couplers delivering pi/2_phase shited outputs and 4 photometric outputs.
    
    :param wl: wavelength
    :type wl: 1D-array of N values of wavelength
    :param zeta_file: zeta coefficients used to deduce splitting and coupling\
        coefficients. It can be left to `None` to use ideal, achromatic values.
    :type zeta_file: dict, default is None
    :return: combiner of GLINT
    :rtype: 3D-array (wl, outputs, inputs)

    """
    # global coupler, splitter_interfA, splitter_interfB, kappa_AB, kappa_BA, splitter_photoA, splitter_photoB
    # global coupler_mat, splitter_mat

    """
    The splitter matrix contains 2*nb baselines + 4 photo = 16 lines
    For the first 12 lines: pair of successive lines (1st and 2nd, 3rd and 4th etc.)
    represents one baseline. Inside a pair, each line represents one beam.
    The table below indicate the coordinates of the splitting coefficient
    (fraction of light sent to one coupler) for beam A (even line), beam B (odd line).
    The last 4 lines are the fraction of light sent to the photometric outputs.

    Structure of the lists in the table in 3 levels:
        - 1st level: for one baseline, splitting coefficient to coupler and to photometry
        - 2nd level: for one baseline, for coupler/photometry branch, indexes of coefficients for beams A and B
        - 3rd level: for one baseline, for coupler/photometry branch, for beam A/B, location in the matrix
    """
    splitter_idx_table = {'null1': [[[0, 0], [1, 1]], [[12, 0], [13, 1]]],
                          'null5': [[[2, 0], [3, 2]], [[12, 0], [14, 2]]],
                          'null3': [[[4, 0], [5, 3]], [[12, 0], [15, 3]]],
                          'null2': [[[6, 1], [7, 2]], [[13, 1], [14, 2]]],
                          'null6': [[[8, 1], [9, 3]], [[13, 1], [15, 3]]],
                          'null4': [[[10, 2], [11, 3]], [[14, 2], [15, 3]]]}
    splitter_mat = np.zeros((wl.size, 16, 4))

    """
    Matrix of all the couplers, the baselines are ordered so that
    this matrix is a block identity matrix.
    Each element of the diagonal per block is a coupler or photometric guide.
    """
    coupler_mat = np.zeros((wl.size, 16, 16), dtype=np.complex128)
    coupler_mat[:, -4:, -4:] = np.tile(np.eye(4), (wl.size, 1, 1))

    splitter_interfA_list = []
    splitter_interfB_list = []
    kappa_AB_list = []
    kappa_BA_list = []
    splitter_photoA_list = []
    splitter_photoB_list = []
    
    for bl in LIST_BASELINES:
        if zeta_dict is None:
            print('Ideal coefficients used')
            kappa_AB = kappa_BA = 0.5 * np.ones(wl.shape)
            splitter_interfA = splitter_interfB = 2/7. * np.ones(wl.shape)
            splitter_photoA = splitter_photoB = 1/7. * np.ones(wl.shape)
        else:
            splitter_interfA, splitter_interfB, kappa_AB, kappa_BA, splitter_photoA, splitter_photoB =\
                zeta_to_splitcoupler_coeffs(zeta_dict, wl, bl)            
        idxA, idxB = splitter_idx_table[bl][0]
        idxAp, idxBp = splitter_idx_table[bl][1]
        # sqrt because wavefront is propagated instead of intensity
        splitter_mat[:, idxA[0], idxA[1]] = splitter_interfA**0.5
        splitter_mat[:, idxB[0], idxB[1]] = splitter_interfB**0.5

        splitter_mat[:, idxAp[0], idxAp[1]] = splitter_photoA**0.5
        splitter_mat[:, idxBp[0], idxBp[1]] = splitter_photoB**0.5

        coupler = create_directional_coupler(kappa_AB, kappa_BA)
        coupler = coupler[:, :2, :2]

        coupler_mat[:, idxA[0]:idxA[0]+2, idxA[0]:idxA[0]+2] = coupler
        
        splitter_interfA_list.append(splitter_interfA)
        splitter_interfB_list.append(splitter_interfB)
        kappa_AB_list.append(kappa_AB)
        kappa_BA_list.append(kappa_BA)
        splitter_photoA_list.append(splitter_photoA)
        splitter_photoB_list.append(splitter_photoB)

    splitter_interfA_list = np.array(splitter_interfA_list)
    splitter_interfB_list = np.array(splitter_interfB_list)
    kappa_AB_list = np.array(kappa_AB_list)
    kappa_BA_list = np.array(kappa_BA_list)
    splitter_photoA_list = np.array(splitter_photoA_list)
    splitter_photoB_list = np.array(splitter_photoB_list)
    
    combiner4T = coupler_mat@splitter_mat
    return combiner4T, splitter_interfA_list, splitter_interfB_list, kappa_AB_list, kappa_BA_list, splitter_photoA_list, splitter_photoB_list


# =============================================================================
# Detector part
# =============================================================================
"""
These functions create the image as sampled on the detector and simulate the noises:
    - photon noise and dark current (Poisson distribution)
    - Read-out noise (Normal distribution)
    - Digitisation to convert pixels' values in 16-bit integers
"""


def create_tracks(ysz, channel_positions, sigmas):
    """
    Create canvas for tracks on the detector.
    
    :param ysz: size of the detector along the spatial axis.
    :type ysz: int
    :param channel_positions: positions of the N tracks for the M different spectral channels
    :type channel_positions: 2D-array of size (N x M) with N the number\
        of tracks and M the number of spectral channels
    :param sigmas: width of the N tracks for the M different spectral channels
    :type sigmas:  2D-array of size (N x M) with N the number\
        of tracks and M the number of spectral channels
    :return: tracks normalised in spectral flux
    :rtype: 3D-array of size (N x ysz x M)

    """
    spatial_axis = np.arange(ysz)
    channel_positions = np.array(channel_positions)
    
    if channel_positions.ndim == 1:
        channel_positions = channel_positions[:,None]
    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([sigmas])
    if sigmas.ndim == 1:
        sigmas = sigmas[:,None]
        

    tracks = np.exp(-(spatial_axis[:,None,None] - channel_positions[None,:,:])**2 / (2*sigmas[None,:,:]**2))
    # Tracks structure is (spatial axis, output, spectral dispersion)
    tracks = np.transpose(tracks, (1, 2, 0)) # New axes order: output, spectral, spatial
    tracks = tracks / np.sum(tracks, 2)[:,:,None] # Normalise values along spectral axis
    
    return tracks


def create_image(ysz, xsz, col_start, channel_positions, sigmas, outputs):
    """
    Create the image projected on the detector. Origin of the detector is considered
    top-left.

    :param ysz: number of rows on the detector
    :type ysz: int
    :param xsz: number of columns on the detector
    :type xsz: int
    :param col_bounds: i-th column at which the spectrum starts being displayed.\
        The bounds may vary deending on the output.
    :type col_start: list of N integers for N outputs
    :param channel_positions: positions of the N tracks for the M different spectral channels
    :type channel_positions: 2D-array of size (N x M) with N the number\
        of tracks and M the number of spectral channels
    :param sigmas: width of the N tracks for the M different spectral channels
    :type sigmas:  2D-array of size (N x M) with N the number\
        of tracks and M the number of spectral channels
    :param outputs: Spectral fluxes of the different outputs.
    :type outputs: 2D-array of (N outputs x M spectral channels)
    :return: Image projected on the detector.
    :rtype: 2D-array of size (ysz x xsz)

    """    
    if not isinstance(col_start, (list, np.ndarray)):
        raise('col_start needs to be a list or 1D-array')
    
    col_start = np.array(col_start)
    nb_wl = outputs.shape[1]

    if np.any(col_start + nb_wl > xsz):
        print('Part of some spectra may be out of the detector')
        
    if len(col_start) == 1:
        col_start = np.array([col_start[0]] * nb_wl)
    
    tracks = create_tracks(ysz, channel_positions, sigmas)
    flux = outputs[:,:,None] * tracks
    img = np.zeros((ysz, xsz))
    for k in range(flux.shape[0]): # Iterate over outputs
        img += np.pad(flux[k].T, ((0,0),(col_start[k], xsz - (col_start[k]+nb_wl))),\
                             mode='constant', constant_values=0.)
        
    return img, tracks


def add_noise(image, Ndark, gainsys, offset, sigma_ron, activate_poisson, activate_ron, activate_digit):
    """
    Wrapper which generate all the noises when the signal is acquired by the\
        the detector.
        
    Note: `image` can be the signal as projected on the detector or the outputs themselves.
        
    :param image: Frame to noise
    :type image: 2D-array
    :param Ndark: dark current, in electrons per frame.
    :type Ndark: float
    :param gainsys: Conversion gain from e- to ADU, in ADU/e-
    :type gainsys: float
    :param offset: Offset added by the readout in the digitilization process
    :type offset: float
    :param sigma_ron: standard deviation of the reda-out noise
    :type sigma_ron: float
    :param activate_poisson: add Poisson noise if `True`
    :type activate_poisson: bool
    :param activate_ron: add read-out process if `True`
    :type activate_ron: bool
    :param activate_digit: digitise the frame if `True`
    :type activate_digit: bool
    :return: Noised frames
    :rtype: 2D-array

    """
    noisy_image = image
    if activate_poisson:
        noisy_image = add_poisson_noise(noisy_image, Ndark)
    if activate_ron:
        noisy_image = do_readout(noisy_image, gainsys, offset, sigma_ron)
    if activate_digit:
        noisy_image = convert16bits(noisy_image)

    return noisy_image


def add_poisson_noise(image, Ndark):
    """
    Generate the Poisson noise from photons and dark current.
    
    :param image: signal to add Poisson noise, in electron\
        (so consider to add the quantum efficiency in the `image` input)
    :type image: 2D-array
    :param Ndark: dark current, in electrons per frame.
    :type Ndark: float
    :return: image with Poisson noise.
    :rtype: 2D-array

    """
    dark = Ndark * np.ones(image.shape)
    return np.random.poisson(dark+image)

    
def do_readout(image, gainsys, offset, sigma_ron):
    """
    Simulate the read-out: applies the conversion gain ADU/e- and add\
        Gaussian noise.

    :param image: Image acquired by the pixel area
    :type image: 2D-array
    :param gainsys: Conversion gain from e- to ADU, in ADU/e-
    :type gainsys: float
    :param offset: Offset added by the readout in the digitilization process
    :type offset: float
    :param sigma_ron: standard deviation of the reda-out noise
    :type sigma_ron: float
    :return: image as read-out
    :rtype: 2D-array

    """
    signal = image * gainsys # ADU
    ron = np.random.normal(0, sigma_ron, image.shape)
    signal = signal + offset + ron
    return signal
    

def convert16bits(image):
    """
    Digitilize the frame on 16-bits ints.
    
    :param image: frame to digitise
    :type image: 2D-array
    :return: Digitised array
    :rtype: 2D-array of int16

    """
    converted = image.astype(np.uint16)
    if np.any(converted>2**16):
        print('WARNING: saturation of the detector')
    return converted
    

# =============================================================================
# Pupil part (mask + atmosphere)
# =============================================================================
def atmo_screen(isz, ll, r0, L0, fc=19.5, correc=1.0, pdiam=None, seed=None):
    """
    Credit: xaosim lib (Frantz Martinache).
    
    The Kolmogorov - Von Karman phase screen generation algorithm.
    Adapted from the work of Carbillet & Riccardi (2010).
    http://cdsads.u-strasbg.fr/abs/2010ApOpt..49G..47C
    Kolmogorov screen can be altered by an attenuation of the power
    by a correction factor *correc* up to a cut-off frequency *fc*
    expressed in number of cycles across the phase screen.
    
    :param isz:  the size of the array to be computed (in pixels)
    :type isz: int
    :param ll: the physical extent of the phase screen (in meters)
    :type ll: float
    :param r0: the Fried parameter, measured at a given wavelength (in meters)
    :type r0: float
    :param L0: the outer scale parameter (in meters)
    :type L0: float
    :param fc: DM cutoff frequency (in lambda/D), defaults to 19.5
    :type fc: float, optional
    :param correc: correction of wavefront amplitude (factor 10, 100, ...), defaults to 1.0
    :type correc: float, optional
    :param pdiam: pupil diameter (in meters), defaults to None. If None,\
        pdiam is set to the physical extent of the phase screen `ll`
    :type pdiam: float, optional
    :param seed: set the seed of the random number generator, defaults to None
    :type seed: int, optional
    :return: atmospheric screen in radian
    :rtype: 2D-array

    """
    if not seed is None:
        np.random.seed(seed)

    phs = 2*np.pi * (np.random.rand(isz, isz) - 0.5)

    xx, yy = np.meshgrid(np.arange(isz)-isz/2, np.arange(isz)-isz/2)
    rr = np.hypot(yy, xx)
    rr = np.fft.fftshift(rr)
    rr[0, 0] = 1.0

    modul = (rr**2 + (ll/L0)**2)**(-11/12.)

    if pdiam is not None:
        in_fc = (rr < fc * ll / pdiam)
    else:
        in_fc = (rr < fc)

    modul[in_fc] /= correc

    screen = np.fft.ifft2(modul * np.exp(1j*phs)) * isz**2
    screen *= np.sqrt(2*0.0228)*(ll/r0)**(5/6.)

    screen -= screen.mean()
    return(screen)


def generatePhaseScreen(wavel_r0, isz, ll, r0, L0, fc=19.5, correc=1., pdiam=None, seed=None):
    """
    Generate phase screen in meter.

    :param isz:  the size of the array to be computed (in pixels)
    :type isz: int
    :param ll: the physical extent of the phase screen (in meters)
    :type ll: float
    :param r0: the Fried parameter, measured at a given wavelength (in meters)
    :type r0: float
    :param L0: the outer scale parameter (in meters)
    :type L0: float
    :param fc: DM cutoff frequency (in lambda/D), defaults to 19.5
    :type fc: float, optional
    :param correc: correction of wavefront amplitude (factor 10, 100, ...), defaults to 1.0
    :type correc: float, optional
    :param pdiam: pupil diameter (in meters), defaults to None. If None,\
        pdiam is set to the physical extent of the phase screen `ll`
    :type pdiam: float, optional
    :param seed: set the seed of the random number generator, defaults to None
    :type seed: int, optional
    :return: atmospheric screen
    :rtype: 2D-array

    """
    phs_screen = atmo_screen(isz, ll, r0, L0, fc=fc,
                             correc=correc, pdiam=pdiam, seed=seed)
    phs_screen = phs_screen.real * wavel_r0 / (2*np.pi)
    return phs_screen


def movePhaseScreen(phase_screens, v_wind, angle_wind, time, meter2pixel):
    """
    Simulate the movement of the phase because of wind.

    :param phase_screens: Phase screen to shift
    :type phase_screens: 2D-array
    :param v_wind: speed of the wind in m/s
    :type v_wind: float
    :param angle_wind: Orientation of the wind in degre
    :type angle_wind: float
    :param time: time spent since the creation of the phase screen
    :type time: float
    :param meter2pixel: conversion factor from meter to pixel
    :type meter2pixel: float
    :return: Moved phase screen
    :rtype: 2D-array

    """

    # phase_screens = cp.array(phase_screens, dtype=cp.float32)
    yshift_in_pix = int(
        np.around(v_wind * time * meter2pixel * np.sin(np.radians(angle_wind))))
    xshift_in_pix = int(
        np.around(v_wind * time * meter2pixel * np.cos(np.radians(angle_wind))))

    return np.roll(phase_screens, (yshift_in_pix, xshift_in_pix), axis=(-2, -1)), (xshift_in_pix, yshift_in_pix)


def uniform_disk(ysz, xsz, radius, rebin, between_pix=False, norm=False):
    """
    Create a uniform disk. Can be used for creating an aperture or a UD-star.

    :param ysz: number of rows of the array containing the pupil
    :type ysz: int
    :param xsz: number of columns of the array containing the pupil
    :type xsz: int
    :param radius: radius of the disk, in pixel
    :type radius: int
    :param rebin: rebin factor to smooth edges
    :type rebin: int
    :param between_pix: DESCRIPTION, defaults to False
    :type between_pix: bool, optional
    :param norm: normalised the pupil so that `np.sum(pupil**2)=1`, defaults to False
    :type norm: bool, optional
    :return: (ys x xs) array with a uniform disk of radius `radius`
    :rtype: 2D-array

    """
    xsz2 = xsz * rebin
    ysz2 = ysz * rebin
    radius2 = radius * rebin

    if between_pix is False:
        xx, yy = np.meshgrid(np.arange(xsz2)-xsz2//2, np.arange(ysz2)-ysz2//2)
    else:
        xx, yy = np.meshgrid(np.arange(xsz2)-xsz2//2+0.5,
                             np.arange(ysz2)-ysz2//2+0.5)
    mydist = np.hypot(yy, xx)
    res = np.zeros_like(mydist)
    res[mydist <= radius2] = 1.0
    res = np.reshape(res, (ysz, rebin, xsz, rebin))
    res = res.mean(3).mean(1)
    if norm:
        res = res / (np.sum(res))

    return(res)


def calculate_injection(phs_screen, wl, geo_inj=0.8):
    """
    Calculate the injection with the Marechal approximation.

    :param phs_screen: phase screen across the pupil in the same unit as `wl`
    :type phs_screen: 2D-array
    :param wl: operating wavelengths
    :type wl: 1D-array
    :param geo_inj: geometric injection, defaults to 0.8
    :type geo_inj: float, optional
    :return: injection vs wavelength
    :rtype: 1D-array

    """
    rms = np.std(phs_screen)
    strehl = np.exp(-(2*np.pi/wl)**2 * rms**2)
    injections = geo_inj * strehl
    return injections


def make_N_hole_mask(csz, holepositions, holeradius, norm):
    '''
    Makes the 3D printed mask with preset coordinates
    '''
    hrad  = holeradius    # hole radius in pupil-pixels

    array  = np.zeros((csz, csz)) # array filled with zeros
    hole   = uniform_disk(csz, csz, hrad, 5, between_pix=True, norm=norm)

    # The Positions of the apertures 
    maskarray = np.zeros((len(holepositions), *hole.shape), dtype=bool)
    
    for i in range(len(holepositions)):
        dx = int(holepositions[i,0])
        dy = int(holepositions[i,1])
        aperture = np.roll(np.roll(hole, dx, axis=1), dy, axis=0)
        maskarray[i][aperture!=0] = True
        array += np.roll(np.roll(hole, dx, axis=1), dy, axis=0)
        
    return array, maskarray


def create_aperture_mask(pup_tel, pupil_diam, nrings, ring_rad, aper_rad, rot, holes_idx=[36, 3, 33, 22], norm=False, view_pupil=False):
    # scex = xs.instrument("scexao", csz=csz)
    # pupil = scex.cam.pupil
    # pupil_diam = scex.cam.pdiam
    # pscale = pupil_diam / csz
    
    pupil = pup_tel
    csz = len(pupil)
    pscale = pupil_diam / csz
    
    mems_seg_array = xs.pupil.hex_grid_coords(nrings, ring_rad/pscale, rot=rot).T
    glint_mask_pos_csz = mems_seg_array[holes_idx,:]    
    glint_mask, maskarray = make_N_hole_mask(csz, glint_mask_pos_csz, aper_rad/pscale, norm)
    
    if view_pupil:
        x, step = np.linspace(-pupil_diam/2, pupil_diam/2, csz, retstep=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(glint_mask/glint_mask.max() * pupil.max() + pupil, origin='lower',
                   extent=[x[0]-step/2, x[-1]+step/2, x[0]-step/2, x[-1]+step/2],
                   aspect='equal')
        plt.scatter(glint_mask_pos_csz[:,0]*pscale, glint_mask_pos_csz[:,1]*pscale, marker='o', c='r', s=400)
        plt.scatter(mems_seg_array[:,0]*pscale, mems_seg_array[:,1]*pscale, marker='o', c='k')
        plt.grid(False)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Subaru pupil + GLINT mask + Hex mirrors locations')
        plt.tight_layout()
    
    return pupil * glint_mask, maskarray

def make_scexao_pupil(path, rot_angle, flip_ud, plotting=False):
    raw_pupil = fits.getdata(path)
    if flip_ud:
        raw_pupil = np.flipud(raw_pupil)
    raw_pupil = rotate(raw_pupil, rot_angle)
    pupil_grid = hp.make_pupil_grid(raw_pupil.shape, diameter=(8.2 * 0.95)) # aperture size = 7.8m, even though mirror = 8.2m
    subaru_pup = hp.Field(raw_pupil, pupil_grid) # use this as input to hp.Wavefront
    
    if plotting:
        plt.figure()
        plt.imshow(subaru_pup, aspect='equal', cmap='gray',
                   extent=[pupil_grid.x[0]-pupil_grid.delta[0]/2,
                           pupil_grid.x[-1]+pupil_grid.delta[0]/2,
                           pupil_grid.y[0]-pupil_grid.delta[1]/2,
                           pupil_grid.y[-1]+pupil_grid.delta[1]/2,])
        
    return subaru_pup
    
def make_scexao_aperture(normalized=False, with_spiders=True):
    pupil_diameter = 8.2 * 0.95 # In meters
    spider_width1 = 0.20 # In meters
    spider_width2 = 0.1 # In meters
    central_obscuration_ratio = 0.30
    spider_offset = np.array((1.278/2, 0.)) # In meters
    spider_offset2 = np.array((0.5, -1.012)) # In meters
    beta = 48.4 # In degrees
    beta *= np.pi / 180
    
    # Let's add the extra features on the Subaru pupil as seen in SCExAO
    satellite1_pos = np.array((1.745, 1.43)) # In meters, positions for pup_diam = 8.2 * 0.95
    satellite2_pos = np.array((-0.485, -2.28)) # In meters, positions for pup_diam = 8.2 * 0.95
    satellite1_diam = 0.63 # In meters, positions for pup_diam = 8.2 * 0.95
    satellite2_diam = 0.63 # In meters, positions for pup_diam = 8.2 * 0.95
    mirror_edge5 = np.array((-1.5, -3.57)) # In meters, positions for pup_diam = 8.2 * 0.95
    
    if normalized:
        spider_width1 /= pupil_diameter
        spider_width2 /= pupil_diameter
        spider_offset /= pupil_diameter
        spider_offset2 /= pupil_diameter
        satellite1_pos /= pupil_diameter
        satellite2_pos /= pupil_diameter
        satellite1_diam /= pupil_diameter
        satellite2_diam /= pupil_diameter
        mirror_edge5 /= pupil_diameter
        pupil_diameter = 1.
     
    mirror_edge1 = (pupil_diameter * np.cos(beta), pupil_diameter * np.sin(beta))
    mirror_edge2 = (pupil_diameter * np.cos(beta), -pupil_diameter * np.sin(beta))
    mirror_edge3 = (-pupil_diameter * np.cos(beta), pupil_diameter * np.sin(beta))
    mirror_edge4 = (-pupil_diameter * np.cos(beta), -pupil_diameter * np.sin(beta))

    obstructed_aperture = hp.make_obstructed_circular_aperture(pupil_diameter, central_obscuration_ratio)
    
    if not with_spiders:
        return obstructed_aperture
    
    spider1 = hp.make_spider(spider_offset, mirror_edge1, spider_width1)
    spider2 = hp.make_spider(spider_offset, mirror_edge2, spider_width1)
    spider3 = hp.make_spider(-spider_offset, mirror_edge3, spider_width1)
    spider4 = hp.make_spider(-spider_offset, mirror_edge4, spider_width1)
    spider5 = hp.make_spider(spider_offset2, mirror_edge5, spider_width2) # Extra spider to bear the mask "satellite2"
    
    satellite1 = hp.aperture.circular_aperture(satellite1_diam, satellite1_pos)
    satellite1 = hp.aperture.make_obstruction(satellite1)
    satellite2 = hp.aperture.circular_aperture(satellite2_diam, satellite2_pos)
    satellite2 = hp.aperture.make_obstruction(satellite2)
    
    def func(grid):
        return obstructed_aperture(grid) * spider1(grid) * spider2(grid) *\
            spider3(grid) * spider4(grid) * spider5(grid) * satellite1(grid) *\
                satellite2(grid)
                
    return func
    


if __name__ == '__main__':
    psz = 256
    subaru_diameter = 8.2*0.95
    pup_rot = 94
    pup_rot *= np.pi / 180
    scexao = make_scexao_aperture()
    scexao = hp.aperture.make_rotated_aperture(scexao, pup_rot)
    grid = hp.make_uniform_grid([psz, psz], [subaru_diameter, subaru_diameter])
    pup_tel0 = hp.evaluate_supersampled(scexao, grid, 8)
    pup_tel = np.array(pup_tel0, dtype=float)
    pup_tel = np.reshape(pup_tel, (psz, psz))
    pup_tel = np.flipud(pup_tel)
    nring = 3 # Number of rings of the hexagon
    ring_rad = 1.075 # Radius of the 1st ring, in metre
    subpup_diam = 1.
    aper_rad = subpup_diam/2 # Radius of the sub-apertures
    holes_id = [33, 21, 15, 4] # ID of the hexagon coordinates on which the apertures are, the order correspond to the numbering of the beams.
    rot = 30
    # pupil = create_aperture_mask(pup_tel, subaru_diameter, 3, ring_rad, aper_rad, rot, holes_id, norm=False, view_pupil=True)
    
    magnification = 1480.
    gap_size = 6.e-6 # in meter
    num_rings = 3
    segment_flat_to_flat = 606.2e-6
    focal_length = 1 # m
    gap_size *= magnification
    segment_flat_to_flat *= magnification
    
    # Parameters for the simulation
    num_pix = psz
    wavelength = 638e-9
    num_airy = 20
    sampling = 4
    norm = False

    # HCIPy grids and propagator
    pupil_grid = hp.make_pupil_grid(dims=num_pix, diameter=subaru_diameter)
    
    focal_grid = hp.make_focal_grid(sampling, num_airy,
                                       pupil_diameter=subaru_diameter,
                                       reference_wavelength=wavelength,
                                       focal_length=focal_length)
    focal_grid = focal_grid.shifted(focal_grid.delta / 2)
    
    prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)
    
    aper, segments = hp.make_hexagonal_segmented_aperture(num_rings,
                                                        segment_flat_to_flat,
                                                        gap_size,
                                                        starting_ring=0,
                                                        return_segments=True)
    
    aper = hp.evaluate_supersampled(aper, pupil_grid, 1)
    segments = hp.evaluate_supersampled(segments, pupil_grid, 1)
    mems_seg_array = xs.pupil.hex_grid_coords(num_rings, segment_flat_to_flat+gap_size, rot=rot).T
    beams = [33, 21, 15, 4]
    mems_seg_array = mems_seg_array[beams]
    
    glint_mask1 = hp.aperture.circular_aperture(subpup_diam, mems_seg_array[0])(pupil_grid)
    glint_mask2 = hp.aperture.circular_aperture(subpup_diam, mems_seg_array[1])(pupil_grid)
    glint_mask3 = hp.aperture.circular_aperture(subpup_diam, mems_seg_array[2])(pupil_grid)
    glint_mask4 = hp.aperture.circular_aperture(subpup_diam, mems_seg_array[3])(pupil_grid)
    
    glint_mask = glint_mask1 + glint_mask2 + glint_mask3 + glint_mask4
    
    # plt.figure()
    # plt.title('HCIPy aperture')
    # hp.imshow_field(aper + np.ravel(pup_tel) + glint_mask, cmap='gray')
    # plt.scatter(mems_seg_array[:,0], mems_seg_array[:,1])

    # plt.figure()
    # plt.title('Total aperture')
    # hp.imshow_field(aper * np.ravel(pup_tel) * glint_mask, cmap='gray')
    # plt.scatter(mems_seg_array[:,0], mems_seg_array[:,1])

    aper = aper * np.ravel(pup_tel) * glint_mask
    # Instantiate the segmented mirror
    hsm = hp.SegmentedDeformableMirror(segments)
    
    # Make a pupil plane wavefront from aperture
    wf = hp.Wavefront(aper, wavelength)
    
    def aber_to_opd(aber_rad, wavelength):
        aber_m = aber_rad * wavelength / (2 * np.pi)
        return aber_m

    aber_rad_tt = 2.5e-7
    aber_rad_p = 1.8
    
    opd_piston = aber_to_opd(aber_rad_p, wavelength)
    
    ### Put aberrations on both SMs
    # Flatten both SMs
    hsm.flatten()
    
    ## PISTON
    for i in [28]:
        hsm.set_segment_actuators(i, opd_piston / 2 * i/10*0, aber_rad_tt, 0)

    # # HCIPy
    # plt.figure(figsize=(8,8))
    # plt.title('OPD for HCIPy SM')
    # hp.imshow_field(hsm.surface * 2, mask=aper, cmap='RdBu_r')
    # plt.colorbar()
    # plt.scatter(mems_seg_array[:,0], mems_seg_array[:,1])

    # ### Propagate to image plane
    # ## HCIPy
    # # Propagate from pupil plane through SM to image plane
    # im_pistoned_hc = prop(hsm(wf)).intensity
    
    # ### Display intensity of both cases image plane
    # plt.figure(figsize=(18, 9))
    # plt.suptitle('Image plane after SM forrandom arangement')
    # hp.imshow_field(np.log10(im_pistoned_hc), cmap='inferno', vmin=-9)
    # plt.colorbar()
    # idx = np.unravel_index(np.argmax(np.log10(im_pistoned_hc)), (160, 160))
    # grillex = np.reshape(im_pistoned_hc.grid.x, (160, 160))
    # grilley = np.reshape(im_pistoned_hc.grid.y, (160, 160))
    # plt.scatter(grilley[:,0][idx[1]], grillex[0, idx[0]], marker='o', c='r')
    # plt.title('HCIPy random arangement')

    fried_parameter = 200 # meter
    outer_scale = 20 # meter
    velocity = 10 # meter/sec
    
    Cn_squared = hp.Cn_squared_from_fried_parameter(fried_parameter, 500e-9)
    layer = hp.InfiniteAtmosphericLayer(pupil_grid, Cn_squared, outer_scale, velocity)
    plt.figure()
    hp.imshow_field(layer.phase_for(wavelength), cmap='RdBu')
    plt.colorbar()
    

    wf_turb = layer(wf)
    wf_mask = hsm(wf_turb)
    img = prop(wf_mask)
    
    plt.figure()
    hp.imshow_field(np.log10(img.intensity / img.intensity.max()), vmin=-3)
    plt.colorbar()