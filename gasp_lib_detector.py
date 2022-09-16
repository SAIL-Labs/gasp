#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marc-Antoine Martinod

GASP: GLINT As Simulated in Python

Library to model the detector.

See ``gasp_lib_generic'' for more details.
"""

import numpy as np

"""
These functions create the image as sampled on the detector and simulate the noises:
    - photon noise and dark current (Poisson distribution)
    - Read-out noise (Normal distribution)
    - Digitisation to convert pixels' values in 16-bit integers
"""

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
    flux = outputs[:,:,None] * tracks # Shape (tracks, spectral channels, spatial extent)
    img = np.zeros((ysz, xsz))
    for k in range(flux.shape[0]): # Iterate over outputs
        img += np.pad(flux[k].T, ((0,0),(col_start[k], xsz - (col_start[k]+nb_wl))),\
                             mode='constant', constant_values=0.)
        
    return img, tracks

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
    

    


