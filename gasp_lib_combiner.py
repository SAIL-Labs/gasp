#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marc-Antoine Martinod

GASP: GLINT As Simulated in Python

Library to model the optics, the pupils and the atmosphere.

See ``gasp_lib_generic'' for more details.
"""

import numpy as np
import h5py
from scipy.interpolate import interp1d
from itertools import combinations

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

def get_tricoupler_coupling_coeff(coeff_file, wl):
    """
    This function reads files containing the squared coupling coefficients inside a tricoupler.
    These coefficients consists of the ratio of light in the three outputs divided
    by their sum, for an injection in one waveguide after the other.
    
    The file is assumed to follow the HDF5 format with the following content:
        - the wavelength axis in um
    and at least one of these
        - a (3, wl size) array of the coupling coefficient gotten from injection in the left, named ``injection_left''
        - a (3, wl size) array of the coupling coefficient gotten from injection in the centre, named ``injection_centre''
        - a (3, wl size) array of the coupling coefficient gotten from injection in the right, named ``injection_right''

    Missing `injection' will be extrapolated upon the assumption of a full symmetry of the tricoupler.

    :param coeff_file: file containing the coupling coefficients
    :type coeff_file: str
    :param wl: wavelength axis to which the coupling ratios are evaluated
    :type wl: 1D-array
    :return: coupling coefficients of the tricoupler
    :rtype: 9-tuple

    """
    file_coeff = h5py.File(coeff_file, 'r')
    file_content = list(file_coeff.keys())
    wl_scale = np.array(file_coeff['wl_scale']) * 1e-6 # Convert wavelength in metre
    
    # Load the coefficients and check the missing ones
    no_injection_left = False
    no_injection_right = False
    no_injection_centre = False
    
    if 'injection_left' in file_content:
        injection_left = np.array(file_coeff['injection_left'])
        inj_left = np.array([interp1d(wl_scale, elt, 'quadratic')(wl) for elt in injection_left])

    else:
        no_injection_left = True

    if 'injection_centre' in file_content:
        injection_centre = np.array(file_coeff['injection_centre'])
        inj_centre = np.array([interp1d(wl_scale, elt, 'quadratic')(wl) for elt in injection_centre])
    else:
        no_injection_centre = True

    if 'injection_right' in file_content:
        injection_right = np.array(file_coeff['injection_right'])
        inj_right = np.array([interp1d(wl_scale, elt, 'quadratic')(wl) for elt in injection_right])

    else:
        no_injection_right = True
    
    # Now create the kappa coefficients
    no_injection = np.array([no_injection_left, no_injection_centre, no_injection_right])
    if np.all(no_injection):
        return 'No data found. Check content, structure and format.'
    
    if np.any(no_injection): # At least one injection is missing
        has_injection = np.where(no_injection == False)[0][0]
        
        if has_injection == 0: # Left injection is present, used to get the other coupling ratios
            kappa_LL = inj_left[0]
            kappa_LC = inj_left[1]
            kappa_LR = inj_left[2]
            
            kappa_CL = inj_left[1]
            kappa_CC = inj_left[0]
            kappa_CR = inj_left[2]
            
            kappa_RL = inj_left[2]
            kappa_RC = inj_left[1]
            kappa_RR = inj_left[0]
            
        elif has_injection == 1: # Centre injection is present, used to get the other coupling ratios
            kappa_LL = inj_centre[1]
            kappa_LC = inj_centre[0]
            kappa_LR = inj_centre[2]
            
            kappa_CL = inj_centre[0]
            kappa_CC = inj_centre[1]
            kappa_CR = inj_centre[2]
            
            kappa_RL = inj_centre[0]
            kappa_RC = inj_centre[2]
            kappa_RR = inj_centre[1]

        else: # Right injection is present, used to get the other coupling ratios
            kappa_LL = inj_right[2]
            kappa_LC = inj_right[1]
            kappa_LR = inj_right[0]
            
            kappa_CL = inj_right[1]
            kappa_CC = inj_right[2]
            kappa_CR = inj_right[0]
            
            kappa_RL = inj_right[0]
            kappa_RC = inj_right[1]
            kappa_RR = inj_right[2]

        
    else:
        kappa_LL = inj_left[0]
        kappa_LC = inj_left[1]
        kappa_LR = inj_left[2]

        kappa_CL = inj_centre[0]
        kappa_CC = inj_centre[1]
        kappa_CR = inj_centre[2]

        kappa_RL = inj_right[0]
        kappa_RC = inj_right[1]
        kappa_RR = inj_right[2]
        
    return kappa_LL, kappa_LC, kappa_LR, \
        kappa_CL, kappa_CC, kappa_CR, \
            kappa_RL, kappa_RC, kappa_RR

def create_single_tricoupler(kappa_LL, kappa_LC, kappa_LR, \
                             kappa_CL, kappa_CC, kappa_CR, \
                             kappa_RL, kappa_RC, kappa_RR):
    """
    Return a tricoupler based on the coupling ratios.
    The model assumes at least an axial symmetry.
    The symmetry axis goes through the central (nulled) waveguide and is perpendicular
    to the direction made by the two other waveguides.

    :param kappa_LL: Spectral intensity ratio between Left output over Left input
    :type kappa_LL: array
    :param kappa_LC: Spectral intensity ratio between Centre output over Left input
    :type kappa_LC: array
    :param kappa_LR: Spectral intensity ratio between Right output over Left input
    :type kappa_LR: array
    :param kappa_CL: Spectral intensity ratio between Left output over Centre input
    :type kappa_CL: array
    :param kappa_CC: Spectral intensity ratio between Centre output over Centre input
    :type kappa_CC: array
    :param kappa_CR: Spectral intensity ratio between Right output over Centre input
    :type kappa_CR: array
    :param kappa_RL: Spectral intensity ratio between Left output over Right input
    :type kappa_RL: array
    :param kappa_RC: Spectral intensity ratio between Centre output over Right input
    :type kappa_RC: array
    :param kappa_RR: Spectral intensity ratio between Right output over Right input
    :type kappa_RR: array
    :return: Tricoupler model with axes (wavelength, outputs, inputs)
    :rtype: 3D-array

    """
    
    T1 = kappa_LL**0.5
    C1 = kappa_CL**0.5
    C2 = kappa_RL**0.5
    
    phi2 = np.arccos(-C1**2 / (2 * T1 * C2))
    phi1 = np.arccos(C1 / (2 * (T1**2 - C1**2 + C2**2)**0.5)) + np.arctan((T1**2 * C1**2 - C2**4/4)**0.5 / (T1**2 - C1**2/2))

    tricoupler = np.array([[kappa_LL**0.5                   , kappa_CL**0.5 * np.exp(1j*phi1), kappa_RL**0.5 * np.exp(1j*phi2)],
                           [kappa_LC**0.5 * np.exp(1j*phi1) , kappa_CC**0.5                  , kappa_RC**0.5 * np.exp(1j*phi1)],
                           [kappa_LR**0.5 * np.exp(1j*phi2) , kappa_CR**0.5 * np.exp(1j*phi1), kappa_RR**0.5]], dtype=complex)
    
    tricoupler = np.transpose(tricoupler, axes=(2, 0, 1))
    
    return tricoupler

def create_chip(couplers, nb_apertures, photometric_taps, configuration='all-in-all'):
    pass

def create_chip_all_in_all(couplers, nb_apertures, photometric_taps):
    """
    Create a chip based on the all-in-all configuration i.e. all beams interfere
    with each other.
    
    The function works with any coupler among 2x2 (directional coupler), 3x3 and 3x2 (tricouplers)
    and an arbitrary number of apertures.
    However, it is not suitable to make kernel as the 3x3 is reshaped as a 3x2.

    The chip has two functions: 
        - splitting the light to make them interfere with each other and create photometric taps.
            We use matrix formalism for that and rows exist for photometric taps, they
            will be put to 0 if they are not simulated.
        - combine the ligth
        
    The splitters are assumed achromatic and equally split the light among all
    the new split waveguides

    :param couplers: ensemble of couplers to put in the chip. The array **must**\
        follow the following shape: (number of couplers, wavelength, outputs, inputs).
        The values inside must be in the same unit as incoming wavefront, not intensities.
    :type couplers: complex array-like
    :param nb_apertures: Number of apertures to combine
    :type nb_apertures: int
    :param photometric_taps: If True, simulate photometric outputs. \
        If not, the corresponding rows stay to 0.
    :type photometric_taps: bool
    :return: matrix representing the photonic chip. Its shape is (wavelength, outputs, inputs)
    :rtype: array

    """
    
    # nb_apertures = 4
    # nb_baselines = nb_apertures*(nb_apertures-1)//2
    # couplers = np.ones((nb_baselines, 11, 3, 3)) * np.exp(1j*np.pi/3)
    # photometric_taps = True

    nb_baselines = nb_apertures*(nb_apertures-1)//2
    split_ratio = np.ones(couplers.shape[1])
    
    if photometric_taps:
        split_ratio[:] = 1 / nb_apertures
    else:
        split_ratio[:] = 1 / (nb_apertures - 1)
    
    """
    Splitters matrix has a number of rows equal to
    nb_apertures*(nb_apertures-1) outputs going to the combiners + nb_apertures photometric taps
    and nb_apertures columns (inputs)
    """
    splitters = np.zeros((nb_apertures*(nb_apertures-1)+nb_apertures, nb_apertures, couplers.shape[1]))
    
    pairs = np.array([elt for elt in combinations(range(4), 2)]) # list the pairs of beams
    permutations = np.hstack((pairs, pairs[:,::-1]))
    permutations = permutations.reshape((-1, 2)) # list the inputs of every pairs (beam 1 going to combiner A, beam 2 too, beam 1 going to combiner 2, beam 3 too, etc.)
    
    """
    We iteratively build the matrix of the splitters.
    The shape is (wavelength, outputs, inputs)
    """
    for k in range(nb_apertures):
        loc = np.where(permutations == k)
        idx = loc[0][loc[1] == 0]
        splitters[idx, k] = split_ratio
        
        if photometric_taps:
            splitters[splitters.shape[0]-(nb_apertures-k), k] = split_ratio
    
    splitters = np.transpose(splitters, (2, 0, 1)) # To get the shape (wavelength, outputs, inputs)
    
    splitters = splitters**0.5 # We work with complex amplitude, not intensities
    
    combiners = _assemble_couplers(couplers, nb_baselines, nb_apertures)

    """
    The chip is the matrix product of the splitters and the combiners.
    Its shape is (wavelength, outputs, inputs).
    """
    chip = combiners@splitters
    
    return chip


def create_chip_pairwise(couplers, nb_apertures, photometric_taps, **kwargs):
    """
    Create a chip based on the pair-wise configuration i.e. a beam can only interfere
    with another one so that the number of baselines is half the number of beams.
    
    The function works with any coupler among 2x2 (directional coupler), 3x3 and 3x2 (tricouplers)
    and an arbitrary number of apertures.
    However, it is not suitable to make kernel as the 3x3 is reshaped as a 3x2.

    The chip has two functions: 
        - splitting the light to make them interfere with each other and create photometric taps.
            We use matrix formalism for that and rows exist for photometric taps, they
            will be put to 0 if they are not simulated.
        - combine the ligth
        
    The splitters are assumed achromatic and split the light 50/50 
    but the user can choose different and chromatic coefficients
    
    :param couplers: ensemble of couplers to put in the chip. The array **must**\
        follow the following shape: (number of couplers, wavelength, outputs, inputs).
        The values inside must be in the same unit as incoming wavefront, not intensities.
    :type couplers: complex array-like
    :param nb_apertures: Number of apertures to combine
    :type nb_apertures: int
    :param photometric_taps: If True, simulate photometric outputs. \
        If not, the corresponding rows stay to 0.
    :type photometric_taps: bool
    :param **kwargs: keyword ``split_ratio'' can include a 1D-array of size the number of spectral channels\
        of splitting ratios $\alpha$ quantifying the **intensity** \
            sent to the recombiner. Photometric split will be deduced as $1-\alpha$.
    :type **kwargs: array
    :return: matrix representing the photonic chip. Its shape is (wavelength, outputs, inputs)
    :rtype: array

    """
    # nb_apertures = 4
    # nb_baselines = nb_apertures // 2
    # couplers = np.ones((nb_baselines, 11, 2, 2))
    # photometric_taps = True

    nb_baselines = nb_apertures // 2
    """
    Creating the matrix of the splitters.
    """
    split_ratio = np.ones(couplers.shape[1])
    
    if photometric_taps:
        if 'split_ratio' in kwargs.keys():
            split_ratio[:] = kwargs['split_ratio']
        else:
            split_ratio[:] = 0.5
    else:
        split_ratio[:] = 1.
    

    splitters = np.array([[split_ratio if i == k else np.zeros_like(split_ratio) \
                           for i in range(nb_apertures)] \
                          for k in range(nb_apertures)] +\
                         [[1 - split_ratio if i == k else np.zeros_like(split_ratio) \
                                                for i in range(nb_apertures)] \
                                               for k in range(nb_apertures)])
    splitters = np.transpose(splitters, (2, 0, 1)) # To get the shape (wavelength, outputs, inputs)
    
    splitters = splitters**0.5

    combiners = _assemble_couplers(couplers, nb_baselines, nb_apertures)
    
    """
    The chip is the matrix product of the splitters and the combiners.
    Its shape is (wavelength, outputs, inputs).
    """
    chip = combiners@splitters
    
    return chip

def _assemble_couplers(couplers, nb_baselines, nb_apertures):
    """
    Functions called by ``create_chip_all_in_all'' and ``create_chip_pairwise''
    to create the set of combiners inside the chip.
    
    :param couplers: ensemble of couplers to put in the chip. The array **must**\
        follow the following shape: (number of couplers, wavelength, outputs, inputs).
        The values inside must be in the same unit as incoming wavefront, not intensities.
    :type couplers: complex array-like
    :param nb_baselines: Number of baselines
    :type nb_baselines: int
    :param nb_apertures: Number of apertures to combine
    :type nb_apertures: int
    :return: Spectral matrix of the combiners inside the chip.
    :rtype: array

    """
    
    """
    Creating the combiner part.
    We first identify the coupler (2x2, 3x3 or 3x2). If it is a 3x3, we remove 
    the central column we never use here.
    
    Then we build the matrix as a set of blocks.
    """    
    if couplers.shape[2] == 3 and couplers.shape[3] == 3: # If coupler is tricoupler with 3 inputs
        couplers = couplers[:,:,:,[0,2]] # Remove the central column which is never used

    """
    We build the matrix as a set of blocks. It will be an identity block matrix.
    
    We need to create blocks of zeros to fill the matrix but the diagonal.
    Their shape is (wavelength, outputs, inputs)
    """
    zero_blocks = np.zeros((couplers.shape[1], couplers.shape[2], couplers.shape[3]))
    
    """
    Creation of a list of the blocks to be called by ``np.block''
    """
    blocks = [[couplers[i] if i == k else zero_blocks for i in range(nb_baselines)] for k in range(nb_baselines)]
    combiners = np.block(blocks)
    
    """
    Padding the matrix to add columns and rows for the photometric taps., even if not used (i.e full of 0).
    """
    combiners = np.pad(combiners, ((0,0), (0, nb_apertures), (0, nb_apertures)))
    combiners[:, -nb_apertures:, -nb_apertures:] = np.eye(nb_apertures)   
    
    return combiners