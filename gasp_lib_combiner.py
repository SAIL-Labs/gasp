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
