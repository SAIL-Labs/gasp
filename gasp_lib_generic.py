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
    - Organise the functions by alphabetical order
    
This script is to call in the running script.
All the secondary libraries can be imported here, it will simplify
the call of the functions in the main script.
"""

import numpy as np
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
import warnings
from gasp_lib_combiner import *
from gasp_lib_detector import *
from gasp_lib_optics import *

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

# def chunk(count, stop, nb_frames_per_fits, nb_fits, data_list):
#     if (count % nb_frames_per_fits == 0):
#         sub_list = data_list[-nb_frames_per_fits:]
#         return sub_list
#     elif count == stop:
#         sub_list = data_list[-(stop % (nb_fits-1)):]
#         return sub_list
#     else:
#         return None
        
def chunk(count, stop, nb_frames_per_fits, data_list):
    if (count % nb_frames_per_fits == 0):
        sub_list = data_list[-nb_frames_per_fits:]
        return sub_list
    elif count == stop:
        sub_list = data_list[-(stop % nb_frames_per_fits):]
        return sub_list
    else:
        return None    
    
def save_in_fits(path, name_file, dataframe, metadata):
    
    """
    Create the full path and check the presence of a file extension.
    """
    full_name = path + name_file
    if not full_name.lower().endswith(('.fits')):
        full_name = full_name + '.fits'

    """
    Creation of the header
    """
    with warnings.catch_warnings():
        """
        Disable display of warning about too long keyword 
        creating a HIERARCH card.
        """
        warnings.simplefilter('ignore', category=VerifyWarning) 
        hdr = fits.Header()
        for key, value in metadata.items():
            hdr[key] = value
        hdu = fits.PrimaryHDU(dataframe, header=hdr)
    
    """
    Encapsulate in FITS and save the file
    """
    hdul = fits.HDUList([hdu])
    hdul.writeto(full_name, overwrite=True)


if __name__ == '__main__':
    timeline = np.arange(0, 10)
    nb_frames_per_fits = 11#int(np.around(timeline.size / (nb_fits)))
    nb_fits = int(np.around(timeline.size / nb_frames_per_fits))
    liste = []
    out = []
    count = 1
    for i in timeline:
        liste.append(i)
        temp = chunk(count, timeline.size, nb_frames_per_fits, liste)
        if not temp is None:
            out = out + [temp]
            print(count, count//nb_frames_per_fits)
        count += 1