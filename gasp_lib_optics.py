#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Marc-Antoine Martinod

GASP: GLINT As Simulated in Python

Library to model the optics and pupils.

See ``gasp_lib_generic'' for more details.
"""

import numpy as np
import matplotlib.pyplot as plt
import hcipy as hp
from astropy.io import fits
from scipy.ndimage import rotate

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

def calculate_injection_and_phase(wavefront_field, mode_field):
    """
    Give the injection rate and the phase of the light in the waveguide.
    If arrays have more than 1D, the calculation is performed
    on the last axis.
    
    The injection rate is the scalar product of the wavefront by
    the mode field normalised by their norms.
    
    The phase is the complex argument of the scalar product.
    
    :param wavefront_field: complex field of the wavefront
    :type wavefront_field: complex array
    :param mode_field: complex field of the fundamental mode of the waveguide
    :type mode_field: complex array
    :return: injection and phase of the guided wavefront (in radian)
    :rtype: tuple

    """
    projection = np.sum(wavefront_field * np.conj(mode_field), axis=-1)
    overlap = projection / \
        np.sqrt(np.sum(np.abs(wavefront_field)**2, axis=-1) * np.sum(np.abs(mode_field)**2, axis=-1))
        
    return abs(overlap)**2, np.angle(projection) 

def calculate_injection_marechal(phs_screen, wl, geo_inj=0.8):
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

def create_aperture_mask(pup_tel, pupil_diam, nrings, ring_rad, aper_rad, rot, holes_idx=[36, 3, 33, 22], norm=False, view_pupil=False):   
    pupil = pup_tel
    csz = len(pupil)
    pscale = pupil_diam / csz
    
    mems_seg_array = hex_grid_coords(nrings, ring_rad/pscale, rot=rot).T
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

def create_mode_field(x_coord, y_coord, x0, y0, mfd_x, mfd_y, theta):
    """
    Create the complex electric mode field of the waveguide given the waist.
    The waist is the mode field radius i.e the distance of 2sigma from the top
    of the Gaussian to get a drop of amplitude to exp(-2).
    
    :param x_coord: x coordinates of the grid
    :type x_coord: 1D array
    :param y_coord: y coordinates of the grid
    :type y_coord: 1D array
    :param x0: x coordinate of the location of the Gaussian
    :type x0: float
    :param y0: y coordinate of the location of the Gaussian
    :type y0: float
    :param mfd_x: waist (i.e mode field diameter at 4sigma) of the mode field along the x axis, in the same unit as x_coord
    :type mfd_x: float
    :param mfd_y: waist (i.e mode field diameter at 4sigma) of the mode field along the y axis, in the same unit as x_coord
    :type mfd_y: float
    :param theta: Orientation of the Gaussian, in radian
    :type theta: float
    :return: complex mode field of the waveguide in a flattened array
    :rtype: 1D array

    """
    coords = (x_coord, y_coord)
    mode_field = twoD_Gaussian(coords, x0, y0, mfd_x/4, mfd_y/4, theta)
    mode_field = mode_field**0.5
    return mode_field.astype(complex)

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

def hex_grid_coords(nr, radius, rot):
    """
    Credit: XAOSIM package (F. Martinache)
    https://github.com/fmartinache/xaosim
    
    Returns a 2D array of real x,y coordinates for a regular
    hexagonal grid that fits within a hexagon.    
    
    :param nr: number of rings
    :type nr: integer
    :param radius: the radius of a ring
    :type radius: float
    :param rot: rotation angle of the hex grid, in degrees
    :type rot: float
    :return: coordinates of the regular hexagonal grid.
    :rtype: 2D-array

    """
    rotd = rot * np.pi / 180
    RR = np.array([[np.cos(rotd), -np.sin(rotd)],
                   [np.sin(rotd),  np.cos(rotd)]])

    ij0 = np.linspace(-nr, nr, 2*nr+1)
    ii, jj = np.meshgrid(ij0, ij0)
    xx = radius * (ii + 0.5 * jj)
    yy = radius * jj * np.sqrt(3)/2
    cond = np.abs(ii + jj) <= nr
    return RR.dot(np.array((xx[cond], yy[cond])))

def make_mask_aperture(pupil_grid, segments, num_rings, hex_rad, rot, subpup_diam):
    """
    Create an aperture mask to put on GLINT's segmented mirror.

    :param pupil_grid: grid on which the mask is created
    :type pupil_grid: hcipy-pupil object
    :param segments: Segments of the segmented mirror to keep through the mask
    :type segments: list-like
    :param num_rings: number of rings on the segmented mirror
    :type num_rings: int
    :param hex_rad: radius of the hexagonal segment
        (center-to-flat+gap between segments), in same unit as ``pupil_grid''
    :type hex_rad: float
    :param rot: rotation angle of the hex grid, in degrees
    :type rot: float
    :param subpup_diam: diameter of the sub-aperture, in same unit as ``pupil_grid''
    :type subpup_diam: float
    :return: aperture mask, single apertures for injection calculation purposes\
        and coordinates of the apertures on the MEMS
    :rtype: tuple

    """
    mems_seg_array = hex_grid_coords(num_rings, hex_rad, rot=rot).T
    mems_seg_array = mems_seg_array[segments]
    
    sub_aper = []
    for i in range(len(segments)):
        mask = hp.aperture.circular_aperture(subpup_diam, mems_seg_array[i])(pupil_grid)
        mask = np.array(mask, dtype=bool)
        sub_aper.append(mask)
        try:
            mask_aper = mask_aper + mask
        except NameError:
            mask_aper = mask
    
    return mask_aper, sub_aper, mems_seg_array

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

def make_scexao_aperture(normalized=False, with_spiders=True):
    """
    Build the scexao aperture. Unlike a FITS file, it is much easier to scale and rotate.
    
    :param normalized: normalise all dinensions by the diameter of the pupil, defaults to False
    :type normalized: bool, optional
    :param with_spiders: create the spiders, defaults to True
    :type with_spiders: bool, optional
    :return: return the scexao pupil
    :rtype: hcipy pupil-type object

    """
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

def make_scexao_pupil(path, rot_angle, flip_ud, plotting=False):
    """
    Load a FITS file of the SCEXAO aperture and adapt it to the simulation.
    
    :param path: path to the FITS file of the scexao aperture
    :type path: str
    :param rot_angle: rotation angle in degree
    :type rot_angle: float
    :param flip_ud: flip the pupil model upside-down
    :type flip_ud: bool
    :param plotting: plot the pupil to check if it behaves accordingly, defaults to False
    :type plotting: bool, optional
    :return: scexao pupil
    :rtype: array

    """
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

def project_tel_pup_on_glint(pup_tel, psz):
    """
    The Subaru pupil is rotated wrt to GLINT and is reflected once.
    
    :param pup_tel: pupil of the telescope to project
    :type pup_tel: hcipy Field object or array
    :param psz: size of the unflatten pupil
    :type psz: int
    :return: projected telescope pupil
    :rtype: 2D-array

    """
    pup_tel = np.array(pup_tel, dtype=float)
    pup_tel = np.reshape(pup_tel, (psz, psz))
    pup_tel = np.flipud(pup_tel)
    
    return pup_tel

def sft(A2, m, NA, NB, rad, inv):
    """
    Slow Foutier-Transform: keep separated the scales of the object
    and image planes
    which are normally linked in a Fast-Fourier Transform
    The theory of the SFT is described in:
        http://adsabs.harvard.edu/abs/2007OExpr..1515935S
        
    The parameter ``m'' defines the unit of the delivered frequency scale.
    If it is the number of lambda/D, the frequency scale unit is lambda/D.
    And the size of the FoV of the transformed signal is spans from -lambda/D to + lambda/D.
    It is important to keep ``m'' non-dimensional as ``rad'' is in non-dimensional unit.
    
    :param A2: array from which we want the Fourier Transform.
        The dimension must be (..., N, N) with N the size of the array.
        First dimensions can be wavelength, for example.
    :type A2: array
    :param m: Range of the FoV of the plane of destination
    :type m: float
    :param NA: Number of column of A2
    :type NA: int
    :param NB: Number of column of the FT of A2
    :type NB: int
    :param rad: radius of the aperture, in pixel
    :type rad: int
    :param inv: if ``True'', it performs the invert Fourier transform
        If ``"false'', it  performs the Fourier transform
    :type inv: bool
    :return: Fourier transform of A2, frequency scale and step of the
    frequency scale. The unit of the scale is lambda/D
    :rtype: tuple

    """
    """
    Slow Foutier-Transform: keep separated the scales of the object
    and image planes
    which are normally linked in a Fast-Fourier Transform
    The theory of the SFT is described in:
        http://adsabs.harvard.edu/abs/2007OExpr..1515935S

    :param A2: array from which we want the Fourier Transform.
        The dimension must be (..., N, N) with N the size of the array.
        First dimensions can be wavelength, for example.
    :type A2: array
    :param wl: wavelength, in meter
    :type wl: float
    :param diam: diameter of the aperture, in meter
    :type diam: float
    :param pscale: plate scale in mas/pixel
    :type pscale: float
    :param NA: Number of column of A2
    :type NA: int
    :param NB: Number of column of the FT of A2
    :type NB: int
    :param rad: radius of the aperture, in pixel
    :type rad: int
    :param sign: sign of the FT (-1 for the FT, +1 for the inverse FT)
    :type sign: int
    :return: Fourier transform of A2, frequency scale and step of the
    frequency scale. The unit of the scale is lambda/D
    :rtype: tuple

    """
    A2 = np.asarray(A2)
    m = np.asarray(m)

    m2 = NA / (rad*2)
    coeff = m*m2/(NA*NB)

    X = np.zeros((m.size, 1, NA))
    U = np.zeros((m.size, NB))

    stepX = m2/NA
    range_NA = np.arange(NA)-NA/2.
    X[:, 0, :] = range_NA[None,:] * stepX[:,None]
    U[:] = (m[:, None]/NB)*(np.arange(NB)[None, :]-NB/2.)
    U = U.reshape((m.size, 1, NB))
    stepU = m/NB

    if inv:
        sign = 1.0
    else:
        sign = -1.0

    A1 = np.exp(sign * 2j*np.pi *
                np.matmul(np.transpose(U, axes=(0, -1, -2)), X))
    A3 = np.exp(sign * 2j*np.pi *
                np.matmul(np.transpose(X, axes=(0, -1, -2)), U))

    B1 = np.matmul(A1, A2)
    B = np.matmul(B1, A3)
    C = B * coeff[:, None, None]

    return C, U, stepU

def twoD_Gaussian(coords, xo, yo, sigma_x, sigma_y, theta):
    """
    Create a 2D gaussian.

    :param coords: (x, y) coordinates of the grid
    :type coords: tuple-like
    :param xo: x coordinate of the location of the Gaussian
    :type xo: float
    :param yo: y coordinate of the location of the Gaussian
    :type yo: float
    :param sigma_x: Scale factor of the Gaussian along the x axis
    :type sigma_x: float
    :param sigma_y: Scale factor of the Gaussian along the y axis
    :type sigma_y: float
    :param theta: Orientation of the Gaussian, in radian
    :type theta: float
    :return: 2D gaussian profile in a flattened array
    :rtype: array

    """
    xx, yy = np.meshgrid(*coords)
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    return np.ravel(np.exp( - (a*((xx-xo)**2) + 2*b*(xx-xo)*(yy-yo) + c*((yy-yo)**2))))

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

