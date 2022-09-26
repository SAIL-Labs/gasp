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
    
This script is dedicated to parametrize and run the simulation.

The simulation relies on an isotropic and homogeneous grids: all directions have
the same extent and same step size, for a given grid.

Note of the mode field diameter (MFD): 
the values are taken from lab measurements of the
4T GLINT chip.
By using the method moments of moments, we have:
mfd_xy = (9.27e-6, 8.50e-6) metre
By using model fitting of a 2D gaussian curve, we have:
mfd_xy = (5.24e-6, 4.75e-6) metre.
The mode fields of the chip have larger tail than the Gaussian.

Note about the mode field:
it is the same for all the waveguides.

Note about the injection of wavefront:
the injection of the projected phase on the mode field of the waveguide
are defined by the overlap integral.

The alignment between the MEMS and the MLA is perfect.
Injection is biased if the subapertures are misaligned with the MEMS or if
the mode field is not entered to 0.
This first feature is not implemented and no function exist for it,
the shift of the mode field can be done by editing ``mdf_loc'' variable
(all waveguides will be shifted in the same way).

References:
    - outer scale L0: https://ui.adsabs.harvard.edu/abs/2017MNRAS.465.4931O/abstract
"""

import numpy as np
import matplotlib.pyplot as plt
import gasp_lib_generic as lib
import hcipy as hp
import os
from timeit import default_timer as timer
from scipy.interpolate import interp1d

# =============================================================================
# Settings
# =============================================================================
"""
Astrophysical and turbulence activations
"""
# To simulate the flux of a star, otherwise the incoming fluxes are equal to 1
activate_flux = True
# To simulate photon noise
activate_photon_noise = True
# To activate turbulence
activate_turbulence = True

"""
Photonics chromatism activations
"""
# Activate chromatic mode field of the waveguides
activate_chromatic_mode_field = True

"""
Detector noise activations
"""
# To simulate the detector noise
activate_detector_noise = True
# Digitise the images
activate_digitise = False

"""
Operating mode of the nuller activations
"""
# Do a scan of the fringes
activate_fringe_scan = False
# Active beams
active_beams = [False, True, False, False]
# Set the range of the fringe scan, in metre
opd_min = -3e-6
opd_max = 3e-6
# Recreate GLINt Mark II detector layout instead of assuming the output as single rows.
activate_glint_markII_layout = True
# Use GLINt Mark II combiner
activate_glint_markII_combiner = True

"""
Saving data under fits files and memory management
"""
# Save data and monitoring in FITS files
save = True
save_path = 'data/'
fits_name = 'datacube_'
# Number of fits files containing the pictures
nb_fits = 1
# Purge the list of monitoring (signals, injections, phases...) after each creation of FITS file
activate_refresh_monitors = False

"""
Sampling and RNG settings
"""
# Set the seed to an integer for repeatable simulation, to ``None'' otherwise
seed = 1
# Size in pixels of the pupil of the telescope (which will be later cropped into subpupils)
psz = 256
# Oversampling the array of the telescope for various use (e.g. bigger phase screen to mimic turbulence without wrapping)
oversz = 4

"""
Calibration settings
"""
# Set path to some calibration files
zeta_path = '/mnt/96980F95980F72D3/nulling_tricoupler/20210322_zeta_coeff_raw.hdf5'
wavecal_file = '/mnt/96980F95980F72D3/glint/gasp/ressources/20200601_wl_to_px.npy'

# =============================================================================
# Telescope, AO parameters and MEMS parameters
# =============================================================================
# Diameter of the telescope (in meter)
tdiam = 8.2 * 0.95
# Diameter of the sub-pupils (in meter)
subpup_diam = 1.

pup_rot = 94 # Angle of the Subaru pupil in GLINT wrt to MEMS
compr_pup_to_mems = 1 / 1480. # Compression factor from the pupil to the MEMS, tuned by hand
compr_mems_to_mla = 1 / 20. # Compression factor from the MEMS to the MLA (Norris+ 2020)
gap_size = 6.e-6 # in meter, gap size between the segments on the MEMS
num_rings = 3 # Number of rings of the MEMS
segment_flat_to_flat = 606.2e-6 # Side-to-side length of a segmented mirror on the MEMS

# 19.5 l/D is the cutoff frequency of the DM for 1200 modes corrected (source: Vincent)
fc_scex = 19.5
wavel_r0 = 0.5e-6  # wavelength where r0 is measured (in meters)
wavel = 1.55e-6  # Wavelength of observation (in meter)
bandwidth = 0.2e-6  # Bandwidth around the wavelength of observation (in meter)
dwl = 5e-9  # Width of one spectral channel (in meter)
oversampling_wl = 10

meter2pixel = psz / tdiam  # scale factor converting the meter-size in pixel, in pix/m

"""
Parameters for the creation of the pupil.
We assume the sub-apertures are set on a hexagonal pavement
(Hex-shaped segmented mirrors)
"""
holes_id = [28, 22, 31, 33] # ID of the hexagon coordinates on which the apertures are, the order correspond to the numbering of the beams (B1, B2...).
mirror_pistons = [wavel/8, 0, wavel/8, 0]
# mirror_pistons = np.random.uniform(-wavel, wavel, 4)
mirror_pistons = [0., 0., 0., 0.]

# =============================================================================
# Atmo parameters
# =============================================================================
# Fried parameter at wavelength wavel_r0 (in meters), the bigger, the better the seeing is
r0 = 0.16
ll = tdiam * oversz  # Physical extension of the wavefront (in meter)
# Outer scale for the model of turbulence, keep it close to infinity for Kolmogorov turbulence (the simplest form) (in meter)
L0 = 25.5 # outer scale, in metre
wind_speed = 9.8  # speed of the wind (in m/s)
wind_angle = 45  # Direction of the wind (in degree)

# =============================================================================
# Acquisition and detector parameters
# =============================================================================
fps = 2000  # frame rate (in Hz)
delay = 0.001  # delay of the servo loop (in second)
# Detector Integration Time, time during which the detector collects photon (in second)
dit = 1 / fps
timestep = 1e-3  # time step of the simulation (in second)
time_obs = 0.005  # duration of observation (in second)

# Let's define the axe of time on which any event will happened (turbulence, frame reading, servo loop)
timeline = np.around(np.arange(0, time_obs, timestep,
                               dtype=np.float32), int(-np.log10(timestep)))

# Detector is CRED-1
# read_noise = 0.7  # Read noise of the detector (in e-)
# # Quantum efficiency (probability of detection of a photon by the detector)
# QE = 0.6
# # Dark current (false event occured because of the temperature) (in e-/pix/second)
# ndark = 50
# enf = 1.  # 1.25 # Excess noise factor due to the amplification process

# # Detector is CRED-2
read_noise = 30 # Read noise of the detector (in e-)
ndark = 1500 # Dark current (false event occured because of the temperature) (in e-/pix/second)
QE = 0.85
gainsys = 0.5  # ADU/electron
offset = 2750

# Order of null:
# 12, 13, 14, 23, 24, 34
# N1, N5, N3, N2, N6, N4
# Positions of the tracks
null_labels = ['N1 (12)', 'N5 (13)', 'N3 (14)', 'N2 (23)', 'N6 (24)', 'N4 (34)']

channel_positions = np.array([33.,  53.,  72.,  92., 112., 132.,
                              151., 171., 191., 211., 230., 250.,
                              270., 290., 309., 329.])

# Reorganize the positions wrt to the combiner outputs
# Note: for N4, you have to swap the mapping
channel_positions = [channel_positions[11], channel_positions[9],
                     channel_positions[5], channel_positions[7],
                     channel_positions[1], channel_positions[14],
                     channel_positions[3], channel_positions[12],
                     channel_positions[8], channel_positions[10],
                     channel_positions[6], channel_positions[4],
                     channel_positions[15], channel_positions[13], channel_positions[2], channel_positions[0]]

track_width = 0.9 # Width of the tracks in the spatial direction

# =============================================================================
# Flux parameters
# =============================================================================
# Magnitude of the star, the smaller, the brighter.
magnitude = -4.

# Rule of thumb: 0 mag at H = 1e10 ph/um/s/m^2
# e.g. An H=5 object gives 1 ph/cm^2/s/A
MAG0FLUX = 1e10  # ph/um/s/m^2
SCEXAO_THROUGHPUT = 0.2 # Evaluated from Nem's paper about the throughput of scexao
GLINT_THROUGHPUT = 0.01 # Tuned to set limiting magnitude at 0

# =============================================================================
# MLA and injection properties
# =============================================================================
nb_ld_focal_grid = 8 # Number of pixels per resolution element in focal plane (in lambda*f/D)
nb_ld_focal_rings = 15 # Extent of the focal plane (in lambda*f/D)
# empirically found that 15 elements (or higher) biases the calculated injection by less than 2%
# (ref: Ruilier SPIE 1998, use of MFD to define the coupling in a waveguide)

microlens_diameter = 30e-6
microlens_focal_length = 95e-6
mode_field_loc = (0, 0) # Position of the mode field of the waveguide, in um
mfd_xy = (5.24e-6, 4.75e-6) # MFD (x, y axes) of the mode field of the waveguide, in metre, at 1550 nm
mode_field_theta = 0 # Transverse orientation of the mode field, in radian
chromatic_mfd_xy_path = '/mnt/96980F95980F72D3/glint/gasp/ressources/chromatic_mode_field.txt' # MFD (x, y axes) of the mode field of the waveguide, in metre, at different wavelengths

# =============================================================================
# Combiner properties
# =============================================================================
nb_apertures = 4
photometric_taps = True
waveguide_combination = 'all-in-all' # or 'pair-wise', see doc
nb_baselines = nb_apertures // 2

# =============================================================================
# Scan fringes
# =============================================================================
beam_to_scan = 0
scan_range = np.linspace(opd_min, opd_max, 25)

if activate_fringe_scan:
    magnitude = -6
    timeline = np.arange(0, scan_range.size)

# =============================================================================
# Misc
# =============================================================================
"""
Phase mask may be completely shifted.
To prevent aliasing, a new mask is created and the time to calculate the shift
is offseted.
"""
# time offset of the shift of the mask after creation of a new one
TIME_OFFSET = 0.
count_delay = 1
count_dit = 1
debug = []

# =============================================================================
# Run - no change beyond this point
# =============================================================================

# To store in the dataframes files
metadata = {'aflux': (activate_flux, 'Use of real intensities.'),
            'aturb': (activate_turbulence, 'Use of atm turbulence.'),
            'achmodfi': (activate_chromatic_mode_field, 'Use of chromatic mode field for waveguides.'),
            'adetnois': (activate_detector_noise, 'Use of detector noise.'),
            'adigitis': (activate_digitise, 'Digitize signal.'),
            'aphonois': (activate_photon_noise, 'Use of photon noise.'),
            'abeams': (', '.join(str(e) for e in active_beams), 'Opened beams.'),
            'aglint': (activate_glint_markII_combiner, 'Use of GLINT II combiner.'),
            'aglintla': (activate_glint_markII_layout, 'Use of GLINT II detectoir layout.'),
            'bandwdth': (bandwidth, 'Band width in metre.'),
            'memsmla': (compr_mems_to_mla, 'Compression factor between MEMS and MLA.'),
            'pupmems': (compr_pup_to_mems, 'Compression factor between pupil and MEMS.'),
            'dwl': (dwl, 'Width of a spectral channel.'),
            'fps': (fps, 'Frames per second.'),
            'gainsys': (gainsys, 'Detector conversion gain.'),
            'gap_size': (gap_size, 'MEMS gap size.'),
            'holes_id': (', '.join(str(e) for e in holes_id), 'ID of hex mirrors.'),
            'L0': (L0, 'Outer sacle in metre.'),
            'mag': (magnitude, ' Star magnitude.'),
            'mfd_xy': (', '.join(str(e) for e in mfd_xy), 'Mode field wvg position at 1.5 um, in metre.'),
            'ulensdia': (microlens_diameter, 'microlens diameter, in metre.'),
            'ulensfl': (microlens_focal_length, 'microlens focal length, in metre.'),
            'pistons': (', '.join(str(e) for e in mirror_pistons), 'initial piston position of hex mirrors.'),
            'modfiang': (mode_field_theta, 'Mode field wvg angle at 1.5 um, in degree.'),
            'ndark': (ndark, 'Dark current in e-/px/s.'),
            'num_ring': (num_rings, 'Number of rings on the hex segmented mirror.'),
            'offset': (offset, 'detector read-out offset.'),
            'over_wl': (oversampling_wl, 'factor of oversampling the spectral bandwidth.'),
            'oversz': (oversz, 'Oversampling the array of the telescope.'),
            'psz': (psz, 'Size in pixels of the pupil of the telescope.'),
            'pup_rot': (pup_rot, 'Angle of the Subaru pupil wrt MEMS, in degree.'),
            'QE': (QE, 'Quantum efficiency of the detector.'),
            'r0': (r0, 'Fried parameter.'),
            'ron': (read_noise, 'read-out noise.'),
            'seed':(seed, 'RNG seed.'),
            'flatflat': (segment_flat_to_flat, 'twice the apothem, in metre.'),
            'subdiam': (subpup_diam, 'diameter of sub-apertures, in metre.'),
            'tdiam': (tdiam, 'telescope diameter, in metre.'),
            'time_obs': (time_obs, 'Total observation time, in second.'),
            'timestep': (timestep, 'Time discretization of the simulation.'),
            'wavecal': (os.path.basename(wavecal_file), 'File used for wavelength calibration.'),
            'wavel': (wavel, 'Central wavelength of the bandwidth, in metre.'),
            'wavel_r0': (wavel_r0, 'Wavelength of the Fried parameter.'),
            'windangl': (wind_angle, 'Angle of the wind, in degree.'),
            'windfast': (wind_speed, 'Speed of the wind in m/s.')
            }

if activate_chromatic_mode_field:
    metadata['chmfdpth'] = (os.path.basename(chromatic_mfd_xy_path), 'To create chromatic mode fields for wvg.')
    
if activate_glint_markII_combiner:
    metadata['zetapath'] = (os.path.basename(zeta_path), 'Zeta coeff to create GLINT combiner.')
else:
    metadata['nb_aper'] = (nb_apertures, 'Number of sub-apertures.'),
    metadata['photometric_taps'] = (photometric_taps, 'Use of photometric taps.'),
    metadata['wvgconf'] = (waveguide_combination, 'Kind of combination configuration.')

# =============================================================================
# Fool-proof
# =============================================================================
mirror_pistons = np.array(mirror_pistons)
active_beams = np.array(active_beams)
mode_field_theta = np.radians(mode_field_theta)

# =============================================================================
# Simulation starts here
# =============================================================================
start = timer()

"""
Various parameters
"""
# Set the size of the MEMS proportional to the Subaru pupil
gap_size /= compr_pup_to_mems
segment_flat_to_flat /= compr_pup_to_mems

# Number of photons collected
pupil_area = np.pi / 4 * subpup_diam**2
star_photons = MAG0FLUX * 10**(-0.4*magnitude) * \
    SCEXAO_THROUGHPUT * GLINT_THROUGHPUT * pupil_area * bandwidth*1e6 * dit
print('Star photo-electrons', star_photons *
      QE, (star_photons*QE)**0.5)


"""
Create the spectral dispersion
"""
wavecal_coeff = np.load(wavecal_file)
wl0 = np.arange(wavel-bandwidth/2, wavel+bandwidth/2+dwl, dwl)
wl = lib.oversample_wavelength(wl0, oversampling_wl)

"""
Spectrum
"""
star_spectrum = np.exp(-(wl-1.5e-6)**2/(2*wl.std()**2*0.02)) + 0.5*np.exp(-(wl-1.6e-6)**2/(2*wl.std()**2*0.02))
star_spectrum = np.ones_like(wl)
star_spectrum /= star_spectrum.sum() 

"""
Beam coupler
"""
couplers = np.ones((nb_baselines, wl.size, 3, 3)) * 1/3**0.5

"""
Beam combiner
"""
if activate_glint_markII_combiner:
    zeta_dict = lib.load_zeta_file(zeta_path)
    combiner = lib.get_glint_chip_markII(wl, zeta_dict=zeta_dict)[0]
else:
    combiner = lib.create_chip(couplers, nb_apertures, \
                               photometric_taps, waveguide_combination)

"""
Create the pupil, MEMS and aperture mask
"""
pup_rot *= np.pi / 180
scexao = lib.make_scexao_aperture()
scexao = hp.aperture.make_rotated_aperture(scexao, pup_rot)
grid = hp.make_uniform_grid([psz, psz], [tdiam, tdiam])
pup_tel = hp.evaluate_supersampled(scexao, grid, 8)
pup_tel = lib.project_tel_pup_on_glint(pup_tel, psz)
pupil_grid = hp.make_pupil_grid(dims=psz, diameter=tdiam)
microlens_focal_grid = hp.make_focal_grid(nb_ld_focal_grid,
                                          nb_ld_focal_rings,
                                          pupil_diameter=microlens_diameter,
                                          focal_length=microlens_focal_length,
                                          reference_wavelength=wavel)

# Create the mems and instanciate the segments
hex_mems, segments = hp.make_hexagonal_segmented_aperture(num_rings,
                                                    segment_flat_to_flat,
                                                    gap_size,
                                                    starting_ring=0,
                                                    return_segments=True)
hex_mems = hp.evaluate_supersampled(hex_mems, pupil_grid, 5)
segments = hp.evaluate_supersampled(segments, pupil_grid, 5)
hsm = hp.SegmentedDeformableMirror(segments)
hsm.flatten() # Ensure MEMS is flatten

# Set the required pistons
for i in range(len(holes_id)):
    hsm.set_segment_actuators(holes_id[i], mirror_pistons[i], 0., 0.)
    
# Create the aperture mask
aperture_mask, sub_aper_mask, sub_aper_coords = \
    lib.make_mask_aperture(pupil_grid, holes_id, num_rings,
                           segment_flat_to_flat+gap_size,
                           subpup_diam)
translation_term = np.array([np.min(pupil_grid.x / pupil_grid.delta[0]),
                             np.min(pupil_grid.y / pupil_grid.delta[1])])
sub_aper_coords = sub_aper_coords / pupil_grid.delta - translation_term
sub_aper_mask = np.array(sub_aper_mask)

# Compute how many pixels to roll the sub apertures to the center
center = np.array([pupil_grid.x.size**0.5 / 2, pupil_grid.x.size**0.5 / 2])
shift_sub_coords = np.around(center - sub_aper_coords)
shift_sub_coords = shift_sub_coords.astype(int)
shift_sub_coords = shift_sub_coords[:,1] * psz + shift_sub_coords[:,0]

# Final aperture, to plot the overall mask only
aperture = hex_mems * np.ravel(pup_tel) * aperture_mask

# Final subaperture
aperture_surface = aperture * np.exp(1j*2*np.pi/wl[:,None] * 2 * hsm.surface[None,:])

"""
Create the phase screen
"""
if activate_turbulence:
    Cn_squared = hp.Cn_squared_from_fried_parameter(r0, wavel_r0)
    layer = hp.InfiniteAtmosphericLayer(pupil_grid, Cn_squared, L0/10, [wind_speed*np.cos(np.radians(wind_angle)), wind_speed*np.sin(np.radians(wind_angle))])
    layer0 = layer.phase_for(1).copy()


"""
Create the mode field of the waveguides
"""
x_coord = microlens_focal_grid.x[:int(microlens_focal_grid.x.size**0.5)]
y_coord = microlens_focal_grid.x[:int(microlens_focal_grid.x.size**0.5)]

if activate_chromatic_mode_field:
    chromatic_mfd_xy = np.loadtxt(chromatic_mfd_xy_path) # MFD (x, y axes) of the mode field of the waveguide, in metre, at different wavelengths
    interp_mdf_xy = interp1d(chromatic_mfd_xy[:,0], chromatic_mfd_xy[:,1:],
                              axis=-2, fill_value='extrapolate')
    interp_mdf_xy = interp_mdf_xy(wl)
    mfd_xy = interp_mdf_xy
    mode_field = []
    for waist_x, waist_y in interp_mdf_xy:
        monochroma_field = lib.create_mode_field(x_coord, y_coord,
                                                  mode_field_loc[0], mode_field_loc[1],
                                                  waist_x, waist_y, mode_field_theta)
        mode_field.append(monochroma_field)
    mode_field = np.array(mode_field)
else:
    mode_field = lib.create_mode_field(x_coord, y_coord,
                                        mode_field_loc[0], mode_field_loc[1],
                                        mfd_xy[0], mfd_xy[1],
                                        mode_field_theta)
    mode_field = mode_field.reshape((1, *mode_field.shape))

"""
Project the mode field of the waveguides in the pupil plane
"""
# Get the waist of the Gaussian beam in the pupil plane
pupil_waist = wl * microlens_focal_length / (np.pi * mfd_xy[:,0])
# Set the size of the pupil plane in waist unit
pupil_range_size = (pupil_grid.x.min() * compr_pup_to_mems * compr_mems_to_mla) * 2 / pupil_waist * (-1) 
# Set an equivalent to the radius of a circular aperture.
rad = np.pi*mfd_xy[:,0]/2/microlens_focal_grid.delta[0] # 99% of the energy is a disk of this radius
rad = np.around(rad)
rad = rad.astype(int)
# Project the mode field in the pupil plane
shape = mode_field.shape
mode_field = mode_field.reshape((shape[0], int(shape[1]**0.5), int(shape[1]**0.5)))
pupil_mode_field, pupil_mode_field_freq, pupil_mode_field_dfreq = \
    lib.sft(mode_field, pupil_range_size, mode_field.shape[1], psz, rad, True)
pupil_mode_field = pupil_mode_field.reshape(pupil_mode_field.shape[0], -1)

"""
Initialise storage lists
"""
data = []
noisy_data = []
monitor_phases = []
monitor_injections = []
monitor_i_out = []
shifts = []


i_out = np.zeros((5, wl.size))
i_out_noft = np.zeros((5, wl.size))
i_out_bi = np.zeros((5, wl.size))

t_old = timeline[0]

# Number of frames per fits
try:
    nb_frames_per_fits = timeline.size // (nb_fits - 1)
except ZeroDivisionError:
    nb_frames_per_fits = timeline.size

count_fits = 1 # Counter use to chunk and save datacube in several fits
fits_id = 1 # To generate name in fits files' names

print('Start simulation')
start_timeline = timer()
for t in timeline[:]:
    print(list(timeline).index(t)+1, '/', len(timeline))

    if activate_fringe_scan:
        mirror_pistons[beam_to_scan] = scan_range[t]
        hsm.set_segment_actuators(holes_id[beam_to_scan], mirror_pistons[beam_to_scan], 0., 0.)
        aperture_surface = aperture * np.exp(1j*2*np.pi/wl[:,None] * 2 * hsm.surface[None,:])
        
    if activate_turbulence:
        layer.evolve_until(t - t_old) 
        
        """
        Because I just keep the electric_field, hcipy interface
        ``layer(wavefronts, wl)'' or anyother does not work anymore.
        And they would not fit the multi-dimension array broadcast below.
        
        ``layer.phase_for(1)'' returns a phase screen of expression
        2 * np.pi * OPD (because lambda = 1 metre).
        """
        phase_screen = np.exp(1j * layer.phase_for(1)[None,:] / wl[:,None])
    else:
        phase_screen = np.ones((wl.size, pupil_grid.x.size))

    t_old = t

    """
    Simulate injection of the N beams in the N waveguides
    to get the injection efficiency and the phase of the guided wavefronts
    """ 
    # Injection and phase
    injections = []
    phases = []

    for i in np.arange(len(holes_id)): # Iterates over subapertures
        wavefront = aperture_surface * sub_aper_mask[i, None, :]

        """
        Some cells of ``wavefront'' are $-0+0j$ that bias the calculation of
        the phase.
        We need to force them to a positive 0.
        This operation should be done after applying the phase screen as the
        problem arises there too but we lost the full phase screen hence
        the capability to move it upon the sub-aperture.
        """
        wavefront[np.tile(sub_aper_mask[i, None, :], (wl.size, 1)) == False] = 0
        wavefront *= phase_screen

        wavefront = np.roll(wavefront, shift_sub_coords[i], axis=1)
        injection, phase = lib.calculate_injection_and_phase(wavefront, pupil_mode_field)
        injections.append(injection)
        phases.append(phase)

        
    injections = np.array(injections)
    phases = np.array(phases)
    
    # Transpose for compatibility with the code below dealing with the IO chip
    injections = injections.T
    phases = phases.T
    
    # Save some monitoring values
    monitor_injections.append(injections.mean(0)) # Save avg injection over wl
    monitor_phases.append(phases.mean(0)) # Save avg phase over wl

    """
    Creation of the guided wavefront.
    
    I don't know how hcipy calculates the intensity of the wavefront so
    for now, I am going to use the injection rate with the number of photons
    calculated in the beginning.
    """
    # Create incoming wavefronts
    a_in = np.exp(1j * phases)
    if activate_flux:
        a_in *= injections**0.5 * star_photons**0.5 #* star_spectrum[:,None]**0.5
    
    a_in[:, ~active_beams] = 0.

    # Get outcoming wavefronts
    a_out = np.einsum('ijk,ik->ij', combiner, a_in)
    
    # Spectrally bin it to add temporal coherence
    a_out = lib.bin_array(a_out, oversampling_wl)
    
    # Get the intensity of the outputs
    i_out = abs(a_out)**2
    monitor_i_out.append(i_out)
    
    if activate_glint_markII_layout:
        # Project on the detector
        col_start = [int(np.around(np.poly1d(elt)(wl0.max()*1e9))) for elt in wavecal_coeff]
        det_img, tracks = lib.create_image(344, 96, [col_start[0]], channel_positions, track_width, i_out.T[:,::-1])
        data.append(det_img)
    else:
        det_img = i_out

    noisy_img = lib.add_noise(det_img, QE, ndark*dit, gainsys, offset, read_noise,\
                          activate_photon_noise, activate_detector_noise,\
                              activate_digitise)
    noisy_data.append(noisy_img)
    
    if save:
        """
        We save the data in FITS: noisy, noise_free and just the intensities.
        The two last are the same if GLINT mark II's detector layout is not used.
        """
        fits_data = lib.chunk(count_fits, timeline.size, nb_frames_per_fits, \
                              nb_fits, noisy_data)
            
        fits_iout = lib.chunk(count_fits, timeline.size, nb_frames_per_fits, \
                              nb_fits, monitor_i_out)

        fits_injections = lib.chunk(count_fits, timeline.size, nb_frames_per_fits, \
                              nb_fits, monitor_injections)

        fits_phases = lib.chunk(count_fits, timeline.size, nb_frames_per_fits, \
                              nb_fits, monitor_phases)

        if not fits_data is None:
            fits_name2 = fits_name + '%03d.fits'%fits_id
            lib.save_in_fits(save_path, fits_name2, fits_data, metadata)
            lib.save_in_fits(save_path, 'i_out_' + '%03d.fits'%fits_id, fits_iout, metadata)
            lib.save_in_fits(save_path, 'injections_'+ '%03d.fits'%fits_id, fits_injections, metadata)
            lib.save_in_fits(save_path, 'phases_'+ '%03d.fits'%fits_id, fits_phases, metadata)
            fits_id += 1
    
        count_fits +=1
        
        if activate_refresh_monitors:
            noisy_data = []
            monitor_i_out = []
            monitor_injections = []
            monitor_phases = []
    

stop_timeline = timer()
print('End simulation')

print(stop_timeline - start_timeline)
data = np.array(data)
noisy_data = np.array(noisy_data)
monitor_injections = np.array(monitor_injections)
monitor_i_out = np.array(monitor_i_out)
monitor_phases = np.array(monitor_phases)

# =============================================================================
# Plots
# =============================================================================
plt.figure()
plt.imshow(data[0])
plt.colorbar()
plt.title('Noiseless frame')

plt.figure()
plt.imshow(noisy_data[0] - offset)
plt.colorbar()
plt.title('Noisy frame')

plt.figure()
count = 0
for k in range(6):
    plt.subplot(3, 2, k+1)
    plt.plot(timeline, monitor_i_out.mean(1)[:,count])
    plt.plot(timeline, monitor_i_out.mean(1)[:,count + 1])
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (count)')
    plt.title(null_labels[k])
    count += 2

if activate_fringe_scan:
    plt.figure()
    count = 0
    for k in range(6):
        plt.subplot(3, 2, k+1)
        plt.plot(scan_range/wavel, monitor_i_out.mean(1)[:,count])
        plt.plot(scan_range/wavel, monitor_i_out.mean(1)[:,count + 1])
        plt.grid()
        plt.xlabel(r'Scan range ($\lambda$)')
        plt.ylabel('Intensity (count)')
        plt.title(null_labels[k])
        count += 2

    plt.figure()
    count = 0
    for k in range(6):
        plt.subplot(3, 2, k+1)
        n = data[:,int(channel_positions[count])-10:int(channel_positions[count])+10,-wl0.size:]
        n = n.mean(2).sum(1)
        an = data[:,int(channel_positions[count+1])-10:int(channel_positions[count+1])+10,-wl0.size:]
        an = an.mean(2).sum(1)
        plt.plot(scan_range/wavel, n)
        plt.plot(scan_range/wavel, an)
        plt.grid()
        plt.xlabel(r'Scan range ($\lambda$)')
        plt.ylabel('Intensity (count)')
        plt.title(null_labels[k])
        count += 2    
