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

"""

import numpy as np
import matplotlib.pyplot as plt
import gasp_lib as lib
import hcipy as hp
from timeit import default_timer as timer
from scipy.interpolate import interp1d

# =============================================================================
# Settings
# =============================================================================
# To save data
save = False
# To simulate photon noise
activate_photon_noise = False
# To simulate the detector noise
activate_detector_noise = True
# Digitise the images
activate_digitise = False
# To simulate the flux of a star, otherwise the incoming fluxes are equal to 1
activate_flux = False
# Use an achromatic phase mask instead of air-delaying (chromatic) one beam with respect to the other
activate_achromatic_phase_shift = False
# To activate turbulence
activate_turbulence = False
# Do a scan of the fringes
activate_fringe_scan = True
# Active beams
active_beams = [True, True, True, True]
# Activate chromatic mode field of the waveguides
activate_chromatic_mode_field = True

# Set the seed to an integer for repeatable simulation, to ``None'' otherwise
seed = 1

# Size in pixels of the pupil of the telescope (which will be later cropped into subpupils)
psz = 256
# Oversampling the array of the telescope for various use (e.g. bigger phase screen to mimic turbulence without wrapping)
oversz = 4

zeta_path = '/mnt/96980F95980F72D3/nulling_tricoupler/20210322_zeta_coeff_raw.hdf5'
# zeta_path = '/mnt/96980F95980F72D3/glint/simulation/zeta_coeff.hdf5'

opd_min = -10e-6
opd_max = 10e-6

# =============================================================================
# Telescope, AO parameters and MEMS parameters
# =============================================================================
# Diameter of the telescope (in meter)
tdiam = 8.2 * 0.95
# Diameter of the sub-pupils (in meter)
subpup_diam = 1.
micro_lens_diam = 30e-6 # Diameter of the micro-lens for injecting light in the chi[]
micro_lens_focus = 96e-6 # Focal length of the micro-lens

pup_rot = 94 # Rotation of the Subaru pupil in GLINT wrt to MEMS
rot_hex_grid = 30 # in degree, rotation of the hex grid of the MEMS
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
ao_correc = 8.  # How well the AO flattens the wavefront

"""
Parameters for the creation of the pupil.
We assume the sub-apertures are set on a hexagonal pavement
(Hex-shaped segmented mirrors)
"""
holes_id = [33, 21, 15, 4] # ID of the hexagon coordinates on which the apertures are, the order correspond to the numbering of the beams (B1, B2...).
mirror_pistons = [wavel/4, 0, wavel/4, 0]
mirror_pistons = [0., 0., 0., 0.]

# =============================================================================
# Atmo parameters
# =============================================================================
# Fried parameter at wavelength wavel_r0 (in meters), the bigger, the better the seeing is
r0 = 0.16
ll = tdiam * oversz  # Physical extension of the wavefront (in meter)
# Outer scale for the model of turbulence, keep it close to infinity for Kolmogorov turbulence (the simplest form) (in meter)
L0 = 1e7 # outer scale, in metre
wind_speed = 9.8  # speed of the wind (in m/s)
angle = 45  # Direction of the wind


# =============================================================================
# Acquisition and detector parameters
# =============================================================================
fps = 2000  # frame rate (in Hz)
delay = 0.001  # delay of the servo loop (in second)
# Detector Integration Time, time during which the detector collects photon (in second)
dit = 1 / fps
timestep = 1e-4  # time step of the simulation (in second)
time_obs = 0.1  # duration of observation (in second)

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

sigma = 0.9 # Width of the tracks in the spatial direction

# =============================================================================
# Flux parameters
# =============================================================================
# Magnitude of the star, the smaller, the brighter.
magnitude = 5

# Rule of thumb: 0 mag at H = 1e10 ph/um/s/m^2
# e.g. An H=5 object gives 1 ph/cm^2/s/A
MAG0FLUX = 1e10  # ph/um/s/m^2
SCEXAO_THROUGHPUT = 0.2

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
chromatic_mfd_xy = np.loadtxt('/mnt/96980F95980F72D3/glint/gasp/chromatic_mode_field.txt') # MFD (x, y axes) of the mode field of the waveguide, in metre, at different wavelengths


# =============================================================================
# Scan fringes
# =============================================================================
beam_to_scan = 0
scan_range = np.linspace(opd_min, opd_max, 1001)

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
# Fool-proof
# =============================================================================
mirror_pistons = np.array(mirror_pistons)
active_beams = np.array(active_beams)

# =============================================================================
# Run
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
    SCEXAO_THROUGHPUT * pupil_area * bandwidth*1e6 * dit
print('Star photo-electrons', star_photons *
      QE, (star_photons*QE)**0.5)

"""
Create the spectral dispersion
"""
wl0 = np.arange(wavel-bandwidth/2, wavel+bandwidth/2+dwl, dwl)
wl = lib.oversample_wavelength(wl0, oversampling_wl)

"""
Beam combiner
"""
zeta_dict = lib.load_zeta_file(zeta_path)
combiner, splitter_interfA_list, splitter_interfB_list, kappa_AB_list, kappa_BA_list, splitter_photoA_list, splitter_photoB_list = lib.get_glint_chip_markII(wl, zeta_dict=zeta_dict)

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
hex_mems = hp.evaluate_supersampled(hex_mems, pupil_grid, 1)
segments = hp.evaluate_supersampled(segments, pupil_grid, 1)
hsm = hp.SegmentedDeformableMirror(segments)
hsm.flatten() # Ensure MEMS is flatten

# Set the required pistons
for i in range(len(holes_id)):
    hsm.set_segment_actuators(holes_id[i], mirror_pistons[i], 1e-3, 1e-3)

# Create the aperture mask
aperture_mask, sub_aper_mask, sub_aper_coords = \
    lib.make_mask_aperture(pupil_grid, holes_id, num_rings,
                           segment_flat_to_flat+gap_size,
                           rot_hex_grid, subpup_diam)
translation_term = np.array([np.min(pupil_grid.x / pupil_grid.delta[0]),
                             np.min(pupil_grid.y / pupil_grid.delta[1])])
sub_aper_coords = sub_aper_coords / pupil_grid.delta - translation_term

# Compute how many pixels to roll the sub apertures to the center
center = np.array([pupil_grid.x.size**0.5 / 2, pupil_grid.x.size**0.5 / 2])
shift_sub_coords = np.around(center - sub_aper_coords)
shift_sub_coords = shift_sub_coords.astype(int)
shift_sub_coords = shift_sub_coords[:,1] * psz + shift_sub_coords[:,0]

# Final aperture
aperture = hex_mems * np.ravel(pup_tel) * aperture_mask

# Final subaperture
sub_apertures = [hex_mems * np.ravel(pup_tel) * elt for elt in sub_aper_mask]
sub_apertures = [hp.Wavefront(elt) for elt in sub_apertures]

"""
Create the phase screen
"""
if activate_turbulence:
    Cn_squared = hp.Cn_squared_from_fried_parameter(r0, wavel_r0)
    layer = hp.InfiniteAtmosphericLayer(pupil_grid, Cn_squared, L0/10, wind_speed)


"""
Create the mode field of the waveguides
"""
x_coord = microlens_focal_grid.x[:int(microlens_focal_grid.x.size**0.5)]
y_coord = microlens_focal_grid.x[:int(microlens_focal_grid.x.size**0.5)]

if activate_chromatic_mode_field:
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

start_timeline = timer()
for t in timeline[:1]:
    if activate_fringe_scan:
        mirror_pistons[beam_to_scan] = scan_range[t]
        
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
    
    t_old = t

    """
    Simulate injection of the N beams in the N waveguides
    to get the injection efficiency and the phase of the guided wavefronts
    """ 
    # Injection and phase
    injections = []
    phases = []

    for i in np.arange(len(holes_id)): # Iterates over subapertures
        wavefront = sub_apertures[i].electric_field * phase_screen
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
        a_in *= injections.T**0.5 * star_photons**0.5
    
    a_in[:, ~active_beams] = 0.

    # Get outcoming wavefronts
    a_out = np.einsum('ijk,ik->ij', combiner, a_in)
    
    # Spectrally bin it to add temporal coherence
    a_out = lib.bin_array(a_out, oversampling_wl)
    
    # Get the intensity of the outputs
    i_out = abs(a_out)**2
    monitor_i_out.append(i_out)
    
    # Project on the detector...
    det_img, tracks = lib.create_image(344, 96, [96-wl0.size], channel_positions, sigma, i_out.T)
    data.append(det_img)

    # ...and add some noise
    noisy_img = lib.add_noise(det_img, ndark, gainsys, offset, read_noise,\
                          activate_photon_noise, activate_detector_noise,\
                              activate_digitise)
        
    noisy_data.append(noisy_img)

stop_timeline = timer()

print(stop_timeline - start_timeline)
monitor_i_out = np.array(monitor_i_out)
data = np.array(data)
noisy_data = np.array(noisy_data)

plt.figure()
plt.imshow(det_img, origin='upper')

# # plt.figure()
# # plt.plot(zeta_dict['wl_scale'], zeta_dict['b1null1'])
# # plt.plot(wl0*1e9, i_out[:,0] / i_out[:,12])

# if activate_fringe_scan:
#     monitor_i_out = monitor_i_out.mean(1)
    
#     plt.figure()
#     count = 0
#     for k in range(6):
#         plt.subplot(3, 2, k+1)
#         plt.plot(scan_range/wavel, monitor_i_out[:,count])
#         plt.plot(scan_range/wavel, monitor_i_out[:,count + 1])
#         plt.grid()
#         plt.xlabel('Scan range')
#         plt.ylabel('Intensity (count)')
#         plt.title(null_labels[k])
#         count += 2

#     plt.figure()
#     count = 0
#     for k in range(6):
#         plt.subplot(3, 2, k+1)
#         n = data[:,int(channel_positions[count])-10:int(channel_positions[count])+10,-wl0.size:]
#         n = n.mean(2).sum(1)
#         an = data[:,int(channel_positions[count+1])-10:int(channel_positions[count+1])+10,-wl0.size:]
#         an = an.mean(2).sum(1)
#         plt.plot(scan_range/wavel, n)
#         plt.plot(scan_range/wavel, an)
#         plt.grid()
#         plt.xlabel('Scan range')
#         plt.ylabel('Intensity (count)')
#         plt.title(null_labels[k])
#         count += 2    