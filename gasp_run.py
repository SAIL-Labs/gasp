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
"""

import numpy as np
import matplotlib.pyplot as plt
import gasp_lib as lib
from timeit import default_timer as timer

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
tdiam = 7.92
# Diameter of the sub-pupils (in meter)
subpup_diam = 1.
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
nring = 3 # Number of rings of the hexagon
ring_rad = 1.075 # Radius of the 1st ring, in metre
aper_rad = subpup_diam/2 # Radius of the sub-apertures
holes_id = [36, 3, 33, 22] # ID of the hexagon coordinates on which the apertures are, the order correspond to the numbering of the beams.
view_pupil = False # To check the pupil and the hex configuration
mirror_pistons = [wavel/4, 0, wavel/4, 0]
mirror_pistons = [0., 0., 0., 0.]

# =============================================================================
# Atmo parameters
# =============================================================================
# Fried parameter at wavelength wavel_r0 (in meters), the bigger, the better the seeing is
r0 = 0.16
ll = tdiam * oversz  # Physical extension of the wavefront (in meter)
# Outer scale for the model of turbulence, keep it close to infinity for Kolmogorov turbulence (the simplest form) (in meter)
L0 = 1e15
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

pupil_area = np.pi / 4 * subpup_diam**2

star_photons = MAG0FLUX * 10**(-0.4*magnitude) * \
    SCEXAO_THROUGHPUT * pupil_area * bandwidth*1e6 * dit
print('Star photo-electrons', star_photons *
      QE, (star_photons*QE)**0.5)

"""
Create the spectral dispersion
"""
wl0 = np.arange(wavel-bandwidth/2, wavel+bandwidth/2, dwl)
wl = lib.oversample_wavelength(wl0, oversampling_wl)

"""
Beam combiner
"""
zeta_dict = lib.load_zeta_file(zeta_path)
combiner, splitter_interfA_list, splitter_interfB_list, kappa_AB_list, kappa_BA_list, splitter_photoA_list, splitter_photoB_list = lib.get_glint_chip_markII(wl, zeta_dict=zeta_dict)

"""
Create the sub-pupil
"""
pupil, pupil_maskarray = lib.create_aperture_mask(psz, nring, ring_rad, aper_rad, holes_id, False, view_pupil)

"""
Create the phase screen
"""
if activate_turbulence:
    phs_screen = lib.generatePhaseScreen(
        wavel_r0, psz*oversz, ll, r0, L0, fc=fc_scex, correc=ao_correc,
        pdiam=tdiam, seed=seed)
else:
    phs_screen = np.zeros((psz*oversz, psz*oversz))


"""
Initialise storage lists
"""
data = []
noisy_data = []
monitor_pistons = []
monitor_injections = []
monitor_i_out = []
shifts = []


i_out = np.zeros((5, wl.size))
i_out_noft = np.zeros((5, wl.size))
i_out_bi = np.zeros((5, wl.size))

for t in timeline[:]:
    if activate_fringe_scan:
        mirror_pistons[beam_to_scan] = scan_range[t]
        
    if activate_turbulence:
        phs_screen_moved, xyshift = lib.movePhaseScreen(
            phs_screen, wind_speed, angle, t-TIME_OFFSET, meter2pixel)
        if xyshift[0] > phs_screen.shape[0] or xyshift[1] > phs_screen.shape[1]:
            if seed != None:
                seed += 20
            phs_screen = lib.generatePhaseScreen(
                wavel_r0, psz*oversz, ll, r0, L0, fc=fc_scex, correc=9, pdiam=tdiam,
                seed=None)
            TIME_OFFSET = t
    
        shifts.append(xyshift)
    else:
        phs_screen_moved = phs_screen
    
    # We stay in phase space hence the simple multiplication below to crop the wavefront.
    phs_pup = pupil * phs_screen_moved[phs_screen_moved.shape[0]//2-psz//2:phs_screen_moved.shape[0]//2+psz//2,
                                       phs_screen_moved.shape[1]//2-psz//2:phs_screen_moved.shape[1]//2+psz//2]

    # Injection and phase
    injections = []
    pistons = []
    for i in range(len(holes_id)):
        pistons.append(np.mean(phs_pup[pupil_maskarray[i]]))
        inj = lib.calculate_injection(phs_pup[pupil_maskarray[i]], wl)
        injections.append(inj)
    injections = np.array(injections)
    pistons = np.array(pistons)
    monitor_injections.append(injections)
    monitor_pistons.append(pistons)

    # Create incoming wavefronts
    a_in = np.exp(1j * 2 * np.pi/wl[:,None] * (pistons[None,:] + mirror_pistons[None,:]))
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


monitor_i_out = np.array(monitor_i_out)
data = np.array(data)
noisy_data = np.array(noisy_data)

plt.figure()
plt.imshow(det_img, origin='upper')

# plt.figure()
# plt.plot(zeta_dict['wl_scale'], zeta_dict['b1null1'])
# plt.plot(wl0*1e9, i_out[:,0] / i_out[:,12])

if activate_fringe_scan:
    monitor_i_out = monitor_i_out.mean(1)
    
    plt.figure()
    count = 0
    for k in range(6):
        plt.subplot(3, 2, k+1)
        plt.plot(scan_range/wavel, monitor_i_out[:,count])
        plt.plot(scan_range/wavel, monitor_i_out[:,count + 1])
        plt.grid()
        plt.xlabel('Scan range')
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
        plt.xlabel('Scan range')
        plt.ylabel('Intensity (count)')
        plt.title(null_labels[k])
        count += 2    