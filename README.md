# GASP
## Glint As Simulated in Python

This simulator aims at helping the development of the GLINT instrument.

The library contains many functions used to simulate GLINT with a variety
of configurations the user can create (mask, number of apertures, IO chip).

The run script is a ready-to-use script where small changes can be done to tune
the parameters of the simulations or bigger changes for simulating a different design.

The script has supposedly been written to require minimum and easy intervention for whatever
the user wants to do (play around or change the design).
I cannot ensure this goal is fulfill so far.

Both run script and library are fully documented and commented to understand
the way of thinking of the developments and how the physics of the instrument
has been implemented.

## Features
- Vectorized Scexao pupil
- Dynamic atmospheric turbulence modelled as an infinite layer (powered by HCIPY), *no AO correction yet*
- Sub-apertures and hexagonal segmented mirror generators and control of these segments in piston, tip and tilt
- Detector noise model of C-Red 2
- Fringe scanner
- Spectral dispersion

## Requirements
- astropy
- h5py
- hcipy
- itertools
- numpy
- scipy
- timeit

## Update 2022-09-27
- Longer exposure time than the timeline unit (e.g. to simulate fringe blurring)
- Correction of bug in the shifting of the phase screen
- Use of different kinds of couplers
- **Ready-to-use script for GLINT Mark II**

## Update 2022-09-22
Tricoupler implemented with preset models and ability to create customized models. The presets models are 'all-in-all' (each beam interfere with all the others pair-wise) and 'pairwise' (create fixed pairs of beams, the number of baselines is half the number of apertures).
Can generate any number of apertures.

Saving system of the data.

Ready-to-use script for GLINT Mark II.

## Update 2022-03-29
The GLINT instrument with directional coupler, 4T mask and C-Red2 has been implemented and ready to use.

## Next step
1. Implementing effects of AO correction
2. Implementing the fringe tracking capability
