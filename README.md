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

## Requirements
- numpy
- astropy
- h5py
- hcipy
- scipy

## Update 2022-03-29
The GLINT instrument with directional coupler, 4T mask and C-Red2 has been implemented and ready to use.

## Next step
1. Implementing a saving system of the data as for real observation data
2. Implementing the tricoupler
3. Implementing the fringe tracking capability
