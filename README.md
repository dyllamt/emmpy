# emmpy
The Python implementation of the effective mass model, which treats experimental transport properties of semiconductors.

This implementation uses the notation of S. Kang and G. J. Snyder _Charge-transport model for conducting polymers_ (2017) and can treat both organic and inorganic semiconductors. When non-polar phonon scattering limits conduction, s=1.

The basic assumptions of this model are:
- particles involved in conduction are described by the Fermi-Dirac distribution function
- there is a transport edge (a band edge), above which carriers contribute to conduction
- a powerlaw describes the diffusivity of particles in energy-space (linear for inorganic semiconductors)
