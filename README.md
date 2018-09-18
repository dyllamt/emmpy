# semi-transport
The Python implementation of the effective mass model, which treats experimental transport properties of semiconductors. This implementation uses the notation of S. Kang and G. J. Snyder _Charge-transport model for conducting polymers_ (2017) and can treat both organic and inorganic semiconductors. When non-polar phonon scattering limits conduction, s=1 (linear).


Basic assumptions of this model:
- Carriers involved in conduction (electrons/holes) are described by the Fermi-Dirac distribution function.
- There is a transport edge (a band edge), above which carriers contribute to conduction.
- A powerlaw describes the diffusivity of particles v.s. particle energy (linear for inorganic semiconductors).
