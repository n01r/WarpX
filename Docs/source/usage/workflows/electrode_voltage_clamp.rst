.. _usage-electrode-voltage-clamp:

Holding embedded electrodes at a prescribed voltage
===================================================

WarpX's electromagnetic update evolves :math:`E` and :math:`B`, but it does not by
itself hold an embedded conductor at a prescribed potential. Over many steps the
effective electrode voltage therefore drifts as the plasma charges the surface.

:class:`pywarpx.multi_electrode_corrector.MultiElectrodeBiasCorrector` adds a
quasi-static correction for several driven electrodes without a Poisson solve per
step.

How it works
------------

At setup, one unit field :math:`E_{0k} = -\nabla\phi_{0k}` is precomputed per
electrode, with electrode :math:`k` at 1 V and all others grounded, together with
the vacuum capacitance matrix :math:`C_{jk} = \epsilon_0 \oint_j E_{0k}\cdot n\, dS`.

Each correction measures the induced charge :math:`Q_j` of the live field and the
plasma-induced charge :math:`Q_{g,j}` that the same plasma would induce with every
electrode grounded, inverts

.. math::

   V = C^{-1} (Q - Q_g),

and adds :math:`\sum_k \delta V_k E_{0k}` with
:math:`\delta V = \mathrm{relaxation}\,(V_\mathrm{target} - V)`. Because the unit
fields are discrete gradients, the correction changes neither :math:`\nabla\cdot E`
nor :math:`\nabla\times E`.

:math:`Q_g` is obtained either from a grounded Poisson solve (``qg_mode="grounded"``)
or, by default, from the discrete Shockley--Ramo pairing
:math:`Q_{g,k} = -\sum_a \rho_a \Psi_k[a]`, which needs no solve. The weighting
potential :math:`\Psi_k` is built from the transpose of WarpX's own discrete EB
operator and charge functional; on a non-symmetric cut-cell operator the plain
unit-voltage basis is not the adjoint of that functional, and using it would
mis-book the charge.

Usage
-----

.. code-block:: python

   from pywarpx.callbacks import (
       installafterEsolve,
       installafterInitatRestart,
       installafterInitEsolve,
   )
   from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

   corrector = MultiElectrodeBiasCorrector(
       sim=sim,
       correction_interval=10,
       electrodes=[
           {"name": "left", "region": "(x<0)", "potential": +300.0},
           {"name": "right", "region": "(x>0)", "potential": -700.0},
       ],
   )
   # afterInitEsolve does not run on a restart, so register both init hooks
   installafterInitEsolve(corrector.setup_after_init)
   installafterInitatRestart(corrector.setup_after_init)
   installafterEsolve(corrector.correct_field)

``pywarpx.multi_electrode_logger`` provides two optional diagnostics:
``MultiElectrodeClampTelemetry`` writes the per-correction voltages and charges to
a CSV without re-measuring anything, and ``GroundedChargeCrossCheck`` sparsely
compares the adjoint observer against an independent grounded Poisson solve.

Supported envelope
------------------

* 3D and RZ, with embedded boundaries enabled.
* Driven electrodes with prescribed (possibly time-dependent) potentials.
* Grounded (PEC) outer field boundaries on every wall; in RZ, regularity at
  :math:`r=0`. Other outer boundaries are rejected, because the weighting
  potential is only the adjoint of the charge functional for a grounded reference.
* In RZ, nodal CIC deposition (``particle_shape=1``) without a charge filter.

Floating electrodes, external circuit coupling and dielectric embedded boundaries
are not modelled.
