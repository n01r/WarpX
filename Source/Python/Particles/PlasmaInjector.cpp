/* Copyright 2025 The WarpX Community
 *
 * Authors: Axel Huebl, Marco Garten
 * License: BSD-3-Clause-LBNL
 */

#include "Python/pyWarpX.H"

#include <Initialization/PlasmaInjector.H>


void init_PlasmaInjector (py::module& m)
{
    py::class_<PlasmaInjector> plasma_injector(m, "PlasmaInjector");
    
    plasma_injector
        .def_readwrite("flux_multiplier", &PlasmaInjector::flux_multiplier,
            "Runtime-adjustable flux multiplier for interactive control. "
            "Scales the particle injection rate. Default: 1.0")
        .def_readwrite("injection_weight", &PlasmaInjector::injection_weight,
            "If > 0, enables Mode 2 where particle weight is fixed and "
            "num_ppc adjusts automatically to maintain physical flux. "
            "If <= 0 (default), uses Mode 1 with fixed num_ppc and weight from flux.")
        .def_readonly("num_particles_per_cell_real", &PlasmaInjector::num_particles_per_cell_real,
            "Number of particles per cell (may be fractional for flux injection)")
        .def("doFluxInjection", &PlasmaInjector::doFluxInjection,
            "Returns true if flux injection is enabled for this injector")
    ;
}
