/* Copyright 2021-2022 The WarpX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyWarpX.H"

#include <WarpX.H>

// see WarpX.cpp - full includes for _fwd.H headers
#include <BoundaryConditions/PEC_Insulator.H>
#include <BoundaryConditions/PML.H>
#include <Diagnostics/MultiDiagnostics.H>
#include <Diagnostics/ReducedDiags/ChargeOnEB.H>
#include <FieldSolver/ElectrostaticSolvers/AdjointWeightingSolve.H>
#include <Diagnostics/ReducedDiags/MultiReducedDiags.H>
#include <EmbeddedBoundary/WarpXFaceInfoBox.H>
#include <FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H>
#include <FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H>
#include <FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H>
#ifdef WARPX_USE_FFT
#   include <FieldSolver/SpectralSolver/SpectralKSpace.H>
#   ifdef WARPX_DIM_RZ
#       include <FieldSolver/SpectralSolver/SpectralSolverRZ.H>
#       include <BoundaryConditions/PML_RZ.H>
#   else
#       include <FieldSolver/SpectralSolver/SpectralSolver.H>
#   endif // RZ ifdef
#endif // use PSATD ifdef
#include <FieldSolver/WarpX_FDTD.H>
#include <Filter/NCIGodfreyFilter.H>
#include <Initialization/ExternalField.H>
#include <Particles/MultiParticleContainer.H>
#include <Fluids/MultiFluidContainer.H>
#include <Fluids/WarpXFluidContainer.H>
#include <Particles/ParticleBoundaryBuffer.H>
#include <AcceleratorLattice/AcceleratorLattice.H>
#include <Utils/TextMsg.H>
#include <Utils/Parser/ParserUtils.H>
#include <Utils/WarpXAlgorithmSelection.H>
#include <Utils/WarpXConst.H>
#include <Utils/WarpXUtil.H>
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"

#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_OpenMP.H>

#if defined(AMREX_DEBUG) || defined(DEBUG)
#   include <cstdio>
#endif
#include <memory>
#include <string>
#include <vector>


//using namespace warpx;

namespace detail
{
    /** Helper Function for Property Getters
     *
     * This queries an amrex::ParmParse entry. This throws a
     * std::runtime_error if the entry is not found.
     *
     * This handles the most common throw exception logic in WarpX instead of
     * going over library boundaries via amrex::Abort().
     *
     * @tparam T type of the amrex::ParmParse entry
     * @param prefix the prefix, e.g., "warpx" or "amr"
     * @param name the actual key of the entry, e.g., "particle_shape"
     * @return the queried value (or throws if not found)
     */
    template< typename T>
    auto get_or_throw (std::string const & prefix, std::string const & name)
    {
        using V = std::decay_t<T>;
        V value;

        bool has_name = false;
        // TODO: if array do queryarr
        // has_name = amrex::ParmParse(prefix).queryarr(name.c_str(), value);
        if constexpr (std::is_same_v<V, bool> || std::is_same_v<V, std::string>) {
            has_name = amrex::ParmParse(prefix).query(name.c_str(), value);
        }
        else {
            has_name = amrex::ParmParse(prefix).queryWithParser(name.c_str(), value);
        }

        if (!has_name) {
            throw std::runtime_error(prefix + "." + name + " is not set yet");
        }
        return value;
    }
}

void init_WarpX (py::module& m)
{
    using ablastr::fields::Direction;

    // Expose the WarpX instance
    m.def("get_instance",
        [] () { return &WarpX::GetInstance(); },
        "Return a reference to the WarpX object.");

    m.def("finalize", &WarpX::Finalize,
        "Close out the WarpX related data");

    // WarpX is a singleton owned by the C++ side: its lifetime ends in
    // WarpX::Finalize (i.e. WarpX::ResetInstance), never when the last Python
    // reference goes away. Without py::nodelete, pybind11's default
    // return_value_policy for the raw pointer returned by get_instance below is
    // take_ownership, and destroying the Python object would leave
    // WarpX::m_instance dangling and WarpX::Finalize double-freeing it.
    py::class_<WarpX, std::unique_ptr<WarpX, py::nodelete>> warpx(m, "WarpX");
    warpx
        // WarpX is a Singleton Class with a private constructor
        //   https://github.com/BLAST-WarpX/warpx/pull/4104
        //   https://pybind11.readthedocs.io/en/stable/advanced/classes.html?highlight=singleton#custom-constructors
        .def(py::init([]() {
            return &WarpX::GetInstance();
        }))
        .def_static("get_instance",
            [] () { return &WarpX::GetInstance(); },
            "Return a reference to the WarpX object."
        )
        .def_static("finalize", &WarpX::Finalize,
            "Close out the WarpX related data"
        )

        .def("initialize_data", &WarpX::InitData,
            "Initializes the WarpX simulation"
        )
        .def("evolve", &WarpX::Evolve,
            "Evolve the simulation the specified number of steps"
        )

        .def_property("omp_threads",
            [](WarpX & /* wx */){
                return detail::get_or_throw<std::string>("amrex", "omp_threads");
            },
            [](WarpX & /* wx */, std::variant<int, std::string> omp_threads_var) {
                std::visit([&]( auto && omp_threads) {
                    amrex::ParmParse pp_amrex("amrex");
                    pp_amrex.add("omp_threads", omp_threads);

                    // set the value if not "system" or "nosmt"
                    if constexpr(std::is_same_v<std::decay_t<decltype(omp_threads)>, int>) {
                        amrex::Print() << "Changing WarpX threads to N=" << omp_threads << "\n";
                        amrex::OpenMP::set_num_threads(omp_threads);
                    }
                }, omp_threads_var);
            },
            "Controls the number of OpenMP threads to use (WarpX default: \"nosmt\").\n"
            "https://amrex-codes.github.io/amrex/docs_html/InputsComputeBackends.html."
        )

        // from amrex::AmrCore / amrex::AmrMesh
        .def_property_readonly("max_level",
            [](WarpX const & wx){ return wx.maxLevel(); },
            "The maximum mesh-refinement level for the simulation."
        )
        .def_property_readonly("finest_level",
            [](WarpX const & wx){ return wx.finestLevel(); },
            "The currently finest level of mesh-refinement used. This is always less or equal to max_level."
        )
        .def("Geom",
            //[](WarpX const & wx, int const lev) { return wx.Geom(lev); },
            py::overload_cast< int >(&WarpX::Geom, py::const_),
            py::arg("lev")
        )
        .def("DistributionMap",
            [](WarpX const & wx, int const lev) { return wx.DistributionMap(lev); },
            //py::overload_cast< int >(&WarpX::DistributionMap, py::const_),
            py::arg("lev")
        )
        .def("boxArray",
            [](WarpX const & wx, int const lev) { return wx.boxArray(lev); },
            //py::overload_cast< int >(&WarpX::boxArray, py::const_),
            py::arg("lev")
        )
        .def("multifab_register",&WarpX::GetMultiFabRegister,
            py::return_value_policy::reference_internal)

        .def("multi_particle_container",
            [](WarpX& wx){ return &wx.GetPartContainer(); },
            py::return_value_policy::reference_internal
        )
        .def("get_particle_boundary_buffer",
            [](WarpX& wx){ return &wx.GetParticleBoundaryBuffer(); },
            py::return_value_policy::reference_internal
        )

        // Expose functions used to sync the charge density multifab
        // accross tiles and apply appropriate boundary conditions
        .def("sync_rho",
            [](WarpX& wx){ wx.SyncRho(); }
        )
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        .def("apply_inverse_volume_scaling_to_charge_density",
            [](WarpX& wx, amrex::MultiFab* rho, int const lev) {
                wx.ApplyInverseVolumeScalingToChargeDensity(rho, lev);
            },
            py::arg("rho"), py::arg("lev")
        )
#endif

        // Expose functions to get the current simulation step and time
        .def("getistep",
            [](WarpX const & wx, int lev){ return wx.getistep(lev); },
            py::arg("lev"),
            "Get the current step on mesh-refinement level ``lev``."
        )
        .def("gett_new",
            [](WarpX const & wx, int lev){ return wx.gett_new(lev); },
            py::arg("lev"),
            "Get the current physical time on mesh-refinement level ``lev``."
        )
        .def("getdt",
            [](WarpX const & wx, int lev){ return wx.getdt(lev); },
            py::arg("lev"),
            "Get the current physical time step size on mesh-refinement level ``lev``."
        )

        .def("set_potential_on_domain_boundary",
            [](WarpX& wx,
               std::string potential_lo_x, std::string potential_hi_x,
               std::string potential_lo_y, std::string potential_hi_y,
               std::string potential_lo_z, std::string potential_hi_z)
            {
                if (potential_lo_x != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_xlo_str = potential_lo_x;
                if (potential_hi_x != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_xhi_str = potential_hi_x;
                if (potential_lo_y != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_ylo_str = potential_lo_y;
                if (potential_hi_y != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_yhi_str = potential_hi_y;
                if (potential_lo_z != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_zlo_str = potential_lo_z;
                if (potential_hi_z != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_zhi_str = potential_hi_z;
                wx.GetElectrostaticSolver().m_poisson_boundary_handler->BuildParsers();
            },
            py::arg("potential_lo_x") = "",
            py::arg("potential_hi_x") = "",
            py::arg("potential_lo_y") = "",
            py::arg("potential_hi_y") = "",
            py::arg("potential_lo_z") = "",
            py::arg("potential_hi_z") = "",
            "Sets the domain boundary potential string(s) and updates the function parser."
        )
        .def("set_potential_on_eb",
            [](WarpX& wx, std::string potential) {
                wx.GetElectrostaticSolver().m_poisson_boundary_handler->setPotentialEB(potential);
            },
            py::arg("potential"),
            "Sets the EB potential string and updates the function parser."
        )
        .def("solve_poisson_efield",
            [] (WarpX& wx, bool const eb_aware_gradient) {
                wx.SolvePoissonEfield(eb_aware_gradient);
            },
            py::arg("eb_aware_gradient") = true,
            "Deposit charge from all species, solve Poisson with the current EB and "
            "domain boundary conditions, replace Efield_fp with the result, and "
            "publish phi_fp when it is registered.\n\n"
            "With an embedded boundary, eb_aware_gradient=True (the default) takes E "
            "from the solve itself, which uses the shortened fluid length on cut "
            "edges. Pass False to extract E with the ordinary full-grid gradient "
            "instead: that is the representation the Yee Faraday update differences, "
            "so the resulting field has no discrete curl, at the cost of a less "
            "accurate local field next to the boundary. Without an embedded boundary "
            "the flag has no effect."
        )
        .def("compute_eb_charge",
            [] (WarpX& wx, const std::string& weighting, const std::string& field) {
                int const lev = 0;
                ablastr::fields::VectorField const E = wx.m_fields.get_alldirs(field, lev);
                if (weighting.empty() || weighting == "1") {
                    return WeightedChargeOnEB(E, lev, nullptr);
                }
                amrex::Parser parser = utils::parser::makeParser(weighting, {"x", "y", "z"});
                return WeightedChargeOnEB(E, lev, &parser);
            },
            py::arg("weighting") = "1",
            py::arg("field") = "Efield_fp",
            "Induced charge eps0 * oint w(x,y,z) E.n dS over the embedded boundary for "
            "the named field, with an optional weighting w(x,y,z) selecting one "
            "electrode. 3D and RZ with EB only."
        )
        .def("grounded_charge_from_adjoint",
            [] (WarpX& /*wx*/, const std::vector<std::string>& psi_fields, int lev) {
                auto const q = WarpXGroundedChargeFromAdjoint(psi_fields, lev);
                return std::vector<amrex::Real>(q.begin(), q.end());
            },
            py::arg("psi_fields"), py::arg("lev") = 0,
            "Plasma-induced charge on each electrode with all electrodes grounded, "
            "evaluated as -sum rho*Psi*dV against a freshly deposited charge density. "
            "No Poisson solve. Shared nodes on box boundaries are counted once and the "
            "RZ measure is WarpX's cylindrical nodal volume."
        )
        .def("solve_adjoint_weighting",
            [] (WarpX& wx, const std::string& region, const std::string& out_name,
                amrex::Real tol, int max_iter) {
                int const lev = 0;
                auto const& levset = wx.fieldEBFactory(lev).getLevelSet();

                amrex::MultiFab* psi = wx.m_fields.get(out_name, lev);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(psi != nullptr,
                    "solve_adjoint_weighting: output field not registered");

                const amrex::BoxArray& ba = psi->boxArray();
                const amrex::DistributionMapping& dm = psi->DistributionMap();

                // Constrained rows are the nodes covered by the EB and the nodes on
                // the grounded outer walls. Leaving the walls free would pose the
                // operator on a space that includes unconstrained boundary values.
                amrex::iMultiFab dmsk(ba, dm, 1, 1);
                dmsk.setVal(0);

                const amrex::Box ndom = amrex::surroundingNodes(wx.Geom(lev).Domain());
                const auto dlo = ndom.smallEnd();
                const auto dhi = ndom.bigEnd();

                for (amrex::MFIter mfi(dmsk); mfi.isValid(); ++mfi) {
                    auto const& dma = dmsk.array(mfi);
                    auto const& ls  = levset.const_array(mfi);
                    amrex::ParallelFor(mfi.growntilebox(),
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            const bool covered = (ls(i,j,k) >= amrex::Real(0.0));
#ifdef WARPX_DIM_RZ
                            // r=0 is a regularity axis, not a grounded wall
                            const bool wall =
                                (i >= dhi[0] || j <= dlo[1] || j >= dhi[1]);
#else
                            const bool wall =
                                (i <= dlo[0] || i >= dhi[0] ||
                                 j <= dlo[1] || j >= dhi[1] ||
                                 k <= dlo[2] || k >= dhi[2]);
#endif
                            dma(i,j,k) = (covered || wall) ? 1 : 0;
                        });
                }

                amrex::MultiFab rhs(ba, dm, 1, 1);
                WarpXBuildAdjointRHSChargeFunctional(rhs, region, dmsk, lev);

                amrex::Real res = -1.0;
                int iters = -1;
                const bool ok = WarpXSolveAdjointWeighting(
                    *psi, rhs, dmsk, lev, tol, max_iter, &res, &iters);
                amrex::Print() << "solve_adjoint_weighting: " << iters
                               << " outer iterations, converged="
                               << (ok ? "true" : "false")
                               << ", rel.residual=" << res << "\n";

                WarpXFinalizeChargeFunctionalPsi(*psi, lev);
                psi->FillBoundary(wx.Geom(lev).periodicity());
                return py::make_tuple(ok, res);
            },
            py::arg("region"), py::arg("out_name"),
            py::arg("tol") = 1.0e-10, py::arg("max_iter") = 200,
            "Solve for the adjoint weighting potential Psi_k of the electrode selected "
            "by region(x,y,z) and write it into the registered nodal field out_name. "
            "Unlike the plain unit-voltage basis, this Psi makes the grounded-charge "
            "identity Q_k = -sum_a rho_a Psi_k[a] exact on WarpX's non-symmetric EB "
            "Laplacian. Requires grounded (PEC) outer boundaries. "
            "Returns (converged, relative_residual); the caller must check converged, "
            "because an unconverged Psi yields a plausible but wrong correction. "
            "3D and RZ with EB only."
        )

        .def("run_div_cleaner",
            [] (WarpX& wx) { wx.ProjectionCleanDivB(); },
            "Executes projection based divergence cleaner on loaded Bfield_fp_external."
        )
        .def_static("calculate_hybrid_external_curlA",
            [] (WarpX& wx) { wx.CalculateExternalCurlA(); },
            "Executes calculation of the curl of the external A in the hybrid solver."
        )
        .def("synchronize_velocity_with_position",
            [] (WarpX& wx) { wx.SynchronizeVelocityWithPosition(); },
            "Synchronize particle velocities and positions."
        )
        // Add some accessor bindings for the Hybrid Ohm's Law Solver
        .def("set_hybrid_pic_substeps",
            [](WarpX& wx, int substeps) {
                wx.get_pointer_HybridPICModel()->m_substeps = substeps;
            },
            py::arg("substeps"),
            "Sets the number of substeps to take in the hybrid solver."
        )
        .def("get_hybrid_pic_substeps",
            [](WarpX& wx) {
                return wx.get_pointer_HybridPICModel()->m_substeps;
            },
            "Gets the number of substeps taken in the hybrid solver."
        )
        .def("set_hybrid_pic_density_floor",
            [](WarpX& wx, amrex::Real n_floor) {
                wx.get_pointer_HybridPICModel()->m_n_floor = n_floor;
            },
            py::arg("n_floor"),
            "Sets the density floor to use in the hybrid solver."
        )
        .def("get_hybrid_pic_density_floor",
            [](WarpX& wx) {
                return wx.get_pointer_HybridPICModel()->m_n_floor;
            },
            "Gets the number of substeps to take in the hybrid solver."
        )
    ;
}
