/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Remi Lehe, Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <array>
#include <memory>

using namespace amrex::literals;

void WarpX::ComputeSpaceChargeField (bool const reset_E_field, bool const reset_B_field)
{
    ABLASTR_PROFILE("WarpX::ComputeSpaceChargeField");
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Reset E and B fields to 0, before calculating space-charge fields if requested
    for (int lev = 0; lev <= max_level; lev++) {
        for (int comp=0; comp<3; comp++) {
            if (reset_E_field) {
                m_fields.get(FieldType::Efield_fp, Direction{comp}, lev)->setVal(0);
            }
            if (reset_B_field) {
                m_fields.get(FieldType::Bfield_fp, Direction{comp}, lev)->setVal(0);
            }
        }
    }

    m_electrostatic_solver->ComputeSpaceChargeField(
        m_fields, *mypc, myfl.get(), max_level );
}

std::unique_ptr<amrex::MultiFab> WarpX::DepositScratchRho (int const lev)
{
    ABLASTR_PROFILE("WarpX::DepositScratchRho");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level == 0,
        "DepositScratchRho is only implemented for a single level");

    amrex::BoxArray nodal_ba = boxArray(lev);
    nodal_ba.surroundingNodes();
    auto rho = std::make_unique<amrex::MultiFab>(
        nodal_ba, DistributionMap(lev), 1, get_ng_depos_rho());

    // MultiParticleContainer::DepositCharge zeroes rho, accumulates every species
    // and applies the RZ inverse-volume scaling; SyncRho filters and sums the
    // guard cells, in the same order the electrostatic solvers use.
    amrex::Vector<amrex::MultiFab*> const rho_lev{rho.get()};
    mypc->DepositCharge(rho_lev, 0._rt);

    amrex::Vector<std::unique_ptr<amrex::MultiFab>> const no_coarse_patch(1);
    SyncRho(rho_lev, amrex::GetVecOfPtrs(no_coarse_patch),
            amrex::GetVecOfPtrs(no_coarse_patch));

#ifndef WARPX_DIM_RZ
    // Reflect the density over PEC boundaries, if needed.
    ApplyRhofieldBoundary(lev, rho.get(), PatchType::fine);
#endif

    return rho;
}

void WarpX::SolvePoissonEfield ()
{
    ABLASTR_PROFILE("WarpX::SolvePoissonEfield");

    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level == 0,
        "SolvePoissonEfield is only implemented for a single level");

    int const lev = 0;
    auto& es = GetElectrostaticSolver();

    // The Poisson boundary handler only defines lobc/hibc when the input file
    // specifies a potential. This entry point is meant to be driven at runtime,
    // e.g. through set_potential_on_eb, so define them here as well; DefinePhiBCs
    // is idempotent.
    es.m_poisson_boundary_handler->DefinePhiBCs(Geom(lev));

    auto const rho = DepositScratchRho(lev);

    amrex::BoxArray nodal_ba = boxArray(lev);
    nodal_ba.surroundingNodes();
    amrex::MultiFab phi(nodal_ba, DistributionMap(lev), 1, 1);
    phi.setVal(0._rt);

    amrex::Vector<amrex::MultiFab*> const rho_lev{rho.get()};
    amrex::Vector<amrex::MultiFab*> const phi_lev{&phi};

    es.setPhiBC(phi_lev, gett_new(lev));

    MultiLevelVectorField efield =
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, finest_level);
    for (int component = 0; component < 3; ++component) {
#ifdef WARPX_DIM_RZ
        if (component == 1) { continue; }
#endif
        efield[lev][component]->setVal(0._rt);
    }

    std::array<amrex::Real, 3> const beta = {0._rt, 0._rt, 0._rt};
    if (EB::enabled()) {
        // with an EB, computePhi also fills the electric field
        es.computePhi(
            rho_lev, phi_lev, beta,
            es.self_fields_required_precision, es.self_fields_absolute_tolerance,
            es.self_fields_max_iters, es.self_fields_verbosity,
            es.is_igf_2d_slices, efield);
    } else {
        es.computePhi(
            rho_lev, phi_lev, beta,
            es.self_fields_required_precision, es.self_fields_absolute_tolerance,
            es.self_fields_max_iters, es.self_fields_verbosity,
            es.is_igf_2d_slices);
        es.computeE(efield, phi_lev, beta);
    }

    // publish phi where a diagnostic can pick it up
    if (m_fields.has(FieldType::phi_fp, lev)) {
        amrex::MultiFab* registered_phi = m_fields.get(FieldType::phi_fp, lev);
        if (registered_phi->boxArray() == phi.boxArray() &&
            registered_phi->DistributionMap() == phi.DistributionMap()) {
            amrex::MultiFab::Copy(*registered_phi, phi, 0, 0, 1,
                amrex::min(registered_phi->nGrowVect(), phi.nGrowVect()));
        }
    }
}
