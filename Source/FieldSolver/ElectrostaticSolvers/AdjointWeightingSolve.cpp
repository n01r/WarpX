/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

/* Adjoint weighting-potential solve.
 *
 * Computes Psi_k = A^{-T} c_k, the weighting potential that makes the
 * grounded-electrode charge identity
 *
 *     Q_k(rho) = - sum_a rho_a Psi_k[a]
 *
 * exact rather than approximate. The plain unit-voltage basis solves
 * A psi = -A_FD (the Dirichlet coupling column) while the identity needs
 * A^T psi = c_k (the charge-functional row). Those differ unless A is symmetric,
 * and WarpX's EB Laplacian is not symmetric at cut cells.
 *
 * amrex::MLLinOp exposes only a forward apply, so A^T is applied matrix-free
 * (AdjointWeightingPotential.H) and the system is solved with BiCGSTAB,
 * right-preconditioned by a forward MLMG solve of the same EB Laplacian. A is
 * non-symmetric only in the thin cut-cell shell, so A^{-1} is a good approximate
 * inverse of A^{-T}; the outer iteration count is a few tens and independent of
 * resolution.
 */

#include "AdjointWeightingSolve.H"

#include "AdjointWeightingPotential.H"
#include "Diagnostics/ReducedDiags/ChargeOnEB.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <AMReX_EBFabFactory.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_LO_BCTYPES.H>
#include <AMReX_MFIter.H>
#include <AMReX_MLEBNodeFDLap_K.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiCutFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Parser.H>

#include <cmath>
#include <exception>
#include <memory>

using namespace amrex::literals;

namespace {

/** Edge-centroid arrays that stay valid on boxes carrying no cut cells.
 *
 * ``MultiCutFab`` only allocates data where the box FabType is
 * ``singlevalued``, and ``MultiCutFab::const_array`` checks this with
 * ``AMREX_ASSERT`` only -- a no-op in a Release build. Reading it on a fully
 * regular or fully covered box is therefore undefined behaviour, which shows up
 * once the domain is decomposed finely enough that some boxes miss the embedded
 * boundary entirely (e.g. a radial split in RZ puts the outer boxes off the
 * conductor). AMReX's own ``MLEBNodeFDLaplacian::Fapply`` guards this with
 * ``edgecent[0]->ok(mfi)`` and dispatches to a non-EB kernel.
 *
 * WarpX has no non-EB *transpose* kernel, and dispatching only the forward path
 * would mix gather and scatter semantics across a box boundary and break the
 * ``SumBoundary`` assembly. Instead we hand the EB kernels an array of 1.0 --
 * the sentinel for a fully open edge -- on those boxes, which makes every
 * ``(ec == 1.0) ? 1.0 : ...`` branch take the uncut path and reduces the EB
 * stencil exactly to the non-EB one.
 */
class EdgeCentFallback
{
public:
    explicit EdgeCentFallback (int lev)
    {
        auto const& edge_cent =
            WarpX::GetInstance().fieldEBFactory(lev).getEdgeCent();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            // The adjoint weighting potential is only defined with an embedded
            // boundary, so the cut-cell data must exist; getEdgeCent() returns
            // null pointers only when the factory is not an EBFArrayBoxFactory.
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(edge_cent[idim] != nullptr,
                "AdjointWeightingSolve requires embedded boundaries to be enabled");
            m_ones[idim].define(edge_cent[idim]->boxArray(),
                                edge_cent[idim]->DistributionMap(),
                                1, edge_cent[idim]->nGrow());
            m_ones[idim].setVal(1.0_rt);
        }
    }

    /** True when this box actually carries cut-cell data. */
    [[nodiscard]] static bool hasCutData (
        amrex::Array<const amrex::MultiCutFab*,AMREX_SPACEDIM> const& edge_cent,
        amrex::MFIter const& mfi)
    {
        // non-null is a precondition, asserted in the constructor
        return edge_cent[0]->ok(mfi);
    }

    [[nodiscard]] amrex::Array4<amrex::Real const> array (
        amrex::Array<const amrex::MultiCutFab*,AMREX_SPACEDIM> const& edge_cent,
        amrex::MFIter const& mfi, int idim) const
    {
        return hasCutData(edge_cent, mfi) ? edge_cent[idim]->const_array(mfi)
                                          : m_ones[idim].const_array(mfi);
    }

private:
    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> m_ones;
};

/** Matrix-free A and A^T on the free-node block, with the scratch buffers the
 * Krylov iteration would otherwise reallocate on every apply.
 */
class AdjointOperator
{
public:
    AdjointOperator (amrex::BoxArray const& ba, amrex::DistributionMapping const& dm,
                     amrex::iMultiFab const& dmsk, int lev)
        : m_dmsk(&dmsk), m_lev(lev),
          m_period(WarpX::GetInstance().Geom(lev).periodicity()),
          m_xg(ba, dm, 1, 1),
          m_owner(amrex::OwnerMask(m_xg, m_period)),
          m_ec_fallback(lev)
    {}

    /** y = A x, or y = A^T x when `transpose`. */
    void apply (amrex::MultiFab& y, amrex::MultiFab const& x, bool transpose) const;

    /** Dot product counting shared nodes on box boundaries exactly once. */
    [[nodiscard]] amrex::Real dot (amrex::MultiFab const& a, amrex::MultiFab const& b) const
    {
        return amrex::MultiFab::Dot(*m_owner, a, 0, b, 0, 1, 0);
    }

    /** Zero the constrained rows, which carry no equation. */
    void restrictToFreeRows (amrex::MultiFab& y) const;

private:
    amrex::iMultiFab const* m_dmsk;
    int m_lev;
    amrex::Periodicity m_period;
    mutable amrex::MultiFab m_xg;
    std::unique_ptr<amrex::iMultiFab> m_owner;
    // built once: apply() runs on every Krylov iteration
    EdgeCentFallback m_ec_fallback;
};

void AdjointOperator::restrictToFreeRows (amrex::MultiFab& y) const
{
    for (amrex::MFIter mfi(y); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        auto const& ya = y.array(mfi);
        auto const& dm = m_dmsk->const_array(mfi);
        amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (dm(i,j,k) != 0) { ya(i,j,k) = 0._rt; }
            });
    }
}

void AdjointOperator::apply (amrex::MultiFab& y, amrex::MultiFab const& x,
                             bool transpose) const
{
    auto& warpx = WarpX::GetInstance();
    auto const& eb_fact = warpx.fieldEBFactory(m_lev);
    auto const& edge_cent = eb_fact.getEdgeCent();
    auto const& levset = eb_fact.getLevelSet();

    const auto dx = warpx.Geom(m_lev).CellSizeArray();
#ifdef WARPX_DIM_RZ
    const amrex::Real dr = dx[0];
    const amrex::Real dz = dx[1];
    const amrex::Real rlo = warpx.Geom(m_lev).ProbLo(0);
#else
    const amrex::Real bx = 1._rt / (dx[0] * dx[0]);
    const amrex::Real by = 1._rt / (dx[1] * dx[1]);
    const amrex::Real bz = 1._rt / (dx[2] * dx[2]);
#endif

    y.setVal(0.0);
    // FillBoundary does not touch ghosts outside a non-periodic physical
    // boundary. Those walls are grounded Dirichlet, so zero the scratch first
    // to make the homogeneous outer condition explicit rather than leaving
    // uninitialised memory in the stencil.
    m_xg.setVal(0.0);
    amrex::MultiFab::Copy(m_xg, x, 0, 0, 1, 0);
    m_xg.FillBoundary(m_period);

    for (amrex::MFIter mfi(y); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        auto const& ya = y.array(mfi);
        auto const& xa = m_xg.const_array(mfi);
        auto const& ls = levset.const_array(mfi);
        auto const& dm = m_dmsk->const_array(mfi);
        auto const& own = m_owner->const_array(mfi);
        // boxes with no cut cells carry no MultiCutFab data; see EdgeCentFallback
        auto const& ecx = m_ec_fallback.array(edge_cent, mfi, 0);
        auto const& ecy = m_ec_fallback.array(edge_cent, mfi, 1);
#ifndef WARPX_DIM_RZ
        auto const& ecz = m_ec_fallback.array(edge_cent, mfi, 2);
#endif

        if (transpose) {
            // amrex::For: the transpose is a scatter into y(i+/-1), which is
            // unsafe under the SIMD pragma of ParallelFor (see issue #7097)
            amrex::For(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // A nodal row on a box interface is valid in both FABs but
                    // is one global degree of freedom: scatter it exactly once
                    // and let SumBoundary collect the contributions.
                    if (own(i,j,k) == 0) { return; }
#ifdef WARPX_DIM_RZ
                    warpx_mlebndfdlap_adotx_rz_transpose_eb(
                        i, j, k, ya, xa, ls, dm, ecx, ecy, dr, dz, rlo);
#else
                    warpx_mlebndfdlap_adotx_transpose_eb(
                        i, j, k, ya, xa, ls, dm, ecx, ecy, ecz, bx, by, bz);
#endif
                });
        } else {
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // AMReX's own forward row, with a homogeneous EB Dirichlet
                    // value, so the transpose above cannot drift from it
                    // unnoticed.
#ifdef WARPX_DIM_RZ
                    amrex::mlebndfdlap_adotx_rz_eb(
                        i, j, k, ya, xa, ls, dm, ecx, ecy, /*xeb=*/0._rt,
                        /*sigr=*/1._rt, dr, dz, rlo, /*alpha=*/0._rt);
#else
                    amrex::mlebndfdlap_adotx_eb(
                        i, j, k, ya, xa, ls, dm, ecx, ecy, ecz, /*xeb=*/0._rt,
                        bx, by, bz);
#endif
                });
        }
    }

    if (transpose) {
        y.SumBoundary(0, 1, amrex::IntVect(1), amrex::IntVect(0), m_period);
        // after the assembly, so that a neighbouring FAB cannot add a
        // contribution back onto a wall row
        restrictToFreeRows(y);
        y.OverrideSync(m_period);
    }
    y.FillBoundary(m_period);
}

/** WarpX's own forward EB Laplacian, configured exactly as
 * ablastr::fields::computePhi's EB branch, used as the preconditioner.
 */
std::unique_ptr<amrex::MLEBNodeFDLaplacian> BuildAdjointPrecondLinOp (
    amrex::BoxArray const& ba, amrex::DistributionMapping const& dm, int lev)
{
    auto& warpx = WarpX::GetInstance();
    auto const& eb_fact = warpx.fieldEBFactory(lev);

    amrex::LPInfo info;
    auto linop = std::make_unique<amrex::MLEBNodeFDLaplacian>(
        amrex::Vector<amrex::Geometry>{warpx.Geom(lev)},
        amrex::Vector<amrex::BoxArray>{ba},
        amrex::Vector<amrex::DistributionMapping>{dm},
        info,
        amrex::Vector<amrex::EBFArrayBoxFactory const*>{&eb_fact});

    // sigma = 1 and no azimuthal mode: the values the transposed RZ row assumes
    linop->setSigma({AMREX_D_DECL(1._rt, 1._rt, 1._rt)});
#ifdef WARPX_DIM_RZ
    linop->setRZ(true);
#endif
    linop->setEBDirichlet(0._rt);

#ifdef WARPX_DIM_RZ
    amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> const lobc = {
        amrex::LinOpBCType::Neumann, amrex::LinOpBCType::Dirichlet};
    amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> const hibc = {
        amrex::LinOpBCType::Dirichlet, amrex::LinOpBCType::Dirichlet};
#else
    amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> const lobc = {AMREX_D_DECL(
        amrex::LinOpBCType::Dirichlet, amrex::LinOpBCType::Dirichlet,
        amrex::LinOpBCType::Dirichlet)};
    amrex::Array<amrex::LinOpBCType,AMREX_SPACEDIM> const hibc = lobc;
#endif
    linop->setDomainBC(lobc, hibc);

    return linop;
}

/** The adjoint is posed against a grounded reference. Refuse any other outer
 * boundary rather than returning a plausible but wrong weighting potential.
 */
void AssertGroundedOuterBoundaries ()
{
    auto const grounded = [] (FieldBoundaryType bc) {
        return bc == FieldBoundaryType::PEC;
    };
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
#ifdef WARPX_DIM_RZ
        // r = 0 is a regularity axis, not a wall
        const bool lo_ok = (idim == 0)
            ? (WarpX::field_boundary_lo[0] == FieldBoundaryType::None)
            : grounded(WarpX::field_boundary_lo[idim]);
#else
        const bool lo_ok = grounded(WarpX::field_boundary_lo[idim]);
#endif
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(lo_ok && grounded(WarpX::field_boundary_hi[idim]),
            "The adjoint weighting potential requires grounded (PEC) outer field "
            "boundaries on every wall; Neumann, periodic and open/PML references "
            "are not supported.");
    }
}

/** Per-node right-hand-side scale S = min(hp,hm) over the node's edges, the same
 * diagonal MLEBNodeFDLaplacian::scaleRHS applies inside MLMG::solve.
 */
void ComputeNodeMinScale (amrex::MultiFab& scale_mf, int lev)
{
    auto& warpx = WarpX::GetInstance();
    auto const& edge_cent = warpx.fieldEBFactory(lev).getEdgeCent();

    const EdgeCentFallback ec_fallback(lev);

    for (amrex::MFIter mfi(scale_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        auto const& sa = scale_mf.array(mfi);
        // boxes with no cut cells carry no MultiCutFab data; see EdgeCentFallback
        auto const& ecx = ec_fallback.array(edge_cent, mfi, 0);
        auto const& ecy = ec_fallback.array(edge_cent, mfi, 1);
#ifndef WARPX_DIM_RZ
        auto const& ecz = ec_fallback.array(edge_cent, mfi, 2);
#endif
        amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                using amrex::Real;
                const Real hpx = (ecx(i  ,j,k) == 1._rt) ? 1._rt : 1._rt + 2._rt*ecx(i  ,j,k);
                const Real hmx = (ecx(i-1,j,k) == 1._rt) ? 1._rt : 1._rt - 2._rt*ecx(i-1,j,k);
                const Real hpy = (ecy(i,j  ,k) == 1._rt) ? 1._rt : 1._rt + 2._rt*ecy(i,j  ,k);
                const Real hmy = (ecy(i,j-1,k) == 1._rt) ? 1._rt : 1._rt - 2._rt*ecy(i,j-1,k);
                Real scale = amrex::min(hmx, hpx);
                scale = amrex::min(scale, hmy, hpy);
#ifndef WARPX_DIM_RZ
                const Real hpz = (ecz(i,j,k  ) == 1._rt) ? 1._rt : 1._rt + 2._rt*ecz(i,j,k  );
                const Real hmz = (ecz(i,j,k-1) == 1._rt) ? 1._rt : 1._rt - 2._rt*ecz(i,j,k-1);
                scale = amrex::min(scale, hmz, hpz);
#endif
                sa(i,j,k) = scale;
            });
    }
    scale_mf.FillBoundary(warpx.Geom(lev).periodicity());
}

/** z ~= A^{-1} v through one forward MLMG solve.
 *
 * MLMG multiplies the right-hand side by S (see ComputeNodeMinScale) before
 * solving, so v is pre-divided by S to get A z = v rather than A z = S v.
 * Skipping this measured an O(1) round-trip error, i.e. a useless preconditioner.
 * In RZ the operator is self-adjoint under the radial measure W away from cut
 * cells, so W A^{-1} W^{-1} is the natural transpose preconditioner; the axis row
 * coefficient 4/dr^2 corresponds to W(0) = dr/8.
 */
void ApplyPrecond (amrex::MultiFab& z, amrex::MultiFab const& v,
                   AdjointOperator const& op, int lev,
                   amrex::MLMG& mlmg, amrex::Real prec_rtol,
                   amrex::MultiFab const& sinv)
{
    amrex::MultiFab vscaled(v.boxArray(), v.DistributionMap(), 1, v.nGrowVect());
    amrex::MultiFab::Copy(vscaled, v, 0, 0, 1, 0);
    amrex::MultiFab::Multiply(vscaled, sinv, 0, 0, 1, 0);

#ifdef WARPX_DIM_RZ
    auto& warpx = WarpX::GetInstance();
    const amrex::Real dr = warpx.Geom(lev).CellSize(0);
    const amrex::Real rlo = warpx.Geom(lev).ProbLo(0);
    auto const radial_measure = [=] AMREX_GPU_DEVICE (int i) {
        const amrex::Real r = rlo + amrex::Real(i)*dr;
        return (r == 0._rt) ? amrex::Real(1.0/8.0) : r/dr;
    };
    for (amrex::MFIter mfi(vscaled); mfi.isValid(); ++mfi) {
        auto const& va = vscaled.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                va(i,j,k) /= radial_measure(i);
            });
    }
#else
    amrex::ignore_unused(lev);
#endif

    z.setVal(0.0);
    try {
        mlmg.solve({&z}, {&vscaled}, prec_rtol, 0._rt);
    } catch (std::exception const& e) {
        // MLMG aborts on non-convergence unless setThrowException is used. z is
        // aliased as MLMG's solution buffer, so it holds the last iterate: a
        // preconditioner that merely ran out of iterations degrades the outer
        // iteration instead of killing the run.
        amrex::Print() << "WarpXSolveAdjointWeighting: preconditioner solve did not "
                       << "converge (" << e.what() << "); using its last iterate.\n";
    }

#ifdef WARPX_DIM_RZ
    for (amrex::MFIter mfi(z); mfi.isValid(); ++mfi) {
        auto const& za = z.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                za(i,j,k) *= radial_measure(i);
            });
    }
#endif

    op.restrictToFreeRows(z);
    z.FillBoundary(WarpX::GetInstance().Geom(lev).periodicity());
}

/** Discrete Shockley-Ramo pairing sum rho * Psi * dV on distributed nodal fields.
 *
 * Nodal valid boxes overlap at box boundaries, so the product is formed locally
 * and reduced with sum_unique, which applies AMReX's owner mask. The measure is
 * the same cylindrical nodal volume used to normalize Psi.
 */
amrex::Real IntegrateRhoPsi (amrex::MultiFab const& rho, amrex::MultiFab const& psi, int lev)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        rho.boxArray() == psi.boxArray() &&
        rho.DistributionMap() == psi.DistributionMap() &&
        rho.ixType() == psi.ixType(),
        "IntegrateRhoPsi: rho and psi must use the same nodal layout");

    auto& warpx = WarpX::GetInstance();
    amrex::MultiFab product(rho.boxArray(), rho.DistributionMap(), 1, 0);
    const auto dx = warpx.Geom(lev).CellSizeArray();
#ifdef WARPX_DIM_RZ
    const amrex::Real dr = dx[0];
    const amrex::Real dz = dx[1];
    const amrex::Real rlo = warpx.Geom(lev).ProbLo(0);
    const amrex::Real axis_factor = warpx.RZAxisVolumeFactor();
#else
    const amrex::Real node_volume = dx[0] * dx[1] * dx[2];
#endif

    for (amrex::MFIter mfi(product); mfi.isValid(); ++mfi) {
        auto const& qa = rho.const_array(mfi);
        auto const& pa = psi.const_array(mfi);
        auto const& wa = product.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
#ifdef WARPX_DIM_RZ
                const amrex::Real r = rlo + amrex::Real(i)*dr;
                const amrex::Real radial_measure = (r == 0._rt)
                    ? MathConst::pi*dr*axis_factor : 2._rt*MathConst::pi*r;
                const amrex::Real node_volume = dr*dz*radial_measure;
#endif
                wa(i,j,k) = qa(i,j,k,0) * pa(i,j,k,0) * node_volume;
            });
    }

    return product.sum_unique(0, false, warpx.Geom(lev).periodicity());
}

} // namespace

/* The charge functional is stated on EDGES (E at the cut-cell surface) but the row
 * space of A^T is NODAL, so the right-hand side is built in two passes:
 *
 *   A. cut-cell surface loop -> one coefficient per edge, using the same stencil
 *      as WeightedChargeOnEB;
 *   B. edge coefficients -> nodal rhs, through the exact transpose of AMReX's
 *      mlebndfdlap_grad_*_doit with E = -grad(phi) and a grounded EB.
 *
 * Both passes need SumBoundary: their source is a cell or an edge, unique to one
 * box, so a cut cell near a box face can target a node owned by the neighbour.
 */
namespace {

/** Scatter one edge coefficient onto its endpoint nodes, the transpose of
 * AMReX's mlebndfdlap_grad_*_doit row for a grounded embedded boundary.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void ScatterEdgeCoefficient (amrex::Real f, amrex::Real dxi, amrex::Real ec,
                             bool lo_cov, bool hi_cov,
                             amrex::Real* lo_node, amrex::Real* hi_node,
                             amrex::Long* skipped) noexcept
{
    using amrex::Real;
    constexpr Real singular_tol = Real(1.0e-14);

    // both covered: the forward stencil returns zero, so there is no equation
    if (lo_cov && hi_cov) { return; }

    if (!lo_cov && !hi_cov) {
        amrex::Gpu::Atomic::AddNoRet(hi_node, -f*dxi);
        amrex::Gpu::Atomic::AddNoRet(lo_node,  f*dxi);
    } else if (lo_cov) {
        // A centroid of 1 marks an uncut edge, which happens when the EB is
        // aligned with the grid: the full edge length is used (AMReX PR #5623).
        const Real h = (ec == Real(1.0)) ? Real(1.0) : Real(1.0) - Real(2.0)*ec;
        if (amrex::Math::abs(h) < singular_tol) {
            amrex::HostDevice::Atomic::Add(skipped, amrex::Long(1));
        } else {
            amrex::Gpu::Atomic::AddNoRet(hi_node, -f*dxi/h);
        }
    } else {
        const Real h = (ec == Real(1.0)) ? Real(1.0) : Real(1.0) + Real(2.0)*ec;
        if (amrex::Math::abs(h) < singular_tol) {
            amrex::HostDevice::Atomic::Add(skipped, amrex::Long(1));
        } else {
            amrex::Gpu::Atomic::AddNoRet(lo_node, f*dxi/h);
        }
    }
}

} // namespace

void WarpXBuildAdjointRHSChargeFunctional (amrex::MultiFab& rhs,
                                           std::string const& region,
                                           amrex::iMultiFab const& dmsk,
                                           int lev)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    auto& warpx = WarpX::GetInstance();
    auto const& eb_fact = warpx.fieldEBFactory(lev);
    auto const& levset = eb_fact.getLevelSet();
    auto const& edge_cent = eb_fact.getEdgeCent();
    auto const& eb_flag = eb_fact.getMultiEBCellFlagFab();
    auto const& eb_bnd_cent = eb_fact.getBndryCent();
    auto const& eb_bnd_normal = eb_fact.getBndryNormal();
    auto const eb_area_fraction = eb_fact.getAreaFrac();

    const auto dx = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    const amrex::Periodicity& period = warpx.Geom(lev).periodicity();

    amrex::Parser rparser = utils::parser::makeParser(region, {"x","y","z"});
    auto fun_w = utils::parser::compileParser<3>(&rparser);

    // The edge coefficient fields only need Efield_fp's layout; their values are
    // never read, so the right-hand side is a purely geometric functional.
    amrex::Vector<amrex::MultiFab> f(AMREX_SPACEDIM);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        // in RZ the second edge direction is z, i.e. Efield_fp component 2
        const int comp = (AMREX_SPACEDIM == 2 && idim == 1) ? 2 : idim;
        amrex::MultiFab const& E = *warpx.m_fields.get(FieldType::Efield_fp, Direction{comp}, lev);
        f[idim].define(E.boxArray(), E.DistributionMap(), 1, 1);
        f[idim].setVal(0.0);
    }

    // count of cut edges whose transpose denominator was too close to zero
    amrex::Gpu::Buffer<amrex::Long> skip_buf({amrex::Long(0)});
    amrex::Long* skip_ptr = skip_buf.data();

    // ---- pass A: cut-cell surface loop -> edge coefficients ---------------
    for (amrex::MFIter mfi(f[0]); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.tilebox(amrex::IntVect::TheCellVector());
        const amrex::FabType fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue; }
        if (fab_type == amrex::FabType::covered) { continue; }

        auto const& flag = eb_flag.const_array(mfi);
        auto const& normal = eb_bnd_normal.const_array(mfi);
        auto const& cent = eb_bnd_cent.const_array(mfi);
        auto const& a0 = eb_area_fraction[0]->const_array(mfi);
        auto const& a1 = eb_area_fraction[1]->const_array(mfi);
        auto const& f0 = f[0].array(mfi);
        auto const& f1 = f[1].array(mfi);
#if (AMREX_SPACEDIM == 3)
        auto const& a2 = eb_area_fraction[2]->const_array(mfi);
        auto const& f2 = f[2].array(mfi);
#endif

        amrex::For(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (flag(i,j,k).isRegular() || flag(i,j,k).isCovered()) { return; }

                const int in = (normal(i,j,k,0) > 0._rt) ? i : i+1;
                const int jn = (normal(i,j,k,1) > 0._rt) ? j : j+1;
                int ic = i;
                if (normal(i,j,k,0) > 0._rt && cent(i,j,k,0) <= 0._rt) { --ic; }
                if (normal(i,j,k,0) < 0._rt && cent(i,j,k,0) >= 0._rt) { ++ic; }
                int jc = j;
                if (normal(i,j,k,1) > 0._rt && cent(i,j,k,1) <= 0._rt) { --jc; }
                if (normal(i,j,k,1) < 0._rt && cent(i,j,k,1) >= 0._rt) { ++jc; }

                const amrex::Real c0 = (i + 0.5_rt + cent(i,j,k,0))*dx[0] + real_box.lo(0);
                const amrex::Real c1 = (j + 0.5_rt + cent(i,j,k,1))*dx[1] + real_box.lo(1);

#if (AMREX_SPACEDIM == 3)
                const int kn = (normal(i,j,k,2) > 0._rt) ? k : k+1;
                int kc = k;
                if (normal(i,j,k,2) > 0._rt && cent(i,j,k,2) <= 0._rt) { --kc; }
                if (normal(i,j,k,2) < 0._rt && cent(i,j,k,2) >= 0._rt) { ++kc; }
                const amrex::Real c2 = (k + 0.5_rt + cent(i,j,k,2))*dx[2] + real_box.lo(2);

                const amrex::Real pref = PhysConst::epsilon_0 * fun_w(c0, c1, c2);
                amrex::Gpu::Atomic::AddNoRet(&f0(ic,jn,kn),
                    pref*dx[1]*dx[2]*(a0(i+1,j,k)-a0(i,j,k)));
                amrex::Gpu::Atomic::AddNoRet(&f1(in,jc,kn),
                    pref*dx[2]*dx[0]*(a1(i,j+1,k)-a1(i,j,k)));
                amrex::Gpu::Atomic::AddNoRet(&f2(in,jn,kc),
                    pref*dx[0]*dx[1]*(a2(i,j,k+1)-a2(i,j,k)));
#else
                // RZ: c0 is the radius of the surface centroid and c1 is z; the
                // 2*pi*r measure is the one WeightedChargeOnEB applies.
                const amrex::Real pref = PhysConst::epsilon_0 * 2._rt * MathConst::pi * c0
                    * fun_w(c0, 0._rt, c1);
                amrex::Gpu::Atomic::AddNoRet(&f0(ic,jn,k), pref*dx[1]*(a0(i+1,j,k)-a0(i,j,k)));
                amrex::Gpu::Atomic::AddNoRet(&f1(in,jc,k), pref*dx[0]*(a1(i,j+1,k)-a1(i,j,k)));
#endif
            });
    }
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) { f[idim].SumBoundary(period); }

    // ---- pass B: edge coefficients -> nodal rhs ---------------------------
    // "covered" is the EB level set, exactly as mlebndfdlap_grad_*_doit tests it,
    // NOT dmsk, which also marks the outer walls. An edge straddling a grounded
    // wall deposits normally and the wall node is discarded below.
    rhs.setVal(0.0);
    const EdgeCentFallback ec_fallback(lev);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const amrex::Real dxi = 1._rt / dx[idim];
        auto const owner = f[idim].OwnerMask(period);
        // the edge runs from node (i,j,k) to that node shifted by one in idim
        const int si = (idim == 0) ? 1 : 0;
        const int sj = (idim == 1) ? 1 : 0;
        const int sk = (idim == 2) ? 1 : 0;

        for (amrex::MFIter mfi(f[idim]); mfi.isValid(); ++mfi)
        {
            auto const& fa = f[idim].const_array(mfi);
            auto const& ls = levset.const_array(mfi);
            auto const& ec = ec_fallback.array(edge_cent, mfi, idim);
            auto const& ra = rhs.array(mfi);
            auto const& own = owner->const_array(mfi);

            // amrex::For: this is a scatter onto neighbouring nodes
            amrex::For(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (own(i,j,k) == 0) { return; }
                    const amrex::Real fv = fa(i,j,k);
                    if (fv == 0._rt) { return; }
                    ScatterEdgeCoefficient(
                        fv, dxi, ec(i,j,k),
                        ls(i,j,k) >= 0._rt, ls(i+si,j+sj,k+sk) >= 0._rt,
                        &ra(i,j,k), &ra(i+si,j+sj,k+sk), skip_ptr);
                });
        }
    }
    rhs.SumBoundary(period);

    // constrained rows carry no equation
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        auto const& ra = rhs.array(mfi);
        auto const& dm = dmsk.const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (dm(i,j,k) != 0) { ra(i,j,k) = 0._rt; }
            });
    }
    rhs.FillBoundary(period);

    skip_buf.copyToHost();
    amrex::Long skipped = *(skip_buf.hostData());
    amrex::ParallelDescriptor::ReduceLongSum(skipped);
    if (skipped > 0) {
        amrex::Print() << "WarpXBuildAdjointRHSChargeFunctional: skipped " << skipped
                       << " cut edge(s) with a near-singular transpose denominator.\n";
    }
}

bool WarpXSolveAdjointWeighting (amrex::MultiFab& psi,
                                 amrex::MultiFab const& rhs,
                                 amrex::iMultiFab const& dmsk,
                                 int lev, amrex::Real tol, int max_iter,
                                 amrex::Real* final_res, int* iters_out)
{
    AssertGroundedOuterBoundaries();

    auto& warpx = WarpX::GetInstance();
    const amrex::BoxArray& ba = psi.boxArray();
    const amrex::DistributionMapping& dm = psi.DistributionMap();
    const int ng = psi.nGrow();
    auto const& eb_fact = warpx.fieldEBFactory(lev);

    const AdjointOperator op(ba, dm, dmsk, lev);

    // The preconditioner is built once and its MLMG object reused: paying the
    // setup per outer iteration would defeat the point of this route.
    auto linop = BuildAdjointPrecondLinOp(ba, dm, lev);
    amrex::MLMG mlmg(*linop);
    mlmg.setVerbose(0);
    mlmg.setMaxIter(100);
    mlmg.setConvergenceNormType(amrex::MLMGNormType::greater);
    mlmg.setThrowException(true);
    // 1e-4 keeps the outer iteration count flat with resolution; tightening it
    // buys nothing measurable and costs more work per apply.
    constexpr amrex::Real prec_rtol = 1.e-4_rt;

    amrex::MultiFab s(ba, dm, 1, 1);
    ComputeNodeMinScale(s, lev);
    amrex::MultiFab sinv(ba, dm, 1, 1);
    for (amrex::MFIter mfi(s); mfi.isValid(); ++mfi) {
        auto const& sa = s.const_array(mfi);
        auto const& sia = sinv.array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                constexpr amrex::Real floor_s = 1.e-6_rt;
                sia(i,j,k) = (sa(i,j,k) > floor_s) ? 1._rt/sa(i,j,k) : 1._rt;
            });
    }
    sinv.FillBoundary(warpx.Geom(lev).periodicity());

    // built with the EB factory and one ghost cell so MLMG aliases them as its
    // own solution array instead of copying
    amrex::MultiFab y(ba, dm, 1, 1, amrex::MFInfo(), eb_fact);
    amrex::MultiFab z(ba, dm, 1, 1, amrex::MFInfo(), eb_fact);

    // right-preconditioned BiCGSTAB on A^T psi = rhs
    amrex::MultiFab r(ba, dm, 1, ng), rhat(ba, dm, 1, ng);
    amrex::MultiFab p(ba, dm, 1, ng), v(ba, dm, 1, ng);
    amrex::MultiFab s_vec(ba, dm, 1, ng), t_vec(ba, dm, 1, ng);
    amrex::MultiFab true_res(ba, dm, 1, ng);

    psi.setVal(0.0);
    amrex::MultiFab::Copy(r, rhs, 0, 0, 1, 0);      // r0 = rhs - A^T*0
    amrex::MultiFab::Copy(rhat, r, 0, 0, 1, 0);
    p.setVal(0.0);
    v.setVal(0.0);

    const amrex::Real b_norm = std::sqrt(op.dot(rhs, rhs));
    if (b_norm == 0._rt) {
        if (final_res) { *final_res = 0.0; }
        if (iters_out) { *iters_out = 0; }
        return true;
    }

    amrex::Real rho = 1._rt;
    amrex::Real alpha = 1._rt;
    amrex::Real omega = 1._rt;

    bool converged = false;
    int it_used = 0;
    for (int it = 0; it < max_iter; ++it) {
        const amrex::Real rho_new = op.dot(rhat, r);
        if (rho_new == 0._rt) { break; }   // breakdown

        if (it == 0) {
            amrex::MultiFab::Copy(p, r, 0, 0, 1, 0);
        } else {
            const amrex::Real beta = (rho_new / rho) * (alpha / omega);
            amrex::MultiFab::Saxpy(p, -omega, v, 0, 0, 1, 0);  // p = p - omega*v
            amrex::MultiFab::Xpay(p, beta, r, 0, 0, 1, 0);     // p = r + beta*p
        }

        ApplyPrecond(y, p, op, lev, mlmg, prec_rtol, sinv);   // y ~= A^{-1} p
        op.apply(v, y, /*transpose=*/true);                   // v = A^T y

        const amrex::Real rhat_v = op.dot(rhat, v);
        if (rhat_v == 0._rt) { break; }
        alpha = rho_new / rhat_v;

        amrex::MultiFab::Copy(s_vec, r, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(s_vec, -alpha, v, 0, 0, 1, 0); // s = r - alpha*v

        ApplyPrecond(z, s_vec, op, lev, mlmg, prec_rtol, sinv); // z ~= A^{-1} s
        op.apply(t_vec, z, /*transpose=*/true);                // t = A^T z

        const amrex::Real tt = op.dot(t_vec, t_vec);
        omega = (tt == 0._rt) ? 0._rt : op.dot(t_vec, s_vec) / tt;

        amrex::MultiFab::Saxpy(psi, alpha, y, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(psi, omega, z, 0, 0, 1, 0);

        amrex::MultiFab::Copy(r, s_vec, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(r, -omega, t_vec, 0, 0, 1, 0); // r = s - omega*t

        it_used = it + 1;

        // convergence on the true residual, which the normal-equation residual
        // can badly under-report
        op.apply(true_res, psi, /*transpose=*/true);
        amrex::MultiFab::Subtract(true_res, rhs, 0, 0, 1, 0);
        const amrex::Real rel = std::sqrt(op.dot(true_res, true_res)) / b_norm;
        if (final_res) { *final_res = rel; }
        if (rel < tol) { converged = true; break; }

        if (omega == 0._rt) { break; }     // breakdown
        rho = rho_new;
    }
    if (iters_out) { *iters_out = it_used; }
    return converged;
}

/* MLMG multiplies the user's right-hand side by the diagonal S(i,j,k) before
 * solving, so WarpX's own solve is phi = A^{-1} S rhs_phys with
 * rhs_phys(a) = -rho_a/eps0. For the identity Q(phi) = -sum_a rho_a Psi[a] to hold
 * for every rho, Psi = S A^{-T} c / eps0, and a node carrying charge q over the
 * nodal volume dV has rho_a = q/dV, hence the 1/(eps0 dV) below.
 *
 * Covered and wall nodes are already exactly zero in psi, so S is applied
 * unconditionally rather than re-testing the mask.
 */
void WarpXFinalizeChargeFunctionalPsi (amrex::MultiFab& psi, int lev)
{
    auto& warpx = WarpX::GetInstance();
    auto const& edge_cent = warpx.fieldEBFactory(lev).getEdgeCent();

    const auto dx = warpx.Geom(lev).CellSizeArray();
#ifdef WARPX_DIM_RZ
    const amrex::Real dr = dx[0];
    const amrex::Real dz = dx[1];
    const amrex::Real rlo = warpx.Geom(lev).ProbLo(0);
    const amrex::Real axis_factor = warpx.RZAxisVolumeFactor();
#else
    const amrex::Real inv_eps0_dV = 1._rt / (PhysConst::epsilon_0 * dx[0]*dx[1]*dx[2]);
#endif

    const EdgeCentFallback ec_fallback(lev);

    for (amrex::MFIter mfi(psi); mfi.isValid(); ++mfi) {
        auto const& pa = psi.array(mfi);
        // boxes with no cut cells carry no MultiCutFab data; see EdgeCentFallback
        auto const& ecx = ec_fallback.array(edge_cent, mfi, 0);
        auto const& ecy = ec_fallback.array(edge_cent, mfi, 1);
#ifndef WARPX_DIM_RZ
        auto const& ecz = ec_fallback.array(edge_cent, mfi, 2);
#endif
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                using amrex::Real;
                const Real hpx = (ecx(i  ,j,k) == 1._rt) ? 1._rt : 1._rt + 2._rt*ecx(i  ,j,k);
                const Real hmx = (ecx(i-1,j,k) == 1._rt) ? 1._rt : 1._rt - 2._rt*ecx(i-1,j,k);
                const Real hpy = (ecy(i,j  ,k) == 1._rt) ? 1._rt : 1._rt + 2._rt*ecy(i,j  ,k);
                const Real hmy = (ecy(i,j-1,k) == 1._rt) ? 1._rt : 1._rt - 2._rt*ecy(i,j-1,k);
                Real scale = amrex::min(hmx, hpx);
                scale = amrex::min(scale, hmy, hpy);
#ifdef WARPX_DIM_RZ
                const Real r = rlo + Real(i)*dr;
                const Real radial_measure = (r == 0._rt)
                    ? MathConst::pi*dr*axis_factor : 2._rt*MathConst::pi*r;
                pa(i,j,k) *= scale/(PhysConst::epsilon_0*dr*dz*radial_measure);
#else
                const Real hpz = (ecz(i,j,k  ) == 1._rt) ? 1._rt : 1._rt + 2._rt*ecz(i,j,k  );
                const Real hmz = (ecz(i,j,k-1) == 1._rt) ? 1._rt : 1._rt - 2._rt*ecz(i,j,k-1);
                scale = amrex::min(scale, hmz, hpz);
                pa(i,j,k) *= scale * inv_eps0_dV;
#endif
            });
    }
    psi.FillBoundary(warpx.Geom(lev).periodicity());
}

amrex::Vector<amrex::Real>
WarpXGroundedChargeFromAdjoint (std::vector<std::string> const& psi_fields, int lev)
{
    auto& warpx = WarpX::GetInstance();
    auto const rho = warpx.DepositScratchRho(lev);

    amrex::Vector<amrex::Real> q_grounded;
    q_grounded.reserve(psi_fields.size());
    for (auto const& name : psi_fields) {
        amrex::MultiFab const* psi = warpx.m_fields.get(name, lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(psi != nullptr,
            "WarpXGroundedChargeFromAdjoint: weighting potential field not registered");
        q_grounded.push_back(-IntegrateRhoPsi(*rho, *psi, lev));
    }
    return q_grounded;
}
