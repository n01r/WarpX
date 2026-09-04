/* Copyright 2026 The WarpX Community
 *
 * Transpose of the MLEBNodeFDLaplacian apply, for electrode weighting potentials.
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_
#define WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_

#include <AMReX_Algorithm.H>
#include <AMReX_Array4.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

/* The electrode weighting potential Psi_k satisfies the discrete charge identity
 *
 *     Q_k = (C V)_k - sum_nodes rho * Psi_k
 *
 * only when it is built from the adjoint of the operator the solver uses,
 *
 *     A^T Psi_k = -A_DF^T e_k .
 *
 * For a symmetric A the familiar "solve the same Laplace problem with electrode k
 * at 1 V" recipe is correct, but WarpX's EB Laplacian is not symmetric at
 * partially cut cells: mlebndfdlap_adotx_eb_doit applies a per-direction factor
 * 2/(hp+hm) inside the row sum, and those factors differ between directions on
 * the same row.
 *
 * amrex::MLLinOp exposes no transposed apply, so the transpose is written here as
 * a scatter of each forward row. The forward apply used alongside it is AMReX's
 * own kernel (see AdjointWeightingSolve.cpp), so a drift between the two shows up
 * directly in the solver residual.
 *
 * Only free rows scatter, i.e. these apply A^T on the free-free block, which is
 * the operator the Krylov solve needs.
 */

/** Apply the transpose of AMReX's Cartesian EB nodal Laplacian row.
 *
 * \param i,j,k        node index
 * \param y            output, accumulated into; the caller must zero it first
 * \param x            input vector
 * \param levset       level set, negative outside the embedded boundary
 * \param dmsk         Dirichlet mask, nonzero on constrained nodes
 * \param ecx,ecy,ecz  edge centroids, as used by the forward apply
 * \param bx,by,bz     1/dx^2 per direction
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_mlebndfdlap_adotx_transpose_eb (
    int i, int j, int k,
    amrex::Array4<amrex::Real> const& y,
    amrex::Array4<amrex::Real const> const& x,
    amrex::Array4<amrex::Real const> const& levset,
    amrex::Array4<int const> const& dmsk,
    amrex::Array4<amrex::Real const> const& ecx,
    amrex::Array4<amrex::Real const> const& ecy,
    amrex::Array4<amrex::Real const> const& ecz,
    amrex::Real bx, amrex::Real by, amrex::Real bz) noexcept
{
    using amrex::Real;

    if (dmsk(i,j,k) != 0) { return; }

    // recover the same hp/hm/scale this row uses in the forward apply
    const Real hpx = (ecx(i  ,j,k) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) + Real(2.0)*ecx(i  ,j,k));
    const Real hmx = (ecx(i-1,j,k) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) - Real(2.0)*ecx(i-1,j,k));
    const Real hpy = (ecy(i,j  ,k) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) + Real(2.0)*ecy(i,j  ,k));
    const Real hmy = (ecy(i,j-1,k) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) - Real(2.0)*ecy(i,j-1,k));
    const Real hpz = (ecz(i,j,k  ) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) + Real(2.0)*ecz(i,j,k  ));
    const Real hmz = (ecz(i,j,k-1) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) - Real(2.0)*ecz(i,j,k-1));

    Real scale = amrex::min(hmx, hpx);
    scale = amrex::min(scale, hmy, hpy);
    scale = amrex::min(scale, hmz, hpz);

    // the row scale is a scalar, so it is folded into the per-direction factor
    // before the scatter
    const Real vx = x(i,j,k) * scale * bx * Real(2.0) / (hpx + hmx);
    const Real vy = x(i,j,k) * scale * by * Real(2.0) / (hpy + hmy);
    const Real vz = x(i,j,k) * scale * bz * Real(2.0) / (hpz + hmz);

    // A covered neighbour contributes only to the diagonal: its value is
    // Dirichlet data, which belongs in the right-hand side, not the operator.
    Real diag = Real(0.0);

    if (levset(i+1,j,k) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i+1,j,k), vx);
        diag -= vx;
    } else {
        diag -= vx / hpx;
    }
    if (levset(i-1,j,k) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i-1,j,k), vx);
        diag -= vx;
    } else {
        diag -= vx / hmx;
    }

    if (levset(i,j+1,k) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j+1,k), vy);
        diag -= vy;
    } else {
        diag -= vy / hpy;
    }
    if (levset(i,j-1,k) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j-1,k), vy);
        diag -= vy;
    } else {
        diag -= vy / hmy;
    }

    if (levset(i,j,k+1) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j,k+1), vz);
        diag -= vz;
    } else {
        diag -= vz / hpz;
    }
    if (levset(i,j,k-1) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j,k-1), vz);
        diag -= vz;
    } else {
        diag -= vz / hmz;
    }

    amrex::Gpu::Atomic::AddNoRet(&y(i,j,k), diag);
}

#if (AMREX_SPACEDIM == 2)
/** Apply the transpose of AMReX's metric-aware RZ EB nodal Laplacian row.
 *
 * Unlike the Cartesian stencil, the radial coefficients carry the cylindrical
 * metric r +/- dr/2 and the axis row uses the regularity limit
 * 4*(phi_1-phi_0)/dr^2. Writing the transpose as a scatter of each forward row
 * keeps those factors on the row where AMReX evaluates them.
 *
 * This mirrors mlebndfdlap_adotx_rz_eb_doit for sigma_r = 1 and alpha = 0, the
 * configuration the adjoint solve sets up (see BuildAdjointPrecondLinOp).
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_mlebndfdlap_adotx_rz_transpose_eb (
    int i, int j, int k,
    amrex::Array4<amrex::Real> const& y,
    amrex::Array4<amrex::Real const> const& x,
    amrex::Array4<amrex::Real const> const& levset,
    amrex::Array4<int const> const& dmsk,
    amrex::Array4<amrex::Real const> const& ecr,
    amrex::Array4<amrex::Real const> const& ecz,
    amrex::Real dr, amrex::Real dz, amrex::Real rlo) noexcept
{
    using amrex::Real;

    if (dmsk(i,j,k) != 0) { return; }

    const Real r = rlo + Real(i)*dr;
    const Real hpr = (ecr(i  ,j,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) + Real(2.0)*ecr(i  ,j,k);
    const Real hmr = (ecr(i-1,j,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) - Real(2.0)*ecr(i-1,j,k);
    const Real hpz = (ecz(i,j  ,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) + Real(2.0)*ecz(i,j  ,k);
    const Real hmz = (ecz(i,j-1,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) - Real(2.0)*ecz(i,j-1,k);

    Real scale;
    Real diag = Real(0.0);
    const Real x0 = x(i,j,k);
    Real crp = Real(0.0);
    Real crm = Real(0.0);

    if (r == Real(0.0)) {
        // on-axis regularity row: the i-1 neighbour does not enter
        if (levset(i+1,j,k) < Real(0.0)) {
            const Real c = Real(4.0)/(dr*dr);
            crp = c;
            diag -= c;
            scale = Real(1.0);
        } else {
            diag -= Real(4.0)/(dr*dr*hpr*hpr);
            scale = hpr;
        }
    } else {
        const Real fac = Real(2.0)/((hpr+hmr)*r*dr*dr);
        if (levset(i+1,j,k) < Real(0.0)) {
            const Real c = fac*(r + Real(0.5)*dr);
            crp = c;
            diag -= c;
        } else {
            diag -= fac*(r + Real(0.5)*hpr*dr)/hpr;
        }
        if (levset(i-1,j,k) < Real(0.0)) {
            const Real c = fac*(r - Real(0.5)*dr);
            crm = c;
            diag -= c;
        } else {
            diag -= fac*(r - Real(0.5)*hmr*dr)/hmr;
        }
        scale = amrex::min(hmr, hpr);
    }

    const Real facz = Real(2.0)/((hpz+hmz)*dz*dz);
    const bool zp_free = (levset(i,j+1,k) < Real(0.0));
    const bool zm_free = (levset(i,j-1,k) < Real(0.0));
    diag -= facz * (zp_free ? Real(1.0) : Real(1.0)/hpz);
    diag -= facz * (zm_free ? Real(1.0) : Real(1.0)/hmz);

    scale = amrex::min(scale, hmz, hpz);
    if (crp != Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i+1,j,k), x0*crp*scale);
    }
    if (crm != Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i-1,j,k), x0*crm*scale);
    }
    if (zp_free) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j+1,k), x0*facz*scale);
    }
    if (zm_free) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j-1,k), x0*facz*scale);
    }
    amrex::Gpu::Atomic::AddNoRet(&y(i,j,k), x0*diag*scale);
}
#endif

#endif // WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_
