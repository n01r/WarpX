/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ADJOINT_WEIGHTING_SOLVE_H_
#define WARPX_ADJOINT_WEIGHTING_SOLVE_H_

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>
#include <AMReX_iMultiFab.H>

#include <string>
#include <vector>

/** \brief Right-hand side of the adjoint weighting problem for one electrode.
 *
 * Builds (A_DF)^T applied to the charge functional
 * \f$Q(\phi) = \epsilon_0 \oint w\, E\cdot n\, dS\f$ that WarpX books charge with,
 * restricted to the free nodes.
 *
 * @param[out] rhs nodal right-hand side, must have at least one ghost cell
 * @param[in] region weighting \f$w(x,y,z)\f$ selecting the electrode surface
 * @param[in] dmsk Dirichlet mask, nonzero on constrained nodes
 * @param[in] lev refinement level
 */
void WarpXBuildAdjointRHSChargeFunctional (amrex::MultiFab& rhs,
                                           std::string const& region,
                                           amrex::iMultiFab const& dmsk,
                                           int lev);

/** \brief Solve \f$A^T \Psi = rhs\f$ for the adjoint weighting potential.
 *
 * BiCGSTAB on \f$A^T\f$, right-preconditioned by a forward MLMG solve of WarpX's
 * own EB Laplacian. Convergence is judged on the true residual
 * \f$|A^T\Psi - rhs|/|rhs|\f$.
 *
 * @param[out] psi nodal weighting potential
 * @param[in] rhs right-hand side from WarpXBuildAdjointRHSChargeFunctional
 * @param[in] dmsk Dirichlet mask, nonzero on constrained nodes
 * @param[in] lev refinement level
 * @param[in] tol relative residual tolerance
 * @param[in] max_iter maximum number of outer iterations
 * @param[out] final_res relative residual reached, may be nullptr
 * @param[out] iters_out outer iterations used, may be nullptr
 * @return true if the true residual reached tol. An unconverged Psi yields a
 *         wrong correction that looks plausible, so the caller must check this.
 */
bool WarpXSolveAdjointWeighting (amrex::MultiFab& psi,
                                 amrex::MultiFab const& rhs,
                                 amrex::iMultiFab const& dmsk,
                                 int lev, amrex::Real tol, int max_iter,
                                 amrex::Real* final_res = nullptr,
                                 int* iters_out = nullptr);

/** \brief Rescale a solved Psi into the weighting potential the charge identity needs.
 *
 * Applies MLMG's right-hand-side diagonal and the per-unit-charge normalization
 * \f$1/(\epsilon_0 dV)\f$, with WarpX's cylindrical nodal volume in RZ.
 */
void WarpXFinalizeChargeFunctionalPsi (amrex::MultiFab& psi, int lev);

/** \brief Plasma-induced charge on each electrode with all electrodes grounded.
 *
 * Deposits every species onto a scratch nodal charge density and evaluates
 * \f$Q_{g,k} = -\sum_a \rho_a \Psi_k[a] dV\f$ without a Poisson solve. Shared
 * nodes on box boundaries are counted exactly once.
 *
 * @param[in] psi_fields registered nodal weighting potentials, one per electrode
 * @param[in] lev refinement level
 */
amrex::Vector<amrex::Real>
WarpXGroundedChargeFromAdjoint (std::vector<std::string> const& psi_fields, int lev);

#endif // WARPX_ADJOINT_WEIGHTING_SOLVE_H_
