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

/** \brief Charge enclosed by a region, from the divergence of the solved field.
 *
 * Evaluates, for each region weight \f$w_k(x,y,z)\f$,
 * \f[
 *     Q_{div,k} = \epsilon_0 \sum_a w_k[a]\, (\nabla\cdot E)[a]\, dV_a
 * \f]
 * on the nodes, using WarpX's own ``ComputeDivE`` on ``Efield_fp`` and the same
 * nodal control volumes as the charge integration elsewhere in this file.
 *
 * This is the volume counterpart of ``WeightedChargeOnEB``. It is the preferred
 * observer for an electrode's charge, because a region bounded in live fluid
 * does not read the cut cells whose ``E`` the staircase FDTD update freezes:
 * the surface integral samples exactly those cells, and reads a fraction of the
 * charge that does not converge under refinement. Measured requirement: about
 * two cells of vacuum clearance between the metal and the region boundary, a
 * figure that is constant in cells and so shrinks as the grid is refined.
 *
 * Shared nodes on box boundaries are counted exactly once, so the result is
 * independent of the domain decomposition and of the number of ranks.
 *
 * \f$Q_{div}\f$ is the **total** enclosed charge. Subtract
 * ``WarpXLiveChargeInRegions`` over the same weights to isolate the charge on
 * the electrode itself.
 *
 * @param[in] regions parser expressions \f$w_k(x,y,z)\f$, one per electrode
 * @param[in] lev refinement level
 */
amrex::Vector<amrex::Real>
WarpXDivEChargeInRegions (std::vector<std::string> const& regions, int lev);

/** \brief Deposited live charge inside each region.
 *
 * Evaluates \f$\sum_a w_k[a]\, \rho_a\, dV_a\f$ against a freshly deposited
 * scratch charge density, on the same nodes and with the same measure as
 * ``WarpXDivEChargeInRegions``, so that the difference of the two isolates the
 * charge residing on the conductor.
 *
 * @param[in] regions parser expressions \f$w_k(x,y,z)\f$, one per electrode
 * @param[in] lev refinement level
 */
amrex::Vector<amrex::Real>
WarpXLiveChargeInRegions (std::vector<std::string> const& regions, int lev);

#endif // WARPX_ADJOINT_WEIGHTING_SOLVE_H_
