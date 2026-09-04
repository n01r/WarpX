/* Copyright 2023 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_CHARGEONEB_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_CHARGEONEB_H_

#include "ReducedDiags.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Parser.H>
#include <AMReX_REAL.H>

#include <string>

/**
 * \brief Weighted induced charge on the embedded boundary.
 *
 * Computes \f$\epsilon_0 \oint w(x,y,z)\, \mathbf{E}\cdot\mathbf{n}\, dS\f$ over the
 * embedded boundary, with the weighting evaluated at the boundary centroid.
 * Implemented for 3D and RZ (axisymmetric, m=0).
 *
 * @param[in] Efield field to integrate, e.g. Efield_fp or a stored unit field
 * @param[in] lev refinement level
 * @param[in] weighting optional weighting \f$w(x,y,z)\f$; nullptr means \f$w=1\f$
 */
amrex::Real
WeightedChargeOnEB (ablastr::fields::VectorField const & Efield, int lev,
                    amrex::Parser const * weighting = nullptr);

/**
 *  This class mainly contains a function that
 *  computes the total charge at the surface
 *  of the embedded boundary, by using the formula
 *  \f$Q_{tot} = \epsilon_0 \int\int dS \cdot E\f$
 *  where the integral is performed over the
 *  surface of the embedded boundary.
 *
 *  If a weighting function is provided, this computes
 *  \f$Q_{tot} = \epsilon_0 \int\int dS \cdot E \times w(x, y, z)\f$
 */
class ChargeOnEB : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    explicit ChargeOnEB (const std::string& rd_name);

    /**
     * This function computes the charge at the surface of the EB:
     * \f$Q_{tot} = \epsilon_0 \int\int dS \cdot E\f$
     *  where the integral is performed over the EB surface
     *
     * @param[in] step current time step
     */
    void ComputeDiags (int step) final;

private:
    /// Optional parser to add weight inside the integral
    std::unique_ptr<amrex::Parser> m_parser_weighting;
    /// Whether the weighting is activated
    bool m_do_parser_weighting = false;

};

#endif
