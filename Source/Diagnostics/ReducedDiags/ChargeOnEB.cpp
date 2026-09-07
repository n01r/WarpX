/* Copyright 2023 Remi Lehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ChargeOnEB.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"
#include "Utils/Parser/ParserUtils.H"
#include "WarpX.H"

#include <AMReX_Array.H>
#include <AMReX_Config.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_Extension.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace amrex;


// constructor
ChargeOnEB::ChargeOnEB (const std::string& rd_name)
: ReducedDiags{rd_name}
{
    // Only 3D is working for now
#if !(defined WARPX_DIM_3D)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "ChargeOnEB reduced diagnostics only works in 3D");
#endif

#if !(defined AMREX_USE_EB)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "ChargeOnEB reduced diagnostics only works when compiling with EB support");
#endif

    if (!EB::enabled()) {
        throw std::runtime_error("ChargeOnEB reduced diagnostics only works when EBs are enabled at runtime");
    }

    // resize data array
    m_data.resize(1, 0.0_rt);

    // Read optional weighting
    std::string buf;
    const amrex::ParmParse pp_rd_name(rd_name);
    m_do_parser_weighting = pp_rd_name.query("weighting_function(x,y,z)", buf);
    if (m_do_parser_weighting) {
        std::string weighting_string;
        utils::parser::Store_parserString(
            pp_rd_name,"weighting_function(x,y,z)", weighting_string);
        m_parser_weighting = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(weighting_string,{"x","y","z"}));
    }

    if (ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            // write header row
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]Charge (C)\n";
            // close file
            ofs.close();
        }
    }
}
// end constructor


// function that computes the charge at the surface of the EB
void ChargeOnEB::ComputeDiags (const int step)
{
    // Judge whether the diags should be done
    if (!m_intervals.contains(step+1)) { return; }

    if (!EB::enabled()) {
        throw std::runtime_error("ChargeOnEB::ComputeDiags only works when EBs are enabled at runtime");
    }
#if ((defined WARPX_DIM_3D) && (defined AMREX_USE_EB))
    using warpx::fields::FieldType;

    // get a reference to WarpX instance
    auto & warpx = WarpX::GetInstance();

    // Only compute the integral on level 0
    int const lev = 0;

    m_data[0] = WeightedChargeOnEB(
        warpx.m_fields.get_alldirs(FieldType::Efield_fp, lev), lev,
        m_do_parser_weighting ? m_parser_weighting.get() : nullptr);
#endif
}
// end void ChargeOnEB::ComputeDiags

amrex::Real
WeightedChargeOnEB (
    ablastr::fields::VectorField const & Efield,
    int const lev,
    amrex::Parser const * const weighting)
{
#if (defined AMREX_USE_EB) && ((defined WARPX_DIM_3D) || (defined WARPX_DIM_RZ))
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(EB::enabled(),
        "WeightedChargeOnEB requires embedded boundaries to be enabled");

    auto & warpx = WarpX::GetInstance();

    // get EB structures
    amrex::EBFArrayBoxFactory const& eb_box_factory = warpx.fieldEBFactory(lev);
    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_box_factory.getMultiEBCellFlagFab();
    amrex::MultiCutFab const& eb_bnd_cent = eb_box_factory.getBndryCent();
    amrex::MultiCutFab const& eb_bnd_normal = eb_box_factory.getBndryNormal();
    amrex::Array<const amrex::MultiCutFab*,AMREX_SPACEDIM> eb_area_fraction =
        eb_box_factory.getAreaFrac();

    const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dx = warpx.Geom(lev).CellSizeArray();
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();

    // optional spatial weighting w(x,y,z); compileParser handles a null parser
    const bool do_weighting = (weighting != nullptr);
    auto fun_weightingparser = utils::parser::compileParser<3>(weighting);

    // The kernel below samples Efield one cell outside the cell it integrates:
    // i_c is shifted to i-1 or i+1, and the nodal indices i_n/j_n/k_n may be
    // i+1/j+1/k+1. A cut cell touching a box boundary therefore reads into the
    // ghost region. Callers that drive an electrostatic solve at run time
    // (set_potential_on_eb + solve_poisson_efield) leave those ghosts stale, so
    // without this fill the integral depends on the domain decomposition: with
    // one box the embedded boundary is interior and every such read lands in
    // valid data, while a decomposed run reads uninitialised ghosts and returns
    // a different charge. Fill them here so the result is a function of the
    // valid data alone and no caller has to know about this precondition.
    // VectorField is std::array<MultiFab*,3>, so the pointees are non-const.
    const amrex::Periodicity period = warpx.Geom(lev).periodicity();
    for (int idim = 0; idim < 3; ++idim) {
        if (Efield[idim] != nullptr) { Efield[idim]->FillBoundary(period); }
    }

    amrex::Gpu::Buffer<amrex::Real> surface_integral({0.0_rt});
    amrex::Real* surface_integral_pointer = surface_integral.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(*Efield[0], amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box & box = mfi.tilebox( amrex::IntVect::TheCellVector() );

        // Skip boxes that do not intersect with the embedded boundary
        // (i.e. either fully covered or fully regular)
        const amrex::FabType fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue; }
        if (fab_type == amrex::FabType::covered) { continue; }

        auto const& eb_flag_arr = eb_flag.array(mfi);
        const amrex::Array4<const amrex::Real> & eb_bnd_normal_arr = eb_bnd_normal.array(mfi);
        const amrex::Array4<const amrex::Real> & eb_bnd_cent_arr = eb_bnd_cent.array(mfi);

#if (defined WARPX_DIM_3D)
        const amrex::Array4<const amrex::Real> & Ex_arr = Efield[0]->array(mfi);
        const amrex::Array4<const amrex::Real> & Ey_arr = Efield[1]->array(mfi);
        const amrex::Array4<const amrex::Real> & Ez_arr = Efield[2]->array(mfi);
        const amrex::Array4<const amrex::Real> & dSx_fraction_arr = eb_area_fraction[0]->array(mfi);
        const amrex::Array4<const amrex::Real> & dSy_fraction_arr = eb_area_fraction[1]->array(mfi);
        const amrex::Array4<const amrex::Real> & dSz_fraction_arr = eb_area_fraction[2]->array(mfi);
        amrex::Real const dSx = dx[1]*dx[2];
        amrex::Real const dSy = dx[2]*dx[0];
        amrex::Real const dSz = dx[0]*dx[1];
#else
        // RZ (axisymmetric, m=0): only Er and Ez contribute to the flux
        const amrex::Array4<const amrex::Real> & Er_arr = Efield[0]->array(mfi);
        const amrex::Array4<const amrex::Real> & Ez_arr = Efield[2]->array(mfi);
        const amrex::Array4<const amrex::Real> & dSr_fraction_arr = eb_area_fraction[0]->array(mfi);
        const amrex::Array4<const amrex::Real> & dSz_fraction_arr = eb_area_fraction[1]->array(mfi);
#endif

        // amrex::For: iterations accumulate into the shared surface integral;
        // in serial (non-OpenMP) builds HostDevice::Atomic::Add is a plain +=,
        // which is unsafe under the SIMD pragma of ParallelFor (see issue #7097)
        amrex::For( box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                // Only cells that are partially covered do contribute to the integral
                if (eb_flag_arr(i,j,k).isRegular() || eb_flag_arr(i,j,k).isCovered()) { return; }

                // Find nodal point which is outside of the EB
                // (eb_normal points towards the *interior* of the EB)
                int const i_n = (eb_bnd_normal_arr(i,j,k,0) > 0)? i : i+1;
                int const j_n = (eb_bnd_normal_arr(i,j,k,1) > 0)? j : j+1;

                // Find cell-centered point which is outside of the EB
                int i_c = i;
                if ((eb_bnd_normal_arr(i,j,k,0)>0) && (eb_bnd_cent_arr(i,j,k,0)<=0)) { i_c -= 1; }
                if ((eb_bnd_normal_arr(i,j,k,0)<0) && (eb_bnd_cent_arr(i,j,k,0)>=0)) { i_c += 1; }
                int j_c = j;
                if ((eb_bnd_normal_arr(i,j,k,1)>0) && (eb_bnd_cent_arr(i,j,k,1)<=0)) { j_c -= 1; }
                if ((eb_bnd_normal_arr(i,j,k,1)<0) && (eb_bnd_cent_arr(i,j,k,1)>=0)) { j_c += 1; }

#if (defined WARPX_DIM_3D)
                int const k_n = (eb_bnd_normal_arr(i,j,k,2) > 0)? k : k+1;
                int k_c = k;
                if ((eb_bnd_normal_arr(i,j,k,2)>0) && (eb_bnd_cent_arr(i,j,k,2)<=0)) { k_c -= 1; }
                if ((eb_bnd_normal_arr(i,j,k,2)<0) && (eb_bnd_cent_arr(i,j,k,2)>=0)) { k_c += 1; }

                // Contribution to the surface integral $\int dS \cdot E$
                amrex::Real local_integral_contribution = 0;
                local_integral_contribution += Ex_arr(i_c,j_n,k_n)*dSx
                    *(dSx_fraction_arr(i+1,j,k)-dSx_fraction_arr(i,j,k));
                local_integral_contribution += Ey_arr(i_n,j_c,k_n)*dSy
                    *(dSy_fraction_arr(i,j+1,k)-dSy_fraction_arr(i,j,k));
                local_integral_contribution += Ez_arr(i_n,j_n,k_c)*dSz
                    *(dSz_fraction_arr(i,j,k+1)-dSz_fraction_arr(i,j,k));

                // Add weighting if requested by user
                if (do_weighting) {
                    // 3D position of the centroid of the surface element
                    const amrex::Real x =
                        (i + 0.5_rt + eb_bnd_cent_arr(i,j,k,0))*dx[0] + real_box.lo(0);
                    const amrex::Real y =
                        (j + 0.5_rt + eb_bnd_cent_arr(i,j,k,1))*dx[1] + real_box.lo(1);
                    const amrex::Real z =
                        (k + 0.5_rt + eb_bnd_cent_arr(i,j,k,2))*dx[2] + real_box.lo(2);
                    local_integral_contribution *= fun_weightingparser(x, y, z);
                }
#else
                // AMReX stores the RZ area fractions as bare 2D-Cartesian
                // quantities (ScaleAreas multiplies by cell_size only), so the
                // cylindrical measure 2*pi*r is applied here, at the radius of
                // the EB surface centroid.
                const amrex::Real r =
                    (i + 0.5_rt + eb_bnd_cent_arr(i,j,k,0))*dx[0] + real_box.lo(0);
                amrex::Real local_integral_contribution =
                    2._rt * MathConst::pi * r * (
                      Er_arr(i_c,j_n,k)*dx[1]*(dSr_fraction_arr(i+1,j,k)-dSr_fraction_arr(i,j,k))
                    + Ez_arr(i_n,j_c,k)*dx[0]*(dSz_fraction_arr(i,j+1,k)-dSz_fraction_arr(i,j,k)) );

                if (do_weighting) {
                    // in RZ the parser's x is the radius and y is zero
                    const amrex::Real z =
                        (j + 0.5_rt + eb_bnd_cent_arr(i,j,k,1))*dx[1] + real_box.lo(1);
                    local_integral_contribution *= fun_weightingparser(r, 0._rt, z);
                }
#endif
                amrex::HostDevice::Atomic::Add( surface_integral_pointer,
                                                local_integral_contribution );
        });
    }

    // Reduce across MPI ranks
    surface_integral.copyToHost();
    amrex::Real surface_integral_value = *(surface_integral.hostData());
    amrex::ParallelDescriptor::ReduceRealSum( surface_integral_value );
    return PhysConst::epsilon_0 * surface_integral_value;

#else
    amrex::ignore_unused(Efield, lev, weighting);
    WARPX_ABORT_WITH_MESSAGE(
        "WeightedChargeOnEB is only implemented for 3D and RZ with embedded boundaries");
    return 0._rt;
#endif
}
// end amrex::Real WeightedChargeOnEB
