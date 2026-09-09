# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb_mixed_bc  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb_mixed_bc  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000001"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb_picmi  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb_picmi.py  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_adjoint_weighting_decomposed_picmi  # name
        3  # dims
        2  # nprocs (cut, fully regular and fully covered boxes)
        "inputs_test_3d_adjoint_weighting_picmi.py --decomposed"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_adjoint_weighting_picmi  # name
        3  # dims
        2  # nprocs (also exercises shared nodal box interfaces)
        inputs_test_3d_adjoint_weighting_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_adjoint_weighting_ordinary_actuator_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_adjoint_weighting_picmi.py --actuator-gradient ordinary"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_adjoint_weighting_picmi  # name
        RZ  # dims
        2  # nprocs (also exercises shared nodal box interfaces)
        inputs_test_rz_adjoint_weighting_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_adjoint_weighting_ordinary_actuator_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_adjoint_weighting_picmi.py --actuator-gradient ordinary"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

# The clamp's actuator field must be compatible with the Yee Faraday update.
# Both directions are registered: the ordinary gradient must leave B at roundoff,
# and the EB-aware gradient must still grow it, so the two stay distinguishable.
if(WarpX_EB)
    add_warpx_test(
        test_rz_clamp_actuator_faraday_ordinary_picmi  # name
        RZ  # dims
        2  # nprocs (the domain splits in r as well as z)
        "inputs_test_rz_clamp_actuator_faraday_picmi.py --actuator-gradient ordinary"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_clamp_actuator_faraday_eb_aware_picmi  # name
        RZ  # dims
        2  # nprocs (the domain splits in r as well as z)
        "inputs_test_rz_clamp_actuator_faraday_picmi.py --actuator-gradient eb_aware"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_adjoint_weighting_decomposed_picmi  # name
        RZ  # dims
        2  # nprocs (the domain also splits radially, off the conductor)
        inputs_test_rz_adjoint_weighting_decomposed_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_sphere_eb  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_electrostatic_sphere_eb  # inputs
        "analysis_rz.py diags/diag1000001"  # analysis
        "analysis_default_regression.py --path diags/diag1000001 --skip-particles"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_sphere_eb_mr  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_electrostatic_sphere_eb_mr  # inputs
        "analysis_rz_mr.py diags/diag1/"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()
