# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Telemetry for the multi-electrode voltage clamp.

:class:`MultiElectrodeClampTelemetry` writes the state a
:class:`~pywarpx.multi_electrode_corrector.MultiElectrodeBiasCorrector` has
already measured, without repeating any measurement.

:class:`GroundedChargeCrossCheck` sparsely compares the adjoint observer against
an independent grounded Poisson solve. Every selected call is collective and
deliberately expensive.

Both write on rank zero and preserve an existing CSV when the run restarts from a
checkpoint. Install ``setup`` after the corrector's setup callback and ``log``
after its correction callback.
"""

import csv
import json
import os


def _mpi_rank():
    try:
        from mpi4py import MPI  # noqa: PLC0415

        return MPI.COMM_WORLD.Get_rank()
    except Exception:  # noqa: BLE001
        return 0


def _initialize_csv(path, header, preserve_existing):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if preserve_existing and os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="") as stream:
        csv.writer(stream).writerow(header)


class MultiElectrodeClampTelemetry:
    """Append the latest correction state to a CSV, and dump setup metadata once."""

    def __init__(
        self,
        corrector,
        period=1,
        out_csv="clamp_telemetry.csv",
        setup_json="clamp_setup.json",
        include_first=True,
    ):
        self.corrector = corrector
        self.period = int(period)
        if self.period < 1:
            raise ValueError("telemetry period must be positive")
        self.out_csv = out_csv
        self.setup_json = setup_json
        self.include_first = bool(include_first)
        self._last_written_step = None

    def setup(self):
        """Write setup metadata and initialize the CSV."""
        state = self.corrector.setup_state()
        if _mpi_rank() != 0:
            return

        parent = os.path.dirname(os.path.abspath(self.setup_json))
        if parent:
            os.makedirs(parent, exist_ok=True)
        serializable = dict(state)
        serializable["capacitance_matrix"] = state["capacitance_matrix"].tolist()
        with open(self.setup_json, "w", encoding="utf-8") as stream:
            json.dump(serializable, stream, indent=2)
            stream.write("\n")

        # A checkpoint restart has a positive current step: preserve its time
        # series. A fresh step-zero run replaces stale output in the same
        # directory.
        _initialize_csv(
            self.out_csv, self._header(), preserve_existing=state["current_step"] > 0
        )

    def _header(self):
        names = self.corrector.names
        qg_mode = self.corrector.qg_mode
        return (
            ["step", "time_s"]
            + [f"V_before_{name}_V" for name in names]
            + [f"V_after_predicted_{name}_V" for name in names]
            + [f"V_target_{name}_V" for name in names]
            + [f"V_error_before_{name}_V" for name in names]
            + [f"V_error_after_predicted_{name}_V" for name in names]
            + [f"delta_V_{name}_V" for name in names]
            + [f"Q_field_{name}_C" for name in names]
            + [f"Q_grounded_{qg_mode}_{name}_C" for name in names]
        )

    def _selected(self, step):
        return (self.include_first and step == 1) or step % self.period == 0

    def log(self):
        """Append the latest correction state without re-measuring it."""
        state = self.corrector.last_correction_state()
        if state is None:
            return
        step = int(state["step"])
        if step == self._last_written_step or not self._selected(step):
            return
        self._last_written_step = step
        if _mpi_rank() != 0:
            return

        row = [step, float(state["time"])]
        for key in (
            "voltage_before",
            "voltage_after_predicted",
            "target_voltage",
            "voltage_error_before",
            "voltage_error_after_predicted",
            "delta_voltage",
            "field_charge",
            "grounded_charge",
        ):
            row.extend(float(value) for value in state[key])
        with open(self.out_csv, "a", newline="") as stream:
            csv.writer(stream).writerow(row)


class GroundedChargeCrossCheck:
    """Sparsely compare the adjoint grounded charge with a real Poisson solve."""

    def __init__(
        self,
        corrector,
        period,
        out_csv="clamp_grounded_crosscheck.csv",
        include_first=True,
    ):
        if corrector.qg_mode != "reciprocity":
            raise ValueError(
                "GroundedChargeCrossCheck requires qg_mode='reciprocity' "
                "to compare the adjoint observer with a grounded solve"
            )
        self.corrector = corrector
        self.period = int(period)
        if self.period < 1:
            raise ValueError("grounded cross-check period must be positive")
        self.out_csv = out_csv
        self.include_first = bool(include_first)
        self._last_written_step = None

    def setup(self):
        """Initialize the comparison CSV, after the corrector's setup."""
        state = self.corrector.setup_state()
        if _mpi_rank() != 0:
            return
        names = self.corrector.names
        header = (
            ["step", "time_s", "max_relative_difference"]
            + [f"Q_grounded_adjoint_{name}_C" for name in names]
            + [f"Q_grounded_solve_{name}_C" for name in names]
            + [f"difference_{name}_C" for name in names]
            + [f"relative_difference_{name}" for name in names]
        )
        _initialize_csv(
            self.out_csv, header, preserve_existing=state["current_step"] > 0
        )

    def _selected(self, step):
        return (self.include_first and step == 1) or step % self.period == 0

    def log(self):
        """Run and record a selected grounded-solve comparison."""
        import numpy as np  # noqa: PLC0415

        state = self.corrector.last_correction_state()
        if state is None:
            return
        step = int(state["step"])
        if step == self._last_written_step or not self._selected(step):
            return
        self._last_written_step = step

        # collective: reuse the correction's own adjoint result
        comparison = self.corrector.compare_grounded_charge(state["grounded_charge"])

        if _mpi_rank() != 0:
            return
        row = [
            step,
            float(state["time"]),
            float(np.max(comparison["relative_difference"])),
        ]
        for key in (
            "reciprocity_charge",
            "solved_charge",
            "difference",
            "relative_difference",
        ):
            row.extend(float(value) for value in comparison[key])
        with open(self.out_csv, "a", newline="") as stream:
            csv.writer(stream).writerow(row)
