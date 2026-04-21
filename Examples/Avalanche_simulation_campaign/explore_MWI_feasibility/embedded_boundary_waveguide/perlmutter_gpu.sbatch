#!/bin/bash -l

# Copyright 2021-2023 Axel Huebl, Kevin Gott
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

#SBATCH -t 14:00:00
#SBATCH -N 4
#SBATCH -J WarpX
#    note: <proj> must end on _g
#SBATCH -A m4608_g
#SBATCH -q premium
# A100 40GB (most nodes)
#SBATCH -C gpu
# A100 80GB (256 nodes)
#S BATCH -C gpu&hbm80g
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
# ideally single:1, but NERSC cgroups issue
#SBATCH --gpu-bind=none
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -o WarpX.o%j
#SBATCH -e WarpX.e%j
# Send SIGHUP to the batch script 6 minutes before walltime expires.
# WarpX catches HUP via warpx.break_signals=HUP, finishes the current
# timestep, writes a checkpoint (dump_last_timestep=1), and exits cleanly.
#SBATCH --signal=B:HUP@360

# python interpreter & script here
EXE=python3
INPUTS=inputs_microwave_vacuum_te10_scooter_section_70ns.py
# or executable & inputs file
#EXE=./warpx
#INPUTS=inputs

# environment setup
if [[ -z "${MY_PROFILE}" ]]; then
    echo "WARNING: FORGOT TO"
    echo "   source $HOME/perlmutter_gpu_warpx.profile"
    echo "before submission. Doing that now."

    source $HOME/perlmutter_gpu_warpx.profile
fi

# pin to closest NIC to GPU
export MPICH_OFI_NIC_POLICY=GPU

# threads for OpenMP and threaded compressors per MPI rank
#   note: 16 avoids hyperthreading (32 virtual cores, 16 physical)
export OMP_NUM_THREADS=16

# GPU-aware MPI optimizations
export AMREX_DEFAULT_INIT="amrex.use_gpu_aware_mpi=1"

OUTPUT_FILE="output_${SLURM_JOBID}.txt"

# ── Watchdog ──────────────────────────────────────────────────────────────────
# Polls the output file every 10 minutes.  If it has not grown since the last
# check, WarpX is assumed to be hung: send SIGHUP to the srun process group so
# WarpX can attempt a graceful checkpoint-and-exit.  If it still has not exited
# after a further 5 minutes, send SIGKILL to force termination.
WATCHDOG_INTERVAL=600   # seconds between checks (10 min)
WATCHDOG_GRACE=300      # seconds to wait after SIGHUP before SIGKILL (5 min)

watchdog() {
    local srun_pid=$1
    local last_size=-1

    echo "[watchdog] started, monitoring ${OUTPUT_FILE} (pid ${srun_pid})" >&2

    while kill -0 "${srun_pid}" 2>/dev/null; do
        sleep "${WATCHDOG_INTERVAL}"

        # If srun has already exited naturally, stop watching
        kill -0 "${srun_pid}" 2>/dev/null || break

        local cur_size=0
        if [[ -f "${OUTPUT_FILE}" ]]; then
            cur_size=$(stat -c%s "${OUTPUT_FILE}" 2>/dev/null || echo 0)
        fi

        if [[ "${cur_size}" -le "${last_size}" ]]; then
            echo "[watchdog] $(date): no output growth detected \
(${cur_size} B). Sending SIGHUP to srun process group ${srun_pid}." >&2
            # Send to the process group so all MPI ranks receive the signal
            kill -HUP -"${srun_pid}" 2>/dev/null || kill -HUP "${srun_pid}"

            # Give WarpX time to checkpoint gracefully
            local waited=0
            while (( waited < WATCHDOG_GRACE )) && kill -0 "${srun_pid}" 2>/dev/null; do
                sleep 10
                (( waited += 10 ))
            done

            # If still alive, force-kill
            if kill -0 "${srun_pid}" 2>/dev/null; then
                echo "[watchdog] $(date): still alive after ${WATCHDOG_GRACE}s, sending SIGKILL." >&2
                kill -KILL -"${srun_pid}" 2>/dev/null || kill -KILL "${srun_pid}"
            else
                echo "[watchdog] $(date): srun exited gracefully." >&2
            fi
            break
        fi

        last_size="${cur_size}"
        echo "[watchdog] $(date): output file is ${cur_size} B — simulation is running." >&2
    done

    echo "[watchdog] exiting." >&2
}

# ── Near-walltime SIGHUP handler ──────────────────────────────────────────────
# When SLURM delivers SIGHUP to this batch script (--signal=B:HUP@360),
# forward it to the srun process group so WarpX triggers its break-signal
# handler and writes a checkpoint before the allocation expires.
SRUN_PID=""

hup_handler() {
    echo "[sbatch] $(date): received SIGHUP (near walltime). Forwarding to srun (pid ${SRUN_PID})." >&2
    if [[ -n "${SRUN_PID}" ]] && kill -0 "${SRUN_PID}" 2>/dev/null; then
        kill -HUP -"${SRUN_PID}" 2>/dev/null || kill -HUP "${SRUN_PID}"
    fi
}
trap hup_handler HUP

# ── Launch ────────────────────────────────────────────────────────────────────
# CUDA visible devices are ordered inverse to local task IDs
#   Reference: nvidia-smi topo -m
srun --cpu-bind=cores bash -c "
    export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID));
    ${EXE} ${INPUTS} \
        warpx.break_signals=HUP \
        warpx.checkpoint_signals=USR1" \
  > "${OUTPUT_FILE}" &

SRUN_PID=$!

# Start watchdog in the background, passing the srun PID
watchdog "${SRUN_PID}" &
WATCHDOG_PID=$!

# Wait for srun to finish, then clean up the watchdog
wait "${SRUN_PID}"
SRUN_EXIT=$?

kill "${WATCHDOG_PID}" 2>/dev/null
wait "${WATCHDOG_PID}" 2>/dev/null

echo "[sbatch] srun exited with status ${SRUN_EXIT}." >&2
exit "${SRUN_EXIT}"
