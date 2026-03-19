#!/bin/bash
#SBATCH --job-name=robust_external
#SBATCH --time=144:00:00
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=a_css
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#SBATCH -o /home/s4636418/results/slurm-%j.out
#SBATCH -e /home/s4636418/results/slurm-%j.err

set -euo pipefail

echo "========================================================================"
echo "Robust External Benchmark - Job ${SLURM_JOB_ID}"
echo "Node: $(hostname), Started: $(date)"
echo "========================================================================"

JOB_USER="${SLURM_JOB_USER:-${USER:-$(id -un)}}"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_DIR="${PROJECT_DIR}/results/${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

module load anaconda3/2023.09-0 || true

export TMPDIR="/scratch/user/${JOB_USER}/tmp"
export PIP_CACHE_DIR="/scratch/user/${JOB_USER}/.cache/pip"
export PYTHONNOUSERSITE=1

mkdir -p "${TMPDIR}"
mkdir -p "${PIP_CACHE_DIR}"

PYTHON_BIN="/scratch/user/${JOB_USER}/conda-envs/fedlearn-cu118/bin/python"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Missing python at ${PYTHON_BIN}"
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import torch, sys
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), "cuda_ver:", torch.version.cuda)
if not torch.cuda.is_available():
    print("ERROR: GPU not visible. Check partition/gres.")
    sys.exit(2)
print("device_count:", torch.cuda.device_count())
print("device_name:", torch.cuda.get_device_name(0))
PY

nvidia-smi || true

# Priority: CLI arg > environment variable > default.
# Example: sbatch run_robust_external_benchmark.sbatch 200,1000
NUM_CLIENTS_LIST="${1:-${NUM_CLIENTS_LIST:-20,200,1000}}"
SAMPLES_PER_CLIENT="${SAMPLES_PER_CLIENT:-250}"
TRAIN_SIZE="${TRAIN_SIZE:-20000}"
TEST_SIZE="${TEST_SIZE:-3000}"
ROUNDS="${ROUNDS:-30}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
LOCAL_LR="${LOCAL_LR:-0.001}"
SEEDS="${SEEDS:-3}"
FAIRNESS_MODES="${FAIRNESS_MODES:-equal_rounds,equal_updates,equal_accepted_updates}"
TOPK_FRACTIONS="${TOPK_FRACTIONS:-0.01,0.03,0.05}"
DOMINANT_RATIO="${DOMINANT_RATIO:-0.7}"
GAMMA_MIN="${GAMMA_MIN:-0.75}"
GAMMA_MAX="${GAMMA_MAX:-1.30}"
BRIGHTNESS_MIN="${BRIGHTNESS_MIN:-0.70}"
BRIGHTNESS_MAX="${BRIGHTNESS_MAX:-1.30}"
TRACK_INTERVAL_UPDATES="${TRACK_INTERVAL_UPDATES:-20}"
ACC_THRESHOLDS="${ACC_THRESHOLDS:-0.20,0.25}"
COMM_BUDGET_MB="${COMM_BUDGET_MB:-300.0}"
USE_DEVICE="${USE_DEVICE:-auto}"

cd "${PROJECT_DIR}"
: > "${RUN_DIR}/robust_external.log"

echo "Effective NUM_CLIENTS_LIST: ${NUM_CLIENTS_LIST}"
echo "Effective FAIRNESS_MODES: ${FAIRNESS_MODES}"
echo "Effective SEEDS: ${SEEDS}, ROUNDS: ${ROUNDS}"
echo "Effective DEVICE: ${USE_DEVICE}"

srun -u "${PYTHON_BIN}" -u robust_external_benchmark.py \
  --num_clients_list "${NUM_CLIENTS_LIST}" \
  --samples_per_client "${SAMPLES_PER_CLIENT}" \
  --train_size "${TRAIN_SIZE}" \
  --test_size "${TEST_SIZE}" \
  --rounds "${ROUNDS}" \
  --local_epochs "${LOCAL_EPOCHS}" \
  --local_lr "${LOCAL_LR}" \
  --seeds "${SEEDS}" \
  --fairness_modes "${FAIRNESS_MODES}" \
  --topk_fractions "${TOPK_FRACTIONS}" \
  --dominant_ratio "${DOMINANT_RATIO}" \
  --gamma_min "${GAMMA_MIN}" \
  --gamma_max "${GAMMA_MAX}" \
  --brightness_min "${BRIGHTNESS_MIN}" \
  --brightness_max "${BRIGHTNESS_MAX}" \
  --track_interval_updates "${TRACK_INTERVAL_UPDATES}" \
  --acc_thresholds "${ACC_THRESHOLDS}" \
  --comm_budget_mb "${COMM_BUDGET_MB}" \
  --device "${USE_DEVICE}" \
  --checkpoint_every_mode \
  2>&1 | tee -a "${RUN_DIR}/robust_external.log"

EXIT_CODE=${PIPESTATUS[0]}

cp -f "${PROJECT_DIR}"/results/robust_external_*.csv "${RUN_DIR}" 2>/dev/null || true
cp -f "${PROJECT_DIR}"/results/robust_external_*.json "${RUN_DIR}" 2>/dev/null || true

echo "Results saved to ${RUN_DIR}"
exit "${EXIT_CODE}"