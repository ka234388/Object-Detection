#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
ENV_NAME="object_detection"            # no spaces
PYTHON_VERSION="3.10"
RESULTS_DIR="/content/drive/MyDrive/FALL_2025_CVS"   # PLZ CHANGE THE DIRECOTR WHER EYOU ARE RUNNING THE CODE LOCAL LOCATION I USED COLAB AND KAGGLE LOCATIONS
EVAL_SCRIPT="${RESULTS_DIR}/evaluate.py"
REQ_FILE="${RESULTS_DIR}/requirements.txt"

# Model JSONs
FRCNN_JSON="${RESULTS_DIR}/faster_rcnn_results_20250904_022849.json"
DETR_JSON="${RESULTS_DIR}/detr_results_20250904_032038.json"
GDINO_JSON="${RESULTS_DIR}/grounding_dino_results_20250904_145238.json"

echo ">>> Starting evaluation pipeline"
echo "Using env: ${ENV_NAME}"
echo "Eval script: ${EVAL_SCRIPT}"

# --- CONDA SETUP ---
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install Anaconda/Miniconda first."
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- CREATE / ACTIVATE ENV ---
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' exists."
else
  echo "Creating env '${ENV_NAME}' (python=${PYTHON_VERSION})..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi
conda activate "${ENV_NAME}"

# --- TORCH (GPU) ---
echo "Installing PyTorch CUDA build..."
conda install -y -c pytorch -c nvidia pytorch=2.3 torchvision=0.18 torchaudio=2.3 pytorch-cuda=12.1

# --- PIP REQUIREMENTS ---
if [ ! -f "${REQ_FILE}" ]; then
  echo "ERROR: requirements.txt not found: ${REQ_FILE}"
  exit 1
fi
python -m pip install --upgrade pip
pip install -r "${REQ_FILE}"

# --- GPU INFO ---
python - << 'PY'
import torch, torchvision
print("CUDA available:", torch.cuda.is_available())
print("Torch:", torch.__version__, "| TorchVision:", torchvision.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# --- RUN EVALUATIONS ---
if [ ! -f "${EVAL_SCRIPT}" ]; then
  echo "ERROR: evaluate.py not found at ${EVAL_SCRIPT}"
  exit 1
fi

run_one () {
  local NAME="$1"
  local JSON="$2"
  local OUT_DIR="${RESULTS_DIR}/evaluation_${NAME}"
  local LOG_DIR="${RESULTS_DIR}/logs_${NAME}"

  if [ -f "${JSON}" ]; then
    echo ">>> Evaluating ${NAME} ..."
    python "${EVAL_SCRIPT}" "${JSON}" \
      --eval_dir "${OUT_DIR}" \
      --logs_dir "${LOG_DIR}" \
      --samples 5 \
      --score_thr 0.5
  else
    echo "WARNING: ${NAME} results JSON not found at ${JSON}"
  fi
}

run_one "frcnn" "${FRCNN_JSON}"
run_one "detr"  "${DETR_JSON}"
run_one "gdino" "${GDINO_JSON}"

echo ">>> Done. Check ${RESULTS_DIR} for plots, samples, and reports."
