#!/bin/bash
# Einmalig: venv mit Systemsicht im Container pytorch:25.04 (Fallen 1, 2, 4 aus dem musica-Skill), dann Importprobe.
#SBATCH --job-name=crnn_venv
#SBATCH --account=p201395
#SBATCH --gres=gpu:1
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=dev_zen4_0768_h100x4
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
set -u
export BASE="$DATA/crnn_choral"
apptainer exec --nv --bind /data --bind /scratch "$SCRATCH/containers/pytorch-25.04.sif" bash -c '
    set -e
    export PYTHONNOUSERSITE=1
    grep -v "^dill==" /etc/pip/constraint.txt > /tmp/constraint.txt
    export PIP_CONSTRAINT=/tmp/constraint.txt
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    python -m venv --system-site-packages "$BASE/venv"
    source "$BASE/venv/bin/activate"
    pip install --quiet lightning loguru opencv-python-headless rapidfuzz "transformers==4.46.3" torchinfo onnx wandb rich pyyaml scikit-learn python-dotenv
    python -I -c "import torch, cv2, lightning, loguru, rapidfuzz, scipy, numpy, transformers; import sys; sys.path.insert(0, \"$BASE/omr_cli/src\"); import core.models.crnn, core.data.dataset, core.data.datamodule; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0), \"| cv2\", cv2.__version__, \"| lightning\", lightning.__version__, \"| numpy\", numpy.__version__)"
    pip freeze > "$BASE/venv_freeze.txt"'
RC=$?; echo "=== Exit-Code: $RC | Ende $(date)"; exit $RC
