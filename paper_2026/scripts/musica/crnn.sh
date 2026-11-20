#!/bin/bash
# Allgemeiner GPU-Job für die CRNN-Messreihen (docs/arxiv/neufassung/messplan.md).
#   sbatch [--time ...] [--export=ALL,MODUS=...,SEED=...] crnn.sh <skript.py> [Argumente]
# Läuft in natural_music_processing unter $DATA/crnn_choral (Abbild von ~/projects_2026), venv aus einrichten.sh.
# Ergebnisse schreiben die Skripte direkt nach $DATA (modelle/, docs/, daten/); kein Scratch, keine Rettung nötig.
#SBATCH --job-name=crnn
#SBATCH --account=p201395
#SBATCH --gres=gpu:1
#SBATCH --partition=zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
set -u
export BASE="$DATA/crnn_choral"
export NMP="$BASE/volpiano2omr_gt/natural_music_processing"
export OMR_CLI="$BASE/omr_cli" TORCH_PY="$BASE/venv/bin/python"
export WORKERS="${WORKERS:-18}" TORCH_THREADS="${TORCH_THREADS:-4}" OMP_NUM_THREADS=4
SIF="$SCRATCH/containers/pytorch-25.04.sif"
echo "=== Job $SLURM_JOB_ID $(hostname) $(date) | $* | WORKERS=$WORKERS MODUS=${MODUS:-} TAG=${TAG:-} SEED=${SEED:-} INIT=${INIT:-} DATA=${DATA_IDX:-} LM_INIT=${LM_INIT:-}"
# hybrid.py und joint.py lesen den Index aus $DATA (sonst stünde dort der Cluster-Pfad $DATA); Vorgabe: feste Aufteilung
export DATA="$NMP/${DATA_IDX:-../trainingsdaten/omr_gt}"
apptainer exec --nv --bind /data --bind /scratch "$SIF" bash -c '
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TRITON_LIBCUDA_PATH=/.singularity.d/libs
    export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
    cd "$NMP" && "$TORCH_PY" -I "$@"' bash "$@"
RC=$?
echo "=== Exit-Code: $RC | Ende $(date)"
exit $RC
