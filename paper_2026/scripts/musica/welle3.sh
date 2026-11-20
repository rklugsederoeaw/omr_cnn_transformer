#!/bin/bash
# M4 fair: CRNN-Gesamtmodell wie general_v7 (alte GT + Transkribus, viele gotische Chorbücher), aber ohne die fünf
# Prüfhandschriften, so wie LightOn seiten_d. Start hybrid_hybrid_lang, 10 Epochen, Lernrate 1e-4 (Rezept von general_v7),
# drei Seeds; Lesekette auf den 200 Prüfsystemen (ohne Konkordanzen) und auf der festen Testmenge.
#   cd $DATA/crnn_choral/logs && bash ../volpiano2omr_gt/natural_music_processing/skripte/musica/welle3.sh
set -eu
S=../volpiano2omr_gt/natural_music_processing/skripte/musica/crnn.sh
L=welle3_jobs.tsv
for s in 0 1 2; do
    t=_m4g_s$s
    j=$(sbatch --parsable --job-name=m4g_s$s --time=08:00:00 --export=ALL,DATA_IDX=daten/general_ohne_pruef,MODUS=hybrid,TAG=$t,SEED=$s,INIT=hybrid_hybrid_lang,ENC_LR=1e-4,LR_PRE=1e-4,LR_REST=1e-4,EPOCHEN=10 $S skripte/hybrid.py)
    u=$(sbatch --parsable --job-name=l4g_u_s$s --time=01:00:00 --dependency=afterok:$j $S skripte/lesen_reihe.py --index daten/vergleich_m4/ungesehen/music/index.jsonl --split test --source-id -1 --modell hybrid_hybrid$t --lm decoder_lm --name m4g_ungesehen_s$s)
    b=$(sbatch --parsable --job-name=l4g_b_s$s --time=04:00:00 --dependency=afterok:$j $S skripte/lesen_reihe.py --index ../trainingsdaten/omr_gt/music/index.jsonl --split test --modell hybrid_hybrid$t --lm decoder_lm --name m4g_fest_s$s)
    printf "%s\tM4g Training s%s\n%s\tM4g ungesehen s%s\n%s\tM4g feste Testmenge s%s\n" $j $s $u $s $b $s >> $L
done
cat $L
