#!/bin/bash
# M2 Seeds 1 und 2 für die drei Faltungen mit Effekt des Vorwissens (Gutachten M2): Hybrid wie welle1 (Encoder
# zufällig, 40 Epochen, Decoder aus dem Melodiemodell ohne die Quelle), danach Lesekette in Stufen.
#   cd $DATA/crnn_choral/logs && bash ../volpiano2omr_gt/natural_music_processing/skripte/musica/welle4.sh
set -eu
S=../volpiano2omr_gt/natural_music_processing/skripte/musica/crnn.sh
L=welle4_jobs.tsv
for f in a-wn-1799:123667 a-kn-1012:123613 f-pn-lat-12044:123628; do
    q=${f%%:*}; id=${f##*:}
    for s in 1 2; do
        j=$(sbatch --parsable --job-name=m2_hyb_${q}_s$s --time=16:00:00 --export=ALL,DATA_IDX=daten/loso_$q,INIT=none,ENC_LR=1e-3,EPOCHEN=40,SEED=$s,MODUS=hybrid,TAG=_m2_${q}_s$s,LM_INIT=decoder_lm_ohne$id $S skripte/hybrid.py)
        l=$(sbatch --parsable --job-name=l2_${q}_s$s --time=06:00:00 --dependency=afterok:$j $S skripte/lesen_reihe.py --index daten/loso_$q/music/index.jsonl --split test --modell hybrid_hybrid_m2_${q}_s$s --lm decoder_lm_ohne$id --name m2_${q}_s$s)
        printf "%s\tM2 %s hybrid seed %s\n%s\tM2 %s seed %s Lesekette\n" $j $q $s $l $q $s >> $L
    done
done
cat $L
