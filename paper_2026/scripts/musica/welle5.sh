#!/bin/bash
# Wirkung von --schluessel-fest (lesen_rag.py): Auswahl neu auf den zwischengespeicherten Kandidaten von welle2
# (kandidaten_<name>_test.jsonl wird nach <name>_fest kopiert, dann keine Beam-Suche). Feste Testmenge (M1 Seeds 0-2)
# und sechs Faltungen (M2). Ausgabe daten/lesen_rag/<name>_fest/lesungen_{voll,nur_lm,nur_konk}.jsonl.
#   cd $DATA/crnn_choral/logs && bash ../volpiano2omr_gt/natural_music_processing/skripte/musica/welle5.sh
set -eu
N=../volpiano2omr_gt/natural_music_processing
S=$N/skripte/musica/crnn.sh
K=$N/daten/konkordanz
L=welle5_jobs.tsv
lauf() { # lauf <name> <index> <modell> <lm> [weitere Argumente]
    local name=$1 index=$2 modell=$3 lm=$4; shift 4
    cp "$K/kandidaten_${name}_test.jsonl" "$K/kandidaten_${name}_fest_test.jsonl"
    local j; j=$(sbatch --parsable --job-name=fest_$name --time=01:00:00 $S skripte/lesen_reihe.py --index $index --split test \
        --modell $modell --lm $lm --name ${name}_fest --schluessel-fest "$@")
    printf "%s\t%s fest\n" $j $name >> $L
}
for s in 0 1 2; do lauf m1_hybrid_s$s ../trainingsdaten/omr_gt/music/index.jsonl hybrid_hybrid_m1_s$s decoder_lm; done
for f in a-kn-1012:123613 sbe-0611:123606 f-pn-lat-12044:123628 a-wn-1799:123667 d-ka-aug-lx:123612 ch-p-18:638308; do
    q=${f%%:*}; id=${f##*:}
    lauf m2_$q daten/loso_$q/music/index.jsonl hybrid_hybrid_m2_$q decoder_lm_ohne$id
done
cat $L
