#!/bin/bash
# Zweite Welle (docs/arxiv/neufassung/messplan.md). Auf dem Login-Knoten aus $DATA/crnn_choral/logs:
#   bash ../volpiano2omr_gt/natural_music_processing/skripte/musica/welle2.sh
# Lesekette (lesen_reihe.py) für M1 (feste Testmenge, deckt die bekannte M4-Menge ab), M2 (sechs Faltungen) und die
# ungesehene M4-Menge; M3: Basis ohne Salzinnes-Melodien, sechs AMNLT-Trainings, deren Lesungen; LightOn liest
# den GABC-Testteil (bekannte M4-Menge). Jobnummern nach welle2_jobs.tsv.
set -eu
S=../volpiano2omr_gt/natural_music_processing/skripte/musica/crnn.sh
L=${L:-welle2_jobs.tsv}
OHNE=${OHNE:-^$}
sb() { # sb <zweck> <sbatch-Optionen ...> -- <skript> [args]
    local zweck=$1; shift
    if [[ $zweck =~ $OHNE ]]; then echo ""; return 0; fi
    local opts=(); while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    local j; j=$(sbatch --parsable "${opts[@]}" "$S" "$@")
    printf "%s\t%s\n" "$j" "$zweck" >> "$L"
    echo "$j"
}
R=skripte/lesen_reihe.py

# M1: Lesekette auf der festen Testmenge (Konkordanzen je Zeile über sources/<slug>.json)
for m in scratch hybrid; do for s in 0 1 2; do
    sb "M1 $m s$s Lesekette" --job-name=l1_${m}_s$s --time=04:00:00 -- $R --index ../trainingsdaten/omr_gt/music/index.jsonl \
        --split test --modell hybrid_${m}_m1_s$s --lm decoder_lm --name m1_${m}_s$s >/dev/null
done; done

# M4 ungesehen: 200 Prüfsysteme, keine Konkordanzen (keine CantusDB-Folios)
for s in 0 1 2; do
    sb "M4 ungesehen hybrid s$s" --job-name=l4_u_s$s --time=01:00:00 -- $R --index daten/vergleich_m4/ungesehen/music/index.jsonl \
        --split test --source-id -1 --modell hybrid_hybrid_m1_s$s --lm decoder_lm --name m4_ungesehen_s$s >/dev/null
done

# M2: Lesekette je Faltung, Melodiemodell ohne die Quelle, Konkordanzen nur aus fremden Quellen
for f in a-kn-1012:123613 sbe-0611:123606 f-pn-lat-12044:123628 a-wn-1799:123667 d-ka-aug-lx:123612 ch-p-18:638308; do
    q=${f%%:*}; id=${f##*:}
    sb "M2 $q Lesekette" --job-name=l2_$q --time=06:00:00 -- $R --index daten/loso_$q/music/index.jsonl --split test \
        --modell hybrid_hybrid_m2_$q --lm decoder_lm_ohne$id --name m2_$q >/dev/null
done

# M3: Basis für Salzinnes ohne dessen Melodien (Decoder aus decoder_lm_ohne123723), wie M1 hybrid Seed 0
b=$(sb "M3 Basis ohne 123723" --job-name=m3_basis --time=10:00:00 --export=ALL,MODUS=hybrid,TAG=_m1_ohne123723,SEED=0,INIT=v3,EPOCHEN=25,LM_INIT=decoder_lm_ohne123723 -- skripte/hybrid.py)
for x in Einsiedeln:hybrid_hybrid_m2_sbe-0611:decoder_lm_ohne123606 Salzinnes:hybrid_hybrid_m1_ohne123723:decoder_lm_ohne123723 Solesmes:hybrid_hybrid_m1_s0:decoder_lm; do
    IFS=: read -r ds basis lm <<< "$x"
    d=daten/amnlt_crnn/$ds
    # nur AMNLT: Encoder und Decoder zufällig, 100 Epochen
    j=$(sb "M3 $ds nur AMNLT" --job-name=m3_nur_$ds --time=06:00:00 --export=ALL,DATA_IDX=$d,MODUS=scratch,TAG=_m3_nur_$ds,SEED=0,INIT=none,ENC_LR=1e-3,EPOCHEN=100 -- skripte/hybrid.py)
    sb "M3 $ds nur AMNLT Lesung" --job-name=l3_nur_$ds --time=02:00:00 --dependency=afterok:$j -- $R --index $d/music/index.jsonl \
        --split test --source-id -1 --modell hybrid_scratch_m3_nur_$ds --lm $lm --name m3_nur_$ds >/dev/null
    # vortrainiert: Feintuning der Basis, 30 Epochen, Lernrate 1e-4 wie die Gesamtmodelle
    dep=""; [ "$basis" = hybrid_hybrid_m1_ohne123723 ] && dep="--dependency=afterok:$b"
    j=$(sb "M3 $ds vortrainiert" --job-name=m3_vor_$ds --time=04:00:00 $dep --export=ALL,DATA_IDX=$d,MODUS=hybrid,TAG=_m3_vor_$ds,SEED=0,INIT=$basis,ENC_LR=1e-4,LR_PRE=1e-4,LR_REST=1e-4,EPOCHEN=30,LM_INIT=$lm -- skripte/hybrid.py)
    sb "M3 $ds vortrainiert Lesung" --job-name=l3_vor_$ds --time=02:00:00 --dependency=afterok:$j -- $R --index $d/music/index.jsonl \
        --split test --source-id -1 --modell hybrid_hybrid_m3_vor_$ds --lm $lm --name m3_vor_$ds >/dev/null
done
wc -l < "$L"
