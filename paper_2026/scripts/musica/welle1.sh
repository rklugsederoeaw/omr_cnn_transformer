#!/bin/bash
# Erste Welle der Messreihen M1 und M2 (docs/arxiv/neufassung/messplan.md). Auf dem Login-Knoten aus $DATA/crnn_choral/logs:
#   bash ../volpiano2omr_gt/natural_music_processing/skripte/musica/welle1.sh
# Reicht ein: Kontrolle (joint.py auf hybrid_hybrid_lang, Sollwert 5,71 %), M1 (3 Modi x 3 Seeds, je mit joint.py),
# sieben Melodiemodelle ohne eine Quelle (Korpus, dann 30 Epochen), M2 (6 Faltungen x CTC und Hybrid).
# Schreibt jede Jobnummer mit Zweck nach welle1_jobs.tsv.
set -eu
S=../volpiano2omr_gt/natural_music_processing/skripte/musica/crnn.sh
L=${L:-welle1_jobs.tsv}
# OHNE: regulärer Ausdruck auf den Zweck; passende Jobs werden nicht eingereicht (Ausgabe dann leer)
OHNE=${OHNE:-^$}
sb() { # sb <zweck> <sbatch-Optionen ...> -- <skript> [args]
    local zweck=$1; shift
    if [[ $zweck =~ $OHNE ]]; then echo ""; return 0; fi
    local opts=() ; while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
    local j; j=$(sbatch --parsable "${opts[@]}" "$S" "$@")
    printf "%s\t%s\n" "$j" "$zweck" >> "$L"
    echo "$j"
}

sb kontrolle_joint_hybrid_hybrid_lang --job-name=kontrolle --time=01:30:00 --export=ALL,MODELL=hybrid_hybrid_lang,LM=decoder_lm -- skripte/joint.py >/dev/null

# M1: feste Aufteilung, Start aus CRNN v3, 25 Epochen, Lernraten wie hybrid_hybrid_lang (Vorgaben von hybrid.py)
for m in ctc scratch hybrid; do
    for s in 0 1 2; do
        t=_m1_s$s
        j=$(sb "M1 $m seed $s" --job-name=m1_${m}_s$s --time=10:00:00 --export=ALL,MODUS=$m,TAG=$t,SEED=$s,INIT=v3,EPOCHEN=25 -- skripte/hybrid.py)
        [ -n "$j" ] || j=$(eval echo \${VORHER_${m}_s$s:-})  # läuft schon: Jobnummer aus VORHER_<modus>_s<seed>
        [ $m = ctc ] || sb "M1 $m seed $s joint" --job-name=m1j_${m}_s$s --time=01:30:00 ${j:+--dependency=afterok:$j} --export=ALL,MODELL=hybrid_$m$t,LM=decoder_lm -- skripte/joint.py >/dev/null
    done
done

# Melodiemodelle ohne eine CantusDB-Quelle (sechs Faltungen und Salzinnes für M3)
declare -A LM
for id in 123613 123606 123628 123667 123612 638308 123723; do
    k=$(sb "Korpus ohne $id" --job-name=korpus_$id --time=01:00:00 --export=ALL,HOLDOUT=$id -- skripte/decoder_korpus.py)
    LM[$id]=$(sb "Melodiemodell ohne $id" --job-name=lm_$id --time=02:00:00 --dependency=afterok:$k --export=ALL,KORPUS=decoder_korpus_ohne$id,NAME=decoder_lm_ohne$id -- skripte/decoder_lm.py 30)
done

# M2: Leave-one-source-out, Encoder zufällig, 40 Epochen; Hybrid startet den Decoder aus dem Melodiemodell ohne die Quelle
for f in a-kn-1012:123613 sbe-0611:123606 f-pn-lat-12044:123628 a-wn-1799:123667 d-ka-aug-lx:123612 ch-p-18:638308; do
    q=${f%%:*}; id=${f##*:}
    o=ALL,DATA_IDX=daten/loso_$q,INIT=none,ENC_LR=1e-3,EPOCHEN=40,SEED=0
    sb "M2 $q ctc" --job-name=m2_ctc_$q --time=16:00:00 --export=$o,MODUS=ctc,TAG=_m2_$q -- skripte/hybrid.py >/dev/null
    sb "M2 $q hybrid" --job-name=m2_hyb_$q --time=16:00:00 --dependency=afterok:${LM[$id]} --export=$o,MODUS=hybrid,TAG=_m2_$q,LM_INIT=decoder_lm_ohne$id -- skripte/hybrid.py >/dev/null
done
wc -l < "$L"
