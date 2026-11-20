"""Export the cleaned AMNLT Solesmes crops to the project's positional OMR format.

Read the original s-gabc labels and check them against Kurek's paired common
encoding. This script does not train or invoke a model. Run with Python -I.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re

from datasets import Image, load_from_disk
from PIL import Image as PILImage


HERE = Path(__file__).resolve().parent.parent
ROOT = Path("/mnt/d/omr_sources/AMNLT")
SOURCE = ROOT / "deduplicated_original" / "Solesmes_staffLevel"
PAIRED = ROOT / "common_encoding_deduplicated" / "Solesmes_staffLevel"
OUTPUT = HERE / "daten" / "amnlt_solesmes"
VOCABULARY = HERE.parent / "trainingsdaten" / "vokabular.json"
SPLITS = ("train", "validation", "test")
PITCHES = "abcdefghijklm"
GROUP = re.compile(r"\(([^()]*)\)", re.S)
POSITION = re.compile(r"[lz]-?[0-9]+\Z")
CLEF = re.compile(r"[cf][1-5]\Z")
# The local s-gabc grammar identifies these as graphic or rhythmic information.
FORM = set("vVswoO~><'_.º°ª!/¡@")
BARS = set("`,;:^?`")
ACCIDENTALS = set("xy#")


def pos_name(number):
    if number % 2 == 0:
        return f"l{number // 2 + 1}" if number >= 0 else f"l{number // 2}"
    return f"z{(number - 1) // 2 + 1}" if number >= 0 else f"z{(number - 1) // 2}"


def pos_num(token):
    index = int(token[1:])
    if token.startswith("l"):
        return 2 * (index - 1) if index > 0 else 2 * index
    return 2 * index - 1 if index > 0 else 2 * index + 1


def pitch_position(letter):
    """GABC/s-gabc a = z-2, d = l1, h = l3, m = z5."""
    return PITCHES.index(letter.lower()) - 3


def pitch_from_position(token, clef):
    """Use the active clef and staff line to reconstruct the s-gabc letter."""
    line = int(clef[1])
    clef_letter_index = 2 * line + 1  # c3/f3 -> h, c2/f2 -> f
    clef_staff_position = 2 * (line - 1)
    index = clef_letter_index + pos_num(token) - clef_staff_position
    if not 0 <= index < len(PITCHES):
        raise ValueError(f"Out-of-range pitch: {token}, {clef}")
    return PITCHES[index]


def source_groups(transcription):
    groups = list(GROUP.finditer(transcription))
    if transcription.count("(") != len(groups) or transcription.count(")") != len(
        groups
    ):
        raise ValueError("unbalancierte_klammern")
    return [match.group(1) for match in groups]


def read_group(body, reasons, flags, vorzeichen=False):
    """Read tagged s-gabc glyphs conservatively, retaining only pitch events.

    vorzeichen=True: `x`/`y` nach einem Buchstaben sind ein ♭ bzw. eine Auflösung auf dieser Lage (kein Ton) und
    werden als Ereignis ("acc", Präfix, Buchstabe) behalten statt die Zeile auszuschließen."""
    if not body:
        return []
    parts = body.split("<m>")
    if parts[0].strip():
        reasons.add("unmarkierte_musik")
    chars = parts[1:]
    if any(len(char) != 1 for char in chars):
        reasons.add("unbekanntes_zeichen:mehrfachzeichen")
    events = []
    i = 0
    while i < len(chars):
        char = chars[i]
        following = chars[i + 1] if i + 1 < len(chars) else ""
        if len(char) != 1:
            reasons.add("unbekanntes_zeichen:mehrfachzeichen")
            i += 1
            continue
        if char == "C" and chars[i : i + 4] == ["C", "O", "N", "T"]:
            i += 4  # annotation for continuation, not four pitches
            continue
        if char in "cf" and following and following in "12345":
            events.append(("clef", char + following))
            i += 2
            if i < len(chars) and chars[i] == "b":
                # Some transcriptions put the flat after the clef number.
                flags["hat_vorzeichen"] = True
                reasons.add("vorzeichen")
                i += 1
            continue
        if char.lower() in PITCHES:
            events.append(("pitch", char.lower()))
        elif char == "+":
            # A pitched custos is encoded as a note followed by +. Drop both.
            for j in range(len(events) - 1, -1, -1):
                if events[j][0] == "pitch":
                    events.pop(j)
                    break
                if events[j][0] in {"bar", "clef"}:
                    reasons.add("custos_ohne_ton")
                    break
            else:
                reasons.add("custos_ohne_ton")
        elif char in "zZ":
            flags["custos"] += 1
        elif char in ACCIDENTALS:
            flags["hat_vorzeichen"] = True
            if vorzeichen and char in "xy" and events and events[-1][0] == "pitch":
                events[-1] = ("acc", ("b_" if char == "x" else "na_", events[-1][1]))
            else:
                reasons.add("vorzeichen")
        elif char in BARS:
            events.append(("bar", char))
        elif char == "r":
            reasons.add("unsichere_lesung")
            if following and following in "1234567":
                i += 1
        elif char in FORM:
            pass
        elif char in {'"', "“", "”", "»", "*", "|", " "}:
            reasons.add("unbekanntes_zeichen:" + repr(char))
        elif char in "01234567":
            previous = chars[i - 1] if i else ""
            if previous not in (BARS | {"/", "_", "z", "Z", "r"}):
                reasons.add("unbekanntes_zeichen:zahl")
        else:
            reasons.add("unbekanntes_zeichen:" + repr(char))
        i += 1
    return events


def reverse_pitches(tokens):
    clef = None
    pitches = []
    for token in tokens:
        if CLEF.fullmatch(token):
            clef = token
        elif POSITION.fullmatch(token):
            if clef is None:
                raise ValueError("Ton vor Schlüssel")
            pitches.append(pitch_from_position(token, clef))
        elif token not in {",", "L"} and not token.startswith(("b_", "na_")):
            raise ValueError(f"Unknown output token: {token}")
    return pitches


def convert(transcription, vocabulary, vorzeichen=False):
    reasons = set()
    flags = {"hat_vorzeichen": False, "custos": 0}
    try:
        groups = source_groups(transcription)
    except ValueError as exc:
        groups = []
        reasons.add(str(exc))
    event_groups = [read_group(body, reasons, flags, vorzeichen) for body in groups]
    tokens = []
    pitches = []
    initial_clef = ""
    prev_notes = False
    for events in event_groups:
        group_notes = []

        def flush():
            nonlocal prev_notes
            if group_notes:
                if prev_notes:
                    tokens.append(",")
                tokens.extend(group_notes)
                group_notes.clear()
                prev_notes = True

        for kind, value in events:
            if kind == "clef":
                flush()
                if not initial_clef:
                    initial_clef = value
                if value not in vocabulary:
                    reasons.add("schluessel_nicht_im_vokabular")
                tokens.append(value)
                prev_notes = False
            elif kind == "bar":
                flush()
                if tokens and tokens[-1] != "L":
                    tokens.append("L")
                prev_notes = False
            elif kind == "acc":
                name = value[0] + pos_name(pitch_position(value[1]))
                if name not in vocabulary:
                    reasons.add("vorzeichen_nicht_im_vokabular")
                group_notes.append(name)
            else:
                if not initial_clef:
                    reasons.add("ungueltiger_anfangsschluessel")
                position = pitch_position(value)
                if not -2 <= position <= 10:
                    reasons.add("ton_ausserhalb_des_umfangs")
                else:
                    name = pos_name(position)
                    if name not in vocabulary:
                        reasons.add("position_nicht_im_vokabular")
                    group_notes.append(name)
                    pitches.append(value)
        flush()
    if not initial_clef or not tokens or tokens[0] != initial_clef:
        reasons.add("ungueltiger_anfangsschluessel")
    if not pitches:
        reasons.add("keine_noten")
    reconstructed = []
    if not reasons:
        try:
            reconstructed = reverse_pitches(tokens)
        except ValueError:
            reasons.add("rueckrechnung_abweichung")
        if reconstructed != pitches:
            reasons.add("rueckrechnung_abweichung")
    return {
        "schluessel": initial_clef,
        "tokens": tokens,
        "pitches": pitches,
        "hat_vorzeichen": flags["hat_vorzeichen"],
        "reasons": reasons,
        "roundtrip_ok": not reasons and reconstructed == pitches,
    }


def probe(dataset, vocabulary, count=50):
    rows = dataset["test"]
    outcome = Counter()
    for index in range(min(count, len(rows))):
        result = convert(rows[index]["transcription"], vocabulary)
        outcome["total"] += 1
        outcome["ok" if result["roundtrip_ok"] else "excluded"] += 1
        for reason in result["reasons"]:
            outcome[reason.split(":", 1)[0]] += 1
    print(json.dumps(dict(outcome), ensure_ascii=False), flush=True)
    if outcome["rueckrechnung_abweichung"]:
        raise SystemExit("Roundtrip mismatch in 50-row probe")
    return outcome


def write_report(
    stats,
    probe_stats,
    examples,
    clefs,
    page_splits,
    image_hashes,
    pixel_hashes,
    text_hashes,
    names,
    revision,
):
    n_pages = len(page_splits)
    n_overlapping_pages = sum(len(splits) > 1 for splits in page_splits.values())
    image_dups = sum(
        len({row[0] for row in entries}) > 1 for entries in image_hashes.values()
    )
    pixel_dups = sum(
        len({row[0] for row in entries}) > 1 for entries in pixel_hashes.values()
    )
    text_dups = sum(
        len({row[0] for row in entries}) > 1 for entries in text_hashes.values()
    )
    lines = [
        "# Solesmes (AMNLT): OMR-Export",
        "",
        "Quelle: `PRAIG/Solesmes_staffLevel`, Revision `"
        + revision
        + "`; lokal bereinigte Fassung "
        "`deduplicated_original`. Bilder und Index bleiben wegen CC BY-NC 4.0 lokal. "
        "Die Splits sind die ursprünglichen AMNLT-Splits; der Projektleiter entscheidet über Trainingszeilen.",
        "",
        "## Kodierung",
        "",
        "Die Quelle nutzt **s-gabc**, nicht Volpiano. Jedes Musikzeichen trägt `<m>`. "
        "Die s-gabc-Grammatik des lokal vorhandenen GabcParser definiert `a` bis `m` "
        "(auch Großbuchstaben für rhombische Formen) als Notenpositionen. "
        "Die Tonleiter ist an die vier Linien gebunden: `a=z-2`, `d=l1`, `f=l2`, `h=l3`, "
        "`j=l4`, `m=z5`. Der Schlüssel `cN` oder `fN` wird als Token ausgegeben, "
        "und Schlüsselwechsel bleiben erhalten. Die Formel stimmt mit der "
        "s-gabc-zu-common-Umrechnung des lokalen GabcParser überein: "
        "Schlüsselton auf GABC-Position `1+2*N`, unterste Linie auf Position `d`. "
        "`volpiano_geo.py` benutzt eine andere Tonbuchstabenreihe; dessen "
        "Positionskonvention und Rückrechnung wurden als Vergleich herangezogen.",
        "",
        "`x` = ♭, `y` = Auflösung, `#` = Kreuz; auch das nachgestellte `b` "
        "bei `c4b` ist ein Vorzeichenzusatz am Schlüssel. Diese Zeilen erhalten "
        "`hat_vorzeichen=true`, `ok=false` und ein leeres Label. "
        "`v/V` (Fähnchen), `º/°/ª` (Porrectus), `>`, `<`, `~` (Liqueszenz), "
        "`s/w/o` (Form), `!`, `/`, `@` (Abstand/Ansatz), Punkt und Episemen "
        "sind Form- oder Rhythmusangaben und entfallen. `+` nach einem Ton "
        "und `z/Z` sind Custodes und entfallen. `,`, `;`, `:`, `^` und `` ` `` "
        "sind Takt-/Abschnittsstriche und werden als `L` ausgegeben. "
        "Zwischen Silbengruppen steht `,`. `r` ist laut Grammatik eine unsichere "
        "Tonlesung; solche Zeilen werden ausgeschlossen. `CONT` ist eine "
        "Fortsetzungsmarke. Unbekannte Zeichen (u. a. Anführungszeichen und `|`) "
        "sind als Ausschlussgrund gekennzeichnet.",
        "",
        "## Umfang",
        "",
        "| AMNLT-Split | Gesamt | OK | Ausgeschlossen | Mit Vorzeichen |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = stats[split]
        lines.append(
            f"| {split} | {item['total']} | {item['ok']} | {item['total'] - item['ok']} | {item['accidental']} |"
        )
    lines.extend(["", "Ausschlussgründe (Mehrfachnennungen möglich):", ""])
    for split in SPLITS:
        item = stats[split]
        reasons = (
            ", ".join(f"{k}: {v}" for k, v in sorted(item["reasons"].items()))
            or "keine"
        )
        lines.append(f"- `{split}`: {reasons}")
    lines.extend(
        [
            "",
            "Schlüsselverteilung der gültigen Zeilen (Anfangsschlüssel): "
            + ", ".join(f"`{k}` {v}" for k, v in sorted(clefs.items()))
            + ".",
            "",
            "## Rückrechnung und Splits",
            "",
            f"Erste {probe_stats['total']} Testzeilen: {probe_stats['ok']} ohne Tonabweichung "
            f"rückgerechnet, {probe_stats['excluded']} ausgeschlossen. "
            "Bei allen als OK markierten Zeilen ergab die Rückrechnung aus Schlüssel "
            "und Positionen genau die Tonbuchstaben der Quelle; null Abweichungen. "
            "Bildpixel wurden ohne Skalierung als PNG exportiert.",
            "",
            f"Die Bildpfade enthalten {n_pages} verschiedene Seitenkennungen "
            f"(`fN` und numerische Kennungen; {names['f']} bzw. {names['numeric']} Ausschnitte). "
            f"{n_overlapping_pages} Kennungen kommen in mehr als einem Split vor. "
            f"Exakt gleiche Bilddateien zwischen Splits: {image_dups}; "
            f"exakt gleiche decodierte Bildpixel: {pixel_dups}; "
            f"exakt gleiche s-gabc-Transkriptionen: {text_dups}. "
            "Die AMNLT-Splits trennen also keine Seiten. Die Kennungen belegen mehrere "
            "Seiten, aber der Datensatz nennt keine Hand-IDs; eine Zahl verschiedener "
            "Schreiber lässt sich daraus nicht belastbar ableiten. Für eine spätere "
            "unabhängige Auswertung muss nach Seitenkennung neu geteilt werden.",
            "",
            "## Fünf Beispiele aus `test`",
            "",
        ]
    )
    for example in examples[:5]:
        lines.extend(
            [
                f"### `{example['id']}`",
                "",
                f"Bild: `{example['bild']}`; Quellbild: `{example['quellbild']}`",
                "",
                f"Schlüssel: `{example['schluessel']}`",
                "",
                f"s-gabc: `{example['quelle_original']}`",
                "",
                f"Tokens: `{example['label']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Restaufwand",
            "",
            "Für eine Trainingsnutzung: Seitenbasierte Neuaufteilung, Sichtprüfung "
            "ausgewählter Schlüssel/Noten und Einbindung in den Trainingsindex. "
            "Geschätzt 3–5 Stunden. Ausgeschlossene Zeilen mit Vorzeichen oder "
            "unsicherer Lesung brauchen eigene Regeln und manuelle Prüfung.",
            "",
        ]
    )
    (OUTPUT / "bericht.md").write_text("\n".join(lines), encoding="utf-8")


def export(original, paired, vocabulary, probe_stats):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image_dir = OUTPUT / "bilder"
    image_dir.mkdir(exist_ok=True)
    stats = defaultdict(
        lambda: {"total": 0, "ok": 0, "accidental": 0, "reasons": Counter()}
    )
    clefs = Counter()
    page_splits = defaultdict(set)
    image_hashes = defaultdict(list)
    pixel_hashes = defaultdict(list)
    text_hashes = defaultdict(list)
    names = Counter()
    examples = []
    temp = OUTPUT / "index.jsonl.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        for split in SPLITS:
            if len(original[split]) != len(paired[split]):
                raise ValueError(f"Unpaired split: {split}")
            for index in range(len(original[split])):
                row = original[split][index]
                pair = paired[split][index]
                if row["transcription"] != pair["transcription_original"]:
                    raise ValueError(f"Source text mismatch: {split}_{index}")
                image = row["image"]
                if (
                    image["path"] != pair["image"]["path"]
                    or image["bytes"] != pair["image"]["bytes"]
                ):
                    raise ValueError(f"Source image mismatch: {split}_{index}")
                original_index = int(pair["original_index"])
                identifier = f"{split}_{original_index}"
                result = convert(row["transcription"], vocabulary)
                accepted = not result["reasons"]
                if accepted and not result["roundtrip_ok"]:
                    raise ValueError(f"Roundtrip failed: {identifier}")
                image_path = image_dir / f"{identifier}.png"
                with PILImage.open(BytesIO(image["bytes"])) as picture:
                    pixel_digest = hashlib.sha256(
                        str((picture.mode, picture.size)).encode() + picture.tobytes()
                    ).digest()
                    picture.save(image_path, format="PNG", compress_level=1)
                record = {
                    "id": identifier,
                    "split": split,
                    "original_index": original_index,
                    "bild": str(image_path.relative_to(OUTPUT)),
                    "quellbild": image["path"],
                    "schluessel": result["schluessel"],
                    "label": " ".join(result["tokens"]) if accepted else "",
                    "quelle_original": row["transcription"],
                    "ok": accepted,
                    "grund": ";".join(sorted(result["reasons"])),
                    "hat_vorzeichen": result["hat_vorzeichen"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                item = stats[split]
                item["total"] += 1
                item["ok"] += int(accepted)
                item["accidental"] += int(result["hat_vorzeichen"])
                for reason in result["reasons"]:
                    item["reasons"][reason.split(":", 1)[0]] += 1
                if accepted:
                    clefs[result["schluessel"]] += 1
                    if split == "test":
                        examples.append(record)
                page = re.sub(r"_group\d+\.[^.]+$", "", image["path"])
                page_splits[page].add(split)
                names["f" if re.fullmatch(r"f\d+", page) else "numeric"] += 1
                image_hashes[hashlib.sha256(image["bytes"]).digest()].append(
                    (split, identifier)
                )
                pixel_hashes[pixel_digest].append((split, identifier))
                text_hashes[
                    hashlib.sha256(row["transcription"].encode()).digest()
                ].append((split, identifier))
            print(
                split,
                {
                    k: (dict(v) if isinstance(v, Counter) else v)
                    for k, v in stats[split].items()
                },
                flush=True,
            )
    temp.replace(OUTPUT / "index.jsonl")
    examples.sort(key=lambda row: len(row["quelle_original"]))
    revision = json.loads((ROOT / "source_revisions.json").read_text(encoding="utf-8"))[
        "datasets"
    ]["Solesmes_staffLevel"]["source_revision"]
    write_report(
        stats,
        probe_stats,
        examples,
        clefs,
        page_splits,
        image_hashes,
        pixel_hashes,
        text_hashes,
        names,
        revision,
    )
    print("index", OUTPUT / "index.jsonl", flush=True)
    print("report", OUTPUT / "bericht.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", action="store_true", help="Only inspect the first 50 test rows"
    )
    args = parser.parse_args()
    vocabulary = set(json.loads(VOCABULARY.read_text(encoding="utf-8")))
    # GABC's d sits on the bottom line; these check the mapping against
    # the explicit staff-position convention in scripts/volpiano_geo.py.
    assert pitch_position("d") == 0 and pitch_position("h") == 4
    assert pitch_from_position("l3", "c3") == "h"
    assert pitch_from_position("l2", "f2") == "f"
    original = load_from_disk(str(SOURCE)).cast_column("image", Image(decode=False))
    probe_stats = probe(original, vocabulary)
    if not args.probe:
        paired = load_from_disk(str(PAIRED)).cast_column("image", Image(decode=False))
        export(original, paired, vocabulary, probe_stats)


if __name__ == "__main__":
    main()
