"""Volpiano -> geometrische Vierliniennotation (und zurück, verlustfrei).

Positionen von unten, Konvention des crnn-austrian-Modells: l1 z1 l2 z2 l3 z3 l4 z4 l5 z5 ...
(z = Zwischenraum, z_n über l_n); darunter ohne Null: z-1 (unter l1), l-1 (1. Hilfslinie), z-2, l-2 ... Schlüssel: c4 = C-Schlüssel auf l4 (markiert C4 = Volpiano k),
f3 = F-Schlüssel auf l3 (markiert F3 = Volpiano f).

Tokens: Schlüssel (für Volpiano 1), Noten `z2` / Liqueszenz `z2_liq`, Vorzeichen `b_l3` / `na_l3`,
Trenner `-` `--` `---` (unverändert), Abschnittsstriche `div3` `div4`, Umbrüche `br_line` `br_page`
`br_col` (7/77/777), Lücke `lac` (6), alles andere `raw_<zeichen>`.
"""

import re

GAMUT = "9abcdefghjklmnopqrs"  # Γ A B c d e f g a b c' ... (Volpiano ohne i)
CLEF_NOTE = {"c": GAMUT.index("k"), "f": GAMUT.index("f")}
ACCIDENTALS = {"i": "j", "y": "b", "z": "q", "w": "e", "x": "m"}  # an Daten geprüft
ACC_OF_NOTE = {v: k for k, v in ACCIDENTALS.items()}
BREAKS = {"7": "br_line", "77": "br_page", "777": "br_col"}
SYMBOLS = {"3": "div3", "4": "div4", "6": "lac"}
WHITESPACE = {
    " ": "sp",
    "\n": "nl",
    "\r": "cr",
    "\t": "tab",
}  # sonst zerbricht die Leerzeichen-Serialisierung
REV_WHITESPACE = {v: k for k, v in WHITESPACE.items()}
TOKEN = re.compile(r"-+|7{1,3}|.", re.S)


def pos_name(n: int) -> str:
    """Zähler 0 = l1, 1 = z1, … 6 = l4; negativ: -1 = z-1, -2 = l-1, -3 = z-2."""
    if n % 2 == 0:
        return f"l{n // 2 + 1}" if n >= 0 else f"l{n // 2}"
    return f"z{(n - 1) // 2 + 1}" if n >= 0 else f"z{(n - 1) // 2}"


def pos_num(name: str) -> int:
    k = int(name[1:])
    if k == 0:
        raise ValueError(f"{name}: keine Nullposition (unter l1 liegt z-1, dann l-1)")
    if name[0] == "l":
        return 2 * (k - 1) if k > 0 else 2 * k
    return 2 * k - 1 if k > 0 else 2 * k + 1


def _clef(clef: str) -> tuple[int, int]:
    m = re.fullmatch(r"([cf])([1-4])", clef)
    if not m:
        raise ValueError(f"Schlüssel {clef!r}: erwartet c1–c4 oder f1–f4")
    return CLEF_NOTE[m.group(1)], 2 * (int(m.group(2)) - 1)


def to_geo(volpiano: str, clef: str) -> list[str]:
    clef_idx, clef_pos = _clef(clef)
    place = lambda letter: pos_name(GAMUT.index(letter) - clef_idx + clef_pos)  # noqa: E731
    out = []
    for t in TOKEN.findall(volpiano):
        low = t.lower()
        if t[0] == "-":
            out.append(t)
        elif t in BREAKS:
            out.append(BREAKS[t])
        elif t == "1":
            out.append(clef)
        elif low in GAMUT:
            out.append(place(low) + ("_liq" if t.isupper() else ""))
        elif low in ACCIDENTALS:
            out.append(("b_" if t.islower() else "na_") + place(ACCIDENTALS[low]))
        elif t in SYMBOLS:
            out.append(SYMBOLS[t])
        else:
            out.append(f"raw_{WHITESPACE.get(t, t)}")
    return out


def to_volpiano(tokens: list[str], clef: str) -> str:
    """Umkehrung. Schlüssel explizit, da Zeilensegmente (nach 7) kein Schlüssel-Token tragen."""
    clef_idx, clef_pos = _clef(clef)

    def letter(p: str) -> str:
        i = pos_num(p) - clef_pos + clef_idx
        if not 0 <= i < len(GAMUT):
            raise ValueError(
                f"{p} liegt mit Schlüssel {clef} außerhalb des Volpiano-Tonraums"
            )
        return GAMUT[i]

    rev_breaks = {v: k for k, v in BREAKS.items()}
    rev_symbols = {v: k for k, v in SYMBOLS.items()}
    out = []
    for t in tokens:
        if t[0] == "-":
            out.append(t)
        elif t in rev_breaks:
            out.append(rev_breaks[t])
        elif t in rev_symbols:
            out.append(rev_symbols[t])
        elif re.fullmatch(r"[cf][1-4]", t):
            out.append("1")
        elif t.startswith(("b_", "na_")):
            kind, _, p = t.partition("_")
            acc = ACC_OF_NOTE[letter(p)]
            out.append(acc if kind == "b" else acc.upper())
        elif t.startswith("raw_"):
            out.append(REV_WHITESPACE.get(t[4:], t[4:]))
        else:
            p, _, liq = t.partition("_")
            out.append(letter(p).upper() if liq else letter(p))
    return "".join(out)


if __name__ == "__main__":
    # Beispiel des Benutzers: Ave Maria (S. Pietro B.79) mit c4
    geo = to_geo("1---d--f---g--g--g---g--ghgf--ed", "c4")
    assert (
        " ".join(geo)
        == "c4 --- l1 -- l2 --- z2 -- z2 -- z2 --- z2 -- z2 l3 z2 l2 -- z1 l1"
    ), geo
    assert [pos_name(n) for n in (-4, -3, -2, -1, 0, 1, 6, 7, 8)] == [
        "l-2",
        "z-2",
        "l-1",
        "z-1",
        "l1",
        "z1",
        "l4",
        "z4",
        "l5",
    ] and all(pos_num(pos_name(n)) == n for n in range(-12, 13))
    assert to_geo("1---f", "f3")[-1] == "l3" and to_geo("1---k", "c2")[-1] == "l2"

    # Rundreise über den ganzen Dump, für jeden Schlüssel
    import sqlite3
    from collections import Counter

    db = sqlite3.connect(
        "/home/rklug/projects_2026/volpiano2omr_gt/cantus_database.sqlite"
    )
    rows = [r[0] for r in db.execute("select volpiano from chants where volpiano!=''")]
    raw = Counter()
    for clef in ("c1", "c2", "c3", "c4", "f2", "f3", "f4"):
        for v in rows:
            g = to_geo(v, clef)
            assert to_volpiano(" ".join(g).split(" "), clef) == v, (clef, v)
            if clef == "c4":
                raw.update(t for t in g if t.startswith("raw_"))
    print(f"Rundreise ok: {len(rows)} Gesänge × 7 Schlüssel")
    print("nicht zugeordnete Zeichen:", raw.most_common())
