"""Volpiano-Zeile ↔ CRNN-Lesung eines Notensystems: Schlüsselfolge bestimmen und Systeme den Zeilen zuordnen.

clef_dp: Editierdistanz zwischen Hypothese (Schlüssel + Noten, Positionen aus Volpiano unter wechselnden
Schlüsseln) und CRNN-Tokens (nur Noten und Schlüssel). Schlüsselwechsel nur zwischen Neumen
(Volpiano-Trenner), neuer Gesang (`1`) wechselt billig. Der Schlüssel am Zeilenanfang wird immer erzeugt,
auch wenn die Volpiano-Zeile mitten im Gesang beginnt. Exakte Rückverfolgung des Schlüsselwegs.

match_page: monotone Zuordnung Volpiano-Zeilen ↔ Systeme mit Auslassen, Zusammenlegen (zwei Zeilen auf
einem System = überzähliges 7) und Teilen (eine Zeile über zwei Systeme = fehlendes 7).
"""

import re

import numpy as np

from volpiano_geo import CLEF_NOTE, GAMUT, pos_num, to_geo

CLEFS = ["c1", "c2", "c3", "c4", "f2", "f3", "f4"]
POS_TOKEN = re.compile(r"^[lz]-?\d+$")
CLEF_TOKEN = re.compile(r"^[cf][1-5]$")

DEL_NOTE = INS_NOTE = SUB_NOTE = 1.0
DEL_CLEF = INS_CLEF = 0.5  # CRNN übersieht Schlüssel in der Zeile oft
SUB_CLEF = 1.0
MISMATCH = 3.0  # Note gegen Schlüssel
SWITCH = {"-": 2.0, "--": 1.5, "---": 1.5, "1": 0.3}  # Penalty je Wechselort
LEDGER = 0.1  # je Position außerhalb z-1..LEDGER_MAX (Schreiber mieden Hilfslinien)
# Oberste Position im System: z4 = 7 bei vier Linien, z5 = 9 bei fünf (2*n_lines-1).
# run_source.py setzt den Wert aus sources/<slug>.json -> staves.n_lines.
LEDGER_MAX = 7


def crnn_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if POS_TOKEN.match(t) or CLEF_TOKEN.match(t)]


def volpiano_notes(v: str) -> list[tuple[int, str, int]]:
    """[(Gamut-Index, Wechselort davor, Zeichenoffset)]; Wechselort '' = innerhalb einer Neume (kein Wechsel)."""
    out, sep = [], "start"
    for m in re.finditer(r"-+|1|.", v):
        t = m.group()
        if t[0] == "-":
            sep = t if len(t) <= 3 else "---"
        elif t == "1":
            sep = "1"
        elif t.lower() in GAMUT:
            out.append((GAMUT.index(t.lower()), sep, m.start()))
            sep = ""
    return out


def clef_offset(clef: str) -> int:
    return 2 * (int(clef[1]) - 1) - CLEF_NOTE[clef[0]]


def _sub_ins(hyp, t: str) -> tuple[float, float]:
    if CLEF_TOKEN.match(t):
        return (
            (0.0 if t == hyp else SUB_CLEF) if isinstance(hyp, str) else MISMATCH
        ), INS_CLEF
    return (
        MISMATCH if isinstance(hyp, str) else (0.0 if pos_num(t) == hyp else SUB_NOTE)
    ), INS_NOTE


def _emit(prev: np.ndarray, hyp, crnn: list[str], cost_del: float) -> np.ndarray:
    """Eine Hypotheseneinheit (Note = int-Position, Schlüssel = str) an die CRNN-Folge anhängen."""
    out = np.empty(len(crnn) + 1)
    out[0] = prev[0] + cost_del
    for j, t in enumerate(crnn, 1):
        sub, ins = _sub_ins(hyp, t)
        out[j] = min(prev[j] + cost_del, prev[j - 1] + sub, out[j - 1] + ins)
    return out


def _emit_back(
    prev: np.ndarray, out: np.ndarray, hyp, crnn: list[str], cost_del: float, j: int
) -> int:
    """Rückwärts durch _emit: Spalte in prev, aus der out[j] entstand."""
    while True:
        if j > 0:
            sub, ins = _sub_ins(hyp, crnn[j - 1])
            if np.isclose(out[j], prev[j - 1] + sub):
                return j - 1
        if np.isclose(out[j], prev[j] + cost_del):
            return j
        j -= 1  # Einfügung eines CRNN-Tokens innerhalb derselben Einheit


def clef_dp(v: str, crnn: list[str], forced: frozenset[int] = frozenset()) -> dict:
    """forced: Notenindizes, vor denen ein neues System beginnt – dort steht immer ein Schlüssel (frei wählbar,
    auch derselbe wie zuvor), ohne Wechselstrafe."""
    notes = volpiano_notes(v)
    crnn = crnn_tokens(crnn)
    m = len(crnn)
    if not notes:
        return {"cost": float(m), "ned": 1.0, "clefs": [], "n": 0}
    base0 = np.arange(m + 1, dtype=float) * INS_NOTE
    rows = {c: _emit(base0, c, crnn, DEL_CLEF) for c in CLEFS}
    trace = []  # je Note k: {c: (start, switched|None, others|None, others_arg|None, note_row)}
    for k, (g, sep, _) in enumerate(notes):
        new, step = {}, {}
        can_switch = k > 0 and sep != ""
        stack = np.array([rows[c] for c in CLEFS])
        for ci, c in enumerate(CLEFS):
            pos = g + clef_offset(c)
            start, switched, others, arg = rows[c], None, None, None
            if k in forced:
                arg = stack.argmin(axis=0)
                others = stack.min(axis=0)
                switched = _emit(others, c, crnn, DEL_CLEF)
                start = switched
            elif can_switch:
                masked = stack.copy()
                masked[ci] = np.inf
                arg = masked.argmin(axis=0)
                others = masked.min(axis=0) + SWITCH[sep]
                switched = _emit(others, c, crnn, DEL_CLEF)
                start = np.minimum(rows[c], switched)
            note_row = _emit(start, pos, crnn, DEL_NOTE)
            new[c] = note_row + LEDGER * max(0, -1 - pos, pos - LEDGER_MAX)
            step[c] = (start, switched, others, arg, note_row, rows[c])
        trace.append(step)
        rows = new

    c = min(CLEFS, key=lambda x: rows[x][-1])
    cost = float(rows[c][-1])
    j, switches = m, []
    for k in range(len(notes) - 1, -1, -1):
        g = notes[k][0]
        start, switched, others, arg, note_row, stay = trace[k][c]
        j = _emit_back(start, note_row, g + clef_offset(c), crnn, DEL_NOTE, j)
        if k in forced:
            j = _emit_back(others, switched, c, crnn, DEL_CLEF, j)
            switches.append((k, c))
            c = CLEFS[int(arg[j])]
        elif (
            switched is not None
            and switched[j] < stay[j]
            and np.isclose(start[j], switched[j])
        ):
            j = _emit_back(others, switched, c, crnn, DEL_CLEF, j)
            switches.append((k, c))
            c = CLEFS[int(arg[j])]
    clefs = [(0, c)] + switches[::-1]
    n_hyp = len(notes) + len(clefs)
    return {
        "cost": cost,
        "ned": cost / max(n_hyp, m, 1),
        "clefs": clefs,
        "n": len(notes),
    }


def geo_with_clefs(v: str, clefs: list[tuple[int, str]]) -> list[str]:
    """Volpiano -> Tokens mit Schlüsselwechseln vor den angegebenen Noten; Schlüssel-Token am Zeilenanfang."""
    notes = volpiano_notes(v)
    cuts = [notes[k][2] for k, _ in clefs[1:]] + [len(v)]
    out, start = [], 0
    for (_, clef), end in zip(clefs, cuts):
        out.append(clef)
        out += [
            t for t in to_geo(v[start:end], clef) if t != clef
        ]  # '1' ergäbe doppelten Schlüssel
        start = end
    return out


FLAT_TOKEN = re.compile(r"^(b|na)_[lz]-?\d+$")


def _clef_form(t: str) -> bool:
    return bool(re.fullmatch(r"([a-g][1-5])(/[a-g][1-5])*", t))


def export_tokens(geo: list[str]) -> list[str] | None:
    """geo-Tokens (geo_with_clefs) -> Exportformat wie rein.gt_tokens: Trenner ',', Abschnittsstrich 'L', Positionen
    ohne _liq, Schlüssel (auch geschriebene Formen f2/c4) unverändert; None bei lac/raw_* oder ohne Noten.
    Die Ernte schrieb bis 2026-10-11 rohe geo-Tokens – damit lernte das CRNN zwei Beschriftungssysteme."""
    out: list[str] = []
    for t in geo:
        if t.startswith("-"):
            if out and out[-1] != "," and not _clef_form(out[-1]):
                out.append(",")
        elif t in ("div3", "div4"):
            out.append("L")
        elif POS_TOKEN.match(t.removesuffix("_liq")) or FLAT_TOKEN.match(t):
            out.append(t.removesuffix("_liq"))
        elif _clef_form(t):
            out.append(t)
        else:
            return None
    while out and out[-1] == ",":
        out.pop()
    return out if any(POS_TOKEN.match(t) for t in out) else None


def norm_crnn(t: str) -> str:
    """CRNN-Varianten des b (bz3, b-z3, b_z3) vereinheitlichen."""
    m = re.fullmatch(r"b[_-]?([lz]-?\d+)", t)
    return f"b_{m.group(1)}" if m else t


def _is_unit(t: str) -> bool:
    return bool(
        POS_TOKEN.match(t.removesuffix("_liq"))
        or CLEF_TOKEN.match(t)
        or FLAT_TOKEN.match(t)
    )


def geo_positions(geo: list[str], seen: list[tuple[str, int, float]]) -> list:
    """x-Position je Hypothesen-Token (Noten, Schlüssel, Versetzungszeichen) über das Editier-Alignment mit der
    CRNN-Lesung: zugeordnete Einheiten übernehmen die Bildposition [System-Nr., Anteil 0..1], übrige interpoliert."""
    a = [i for i, t in enumerate(geo) if _is_unit(t)]
    b = [(norm_crnn(t), k, x) for t, k, x in seen if _is_unit(norm_crnn(t))]
    n, m = len(a), len(b)
    same = lambda i, j: geo[a[i]].removesuffix("_liq") == b[j][0]  # noqa: E731
    D = np.zeros((n + 1, m + 1))
    D[:, 0], D[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(
                D[i - 1, j] + 1,
                D[i, j - 1] + 1,
                D[i - 1, j - 1] + (0 if same(i - 1, j - 1) else 1),
            )
    g: dict[int, float] = {}
    i, j = n, m
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (0 if same(i - 1, j - 1) else 1):
            g[i - 1] = b[j - 1][1] + b[j - 1][2]
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    known = sorted(g)
    n_staves = max(1, len({k for _, k, _ in seen}))
    out: list = [None] * len(geo)
    for u in range(n):
        if u in g:
            val = g[u]
        elif known:
            lo = max((k for k in known if k < u), default=None)
            hi = min((k for k in known if k > u), default=None)
            if lo is not None and hi is not None:
                val = g[lo] + (g[hi] - g[lo]) * (u - lo) / (hi - lo)
            else:
                ref = lo if lo is not None else hi
                val = g[ref] + 0.01 * (u - ref)
        else:
            val = n_staves * u / max(n, 1)
        k = min(max(int(val), 0), n_staves - 1)
        out[a[u]] = [k, round(min(max(val - k, 0.0), 1.0), 4)]
    return out


def staff_boundary(
    v: str, staves: list[list[str]], clefs: list[tuple[int, str]]
) -> int:
    """Index der ersten Note, die im zweiten System steht (an einer Neumengrenze)."""
    notes = volpiano_notes(v)
    geo = geo_with_clefs(v, clefs)
    seen = [(t, k, 0.5) for k, s in enumerate(staves) for t in s]
    pos = geo_positions(geo, seen)
    note_places = [
        p for t, p in zip(geo, pos) if POS_TOKEN.match(t.removesuffix("_liq"))
    ]
    b = next((u for u, p in enumerate(note_places) if p and p[0] >= 1), None)
    if b is None:
        n1 = sum(1 for t in staves[0] if POS_TOKEN.match(t))
        n2 = sum(1 for t in staves[1] if POS_TOKEN.match(t))
        b = round(len(notes) * n1 / max(1, n1 + n2))
    b = min(max(b, 1), len(notes) - 1)
    while b < len(notes) - 1 and notes[b][1] == "":  # nicht mitten in eine Neume
        b += 1
    return b


def staff_boundaries(
    v: str, staves: list[list[str]], clefs: list[tuple[int, str]]
) -> list[int]:
    """Notenindizes, an denen System 2, 3, … beginnen (n Systeme -> n−1 Grenzen, streng steigend, an Neumengrenzen);
    Verallgemeinerung von staff_boundary auf eine ganze Seite (Klosterneuburg: `7` trägt die Zeilen nicht)."""
    notes = volpiano_notes(v)
    geo = geo_with_clefs(v, clefs)
    seen = [(t, k, 0.5) for k, s in enumerate(staves) for t in s]
    pos = geo_positions(geo, seen)
    places = [p for t, p in zip(geo, pos) if POS_TOKEN.match(t.removesuffix("_liq"))]
    counts = [sum(1 for t in s if POS_TOKEN.match(t)) for s in staves]
    out: list[int] = []
    prev = 0
    for k in range(1, len(staves)):
        b = next((u for u, p in enumerate(places) if p and p[0] >= k), None)
        if b is None:
            b = round(len(notes) * sum(counts[:k]) / max(1, sum(counts)))
        b = min(max(b, prev + 1), len(notes) - 1)
        while b < len(notes) - 1 and notes[b][1] == "":
            b += 1
        out.append(b)
        prev = b
    return out


def page_split(v: str, staves: list[list[str]]) -> list[dict]:
    """Volpiano einer Seite auf n gelesene Systeme verteilen: Grenzen aus dem Seiten-Alignment, dann jedes System
    für sich abgeglichen (entry_result) -> je System {v, ned, clefs, geo}."""
    toks = [t for s in staves for t in s]
    r = clef_dp(v, toks)
    if not r["clefs"] or len(staves) < 2:
        return [{"v": v} | entry_fields(v, r)]
    b = staff_boundaries(v, staves, r["clefs"])
    notes = volpiano_notes(v)
    cuts = [0] + [notes[i][2] for i in b] + [len(v)]
    out = []
    for k, s in enumerate(staves):
        vk = v[cuts[k] : cuts[k + 1]].strip("-")
        rk = entry_result(vk, [s]) if vk else {"ned": 1.0, "clefs": [], "n": 0}
        out.append({"v": vk} | entry_fields(vk, rk))
    return out


def entry_result(v: str, staff_tokens: list[list[str]]) -> dict:
    """Schlüsselfolge für Volpiano-Text v auf einem oder zwei Systemen (bei zweien Schlüssel am Beginn des zweiten)."""
    toks = [t for s in staff_tokens for t in s]
    r = clef_dp(v, toks)
    if len(staff_tokens) == 2 and r["clefs"]:
        b = staff_boundary(v, staff_tokens, r["clefs"])
        r = clef_dp(v, toks, frozenset({b})) | {"boundary": b}
    return r


def entry_fields(v: str, r: dict) -> dict:
    """ned, Schlüssel, Hypothese und – bei zwei Systemen – Ground Truth je Systembild."""
    out = {
        "ned": r["ned"],
        "clefs": r["clefs"],
        "geo": geo_with_clefs(v, r["clefs"]) if r["clefs"] else [],
    }
    if "boundary" in r:
        b, cut = r["boundary"], volpiano_notes(v)[r["boundary"]][2]
        out["boundary"] = b
        out["geo_per_staff"] = [
            geo_with_clefs(v[:cut], [(k, c) for k, c in r["clefs"] if k < b]),
            geo_with_clefs(v[cut:], [(k - b, c) for k, c in r["clefs"] if k >= b]),
        ]
    return out


def match_page(
    line_texts: list[str],
    staff_tokens: list[list[str]],
    skip: float = 0.8,
    window: int = 4,
) -> list[dict]:
    """Monotone Zuordnung; Operationen: match, skip_staff, skip_line, merge (2 Zeilen→1 System), split (1 Zeile→2 Systeme).
    Kosten: NED × Anzahl beteiligter Einheiten; merge/split mit Zuschlag, damit match bei Gleichstand gewinnt."""
    n, m = len(line_texts), len(staff_tokens)
    cache: dict = {}

    def result(i0, i1, j0, j1):
        key = (i0, i1, j0, j1)
        if key not in cache:
            cache[key] = entry_result(
                "---".join(line_texts[i0:i1]), staff_tokens[j0:j1]
            )
        return cache[key]

    D = np.full((n + 1, m + 1), np.inf)
    B: dict = {}
    D[0, 0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            if not np.isfinite(D[i, j]):
                continue
            steps = [(i, j + 1, "skip_staff"), (i + 1, j, "skip_line")]
            if abs(i - j) <= window:
                steps += [
                    (i + 1, j + 1, "match"),
                    (i + 2, j + 1, "merge"),
                    (i + 1, j + 2, "split"),
                ]
            for ii, jj, op in steps:
                if ii > n or jj > m:
                    continue
                if op.startswith("skip"):
                    c = skip
                else:
                    units = (ii - i) + (jj - j)
                    c = result(i, ii, j, jj)["ned"] * units / 2 + (
                        0.3 if op != "match" else 0.0
                    )
                if D[i, j] + c < D[ii, jj]:
                    D[ii, jj] = D[i, j] + c
                    B[(ii, jj)] = (i, j, op)
    out, cur = [], (n, m)
    while cur != (0, 0):
        i, j, op = B[cur]
        entry = {
            "op": op,
            "lines": list(range(i, cur[0])),
            "staves": list(range(j, cur[1])),
        }
        if not op.startswith("skip"):
            r = result(i, cur[0], j, cur[1])
            v = "---".join(line_texts[i : cur[0]])
            entry |= entry_fields(v, r)
        out.append(entry)
        cur = (i, j)
    return out[::-1]


if __name__ == "__main__":
    import random

    rnd = random.Random(1)

    def melody(n: int) -> str:
        """Zufällige Volpiano-Zeile im Bereich d–l mit Neumen- und Silbentrennern."""
        out = []
        for _ in range(n):
            neume = "".join(
                rnd.choice("defghjkl") for _ in range(rnd.choice([1, 1, 2, 3]))
            )
            out.append(neume + rnd.choice(["-", "--", "---"]))
        return "".join(out).rstrip("-")

    def reading(
        v: str, clefs: list[tuple[int, str]], drop_mid_clefs: bool = True
    ) -> list[str]:
        toks = crnn_tokens(geo_with_clefs(v, clefs))
        return [
            t
            for i, t in enumerate(toks)
            if not (drop_mid_clefs and i > 0 and CLEF_TOKEN.match(t))
        ]

    # 1. Schlüsselwechsel c2 -> c3 vor den letzten 5 Noten, CRNN liest den Wechsel nicht
    v = melody(18)
    k = len(volpiano_notes(v)) - 5
    while volpiano_notes(v)[k][1] == "":  # Wechsel nur zwischen Neumen
        k -= 1
    truth = [(0, "c2"), (k, "c3")]
    assert clef_dp(v, reading(v, truth))["clefs"] == truth
    # 2. ohne Wechsel bleibt ein Schlüssel; mit 2 Lesefehlern ebenso
    v2 = melody(15)
    r2 = reading(v2, [(0, "c4")])
    r2[3], r2[8] = "l5", "z-2"
    assert clef_dp(v2, r2)["clefs"] == [(0, "c4")]
    # 3. Seite: Fremdsystem (Textzeile mit Notenrest) wird ausgelassen, Zeile über zwei Systeme wird geteilt
    lines = [melody(14), melody(14), melody(24), melody(14)]
    staves = [
        reading(lines[0], [(0, "c4")]),
        reading(melody(14), [(0, "c3")]),
        reading(lines[1], [(0, "c3")]),
    ]
    long = reading(lines[2], [(0, "c3")])
    staves += [
        long[: len(long) // 2],
        ["c3"] + long[len(long) // 2 :],
        reading(lines[3], [(0, "f3")]),
    ]
    ops = [e["op"] for e in match_page(lines, staves)]
    assert ops == ["match", "skip_staff", "match", "split", "match"], ops
    for c2 in ("c4", "c3"):
        v4 = melody(26)
        notes4 = volpiano_notes(v4)
        b = next(k for k in range(len(notes4) // 2, len(notes4)) if notes4[k][1])
        cut = notes4[b][2]
        s1 = reading(v4[:cut], [(0, "c4")])
        s2 = reading(v4[cut:], [(0, c2)])
        res = match_page([v4], [s1, s2])
        assert [e["op"] for e in res] == ["split"], res
        assert res[0]["clefs"] == [(0, "c4"), (b, c2)], (c2, res[0]["clefs"], b)
        assert [g[0] for g in res[0]["geo_per_staff"]] == ["c4", c2]
    # 5. Seite mit vier Systemen ohne Zeilenmarken: Grenzen gefunden, jedes System liest sauber
    page = [melody(16) for _ in range(4)]
    v5 = "---".join(page)
    st5 = [reading(z, [(0, c)]) for z, c in zip(page, ("c4", "c3", "f3", "c4"))]
    parts = page_split(v5, st5)
    assert [q["v"] for q in parts] == page, [q["v"] for q in parts]
    # tonhöhengleiche Schlüsselform (c3/f1) kostet ein Token
    assert all(q["ned"] < 0.05 for q in parts), [q["ned"] for q in parts]
    assert export_tokens(
        ["f2/c4", "l2", "---", "z3_liq", "--", "div4", "-", "c3", "l3", "---"]
    ) == ["f2/c4", "l2", ",", "z3", ",", "L", ",", "c3", "l3"]
    assert (
        export_tokens(["c4", "l2", "lac"]) is None
        and export_tokens(["c4", "---"]) is None
    )
    print("align ok")
