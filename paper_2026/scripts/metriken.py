"""Zerlegen des gabc-artigen Ziels, getrennte Messung von Noten und Text, AMNLT-Ausrichtung.

Alle Noten-Atome sind Einzelzeichen, deshalb liest zerlege() das Ziel mit und ohne <m> gleich.
Tonlage: Buchstaben a-m (Großbuchstabe nur andere Notenform), ohne Schlüssel, Custos, Vorzeichen.
"""

import re
from collections import Counter

from rapidfuzz.distance import Levenshtein as lev

GRUPPE = re.compile(r"([^()]*)\(([^()]*)\)")
TON = set("abcdefghijklm")


def zerlege(s):
    """Liste (Text vor der Gruppe, Noten-Atome) und der Text hinter der letzten Klammer."""
    paare = [(m[1], list(m[2].replace("<m>", ""))) for m in GRUPPE.finditer(s)]
    return paare, GRUPPE.sub("", s)


def noten_token(atome):
    """Fasst Schlüssel (c4), Custos (f+), Vorzeichen (gx) und Taktstrich mit Ziffer (;6) zusammen."""
    aus, i = [], 0
    while i < len(atome):
        a, n = atome[i], atome[i + 1] if i + 1 < len(atome) else ""
        if (a in "cf" and n in tuple("12345")) or (a == ";" and n.isdigit()):
            aus.append(a + n)
            i += 2
        elif a.lower() in TON and n in tuple("+xy"):
            aus.append(a + n)
            i += 2
        else:
            aus.append(a)
            i += 1
    return aus


def ist_ton(t):
    return len(t) == 1 and t.lower() in TON


def ist_schluessel(t):
    return len(t) == 2 and t[0] in "cf" and t[1].isdigit()


def sicht(s):
    """Noten-Token, Töne, Schlüssel, Grenzen (Tonzahl am Gruppenende und an !), Silben, Text."""
    paare, rest = zerlege(s)
    token, toene, schl, grenzen, silben = [], [], [], set(), []
    for text, atome in paare:
        if text.strip():
            silben.append(text.strip())
        for t in noten_token(atome):
            token.append(t)
            if t == "!":
                grenzen.add(len(toene))
            elif ist_ton(t):
                toene.append(t.lower())
            elif ist_schluessel(t):
                schl.append(t)
        grenzen.add(len(toene))
    if rest.strip():
        silben.append(rest.strip())
    grenzen.discard(0)
    grenzen.discard(len(toene))
    text = "".join(t for t, _ in paare) + rest
    return {
        "token": token,
        "toene": toene,
        "schl": schl,
        "grenzen": grenzen,
        "silben": silben,
        "zeichen": re.sub(r"\s+", "", text),
        "woerter": text.split(),
        "kompakt": s.replace("<m>", ""),
    }


def schleife(s, min_wdh=6, max_periode=60):
    """Endet s mit einer Zeichenfolge, die sich mindestens min_wdh-mal direkt wiederholt?"""
    s = s.rstrip()  # 116 Referenzen enden auf Leerzeichen
    for p in range(1, max_periode + 1):
        if len(s) >= p * min_wdh and all(
            s[-p:] == s[-p * (k + 1) : -p * k or None] for k in range(min_wdh)
        ):
            return True
    return False


def grenz_treffer(r, h):
    """Treffer, Falsche, Fehlende der Grenzen, über die Ausrichtung der Tonfolgen abgebildet."""
    h_nach_r = {}
    for op in lev.opcodes(h["toene"], r["toene"]):
        if op.tag in ("equal", "replace"):  # falscher Ton: Position bleibt vergleichbar
            for k in range(op.src_end - op.src_start):
                h_nach_r[op.src_start + k + 1] = op.dest_start + k + 1
    abgebildet = {h_nach_r.get(g) for g in h["grenzen"]}
    tp = len(abgebildet & r["grenzen"])
    return tp, len(h["grenzen"]) - tp, len(r["grenzen"]) - tp


def amnlt_token(s):
    """Tokenfolge für AMLER und bWER (Fuentes-Martínez u. a. 2026, Abschnitt 6, Formeln 6 bis 8).

    Das Paper spricht nur von »music and lyrics tokens« und legt die Zerlegung nicht fest. Gewählt:
    jede Silbe (Text außerhalb der Klammern, getrennt an Klammern und Leerraum) ist ein Token,
    '(' und ')' sind eigene Token (»alignment vocabulary«, Abschnitt 4.2), jedes Notenzeichen in
    der Klammer ist ein Token mit Präfix <m> (»music-aware gabc«, Abschnitt 5.2), damit der Ton a
    im Bag-of-Words nicht mit der Silbe a zusammenfällt. Silben statt Buchstaben, weil bWER nach
    Vidal u. a. 2023 auf Wörtern zählt und sonst jeder Buchstabendreher als Ausrichtungsfehler
    gälte. Leerzeichen und Zeilenumbrüche trennen nur: Wortgrenzen und Systemumbrüche der
    Seitenlesungen zählen nicht (das Paper misst Einzelsysteme, Abschnitt 4).
    Eingabe mit und ohne <m> ergibt dieselbe Folge.
    """
    token, silbe, in_noten = [], "", False
    for z in s.replace("<m>", ""):
        if z in "()" or z.isspace():
            if silbe:
                token.append(silbe)
                silbe = ""
            if not z.isspace():
                token.append(z)
                in_noten = z == "("
        elif in_noten:
            token.append("<m>" + z)
        else:
            silbe += z
    return token + [silbe] if silbe else token


def bwer_fehler(ref_token, hyp_token):
    """Editierabstand ohne Reihenfolge (Zähler von Formel 7): (| |G|-|P| | + Summe |f_G - f_P|) / 2."""
    fr, fh = Counter(ref_token), Counter(hyp_token)
    verschieden = sum(abs(fr[w] - fh[w]) for w in fr.keys() | fh.keys())
    return (abs(len(ref_token) - len(hyp_token)) + verschieden) // 2


def einheiten(s):
    """Je Gruppe (Silbe, Tonfolge); Silbe oder Tonfolge können leer sein (Schlüssel, Striche)."""
    return [
        (text.strip(), "".join(x.lower() for x in noten_token(atome) if ist_ton(x)))
        for text, atome in zerlege(s)[0]
    ]


def auslassungen(ref, hyp):
    """Einheiten der Referenz (Silbe mit Tönen), die in der Lesung ganz fehlen, und ihre Töne.

    Eine Einheit gilt als ausgelassen, wenn in der Ausrichtung der Silbenfolgen ihre Silbe und in
    der Ausrichtung der Tonfolgen alle ihre Töne gestrichen sind (Editierabstand, Operation
    Löschen). Eine falsch gelesene oder mit der Nachbarsilbe verschmolzene Silbe zählt nicht,
    solange ihre Töne in der Lesung stehen.
    Rückgabe: (Einheiten mit Silbe und Tönen in der Referenz, davon ausgelassen, verlorene Töne).
    """
    a, b = einheiten(ref), einheiten(hyp)

    def gestrichen(x, y):  # Positionen in x, die beim Umschreiben nach y gelöscht werden
        return {
            i
            for op in lev.opcodes(x, y)
            if op.tag == "delete"
            for i in range(op.src_start, op.src_end)
        }

    silbe_weg = gestrichen([t for t, _ in a if t], [t for t, _ in b if t])
    ton_weg = gestrichen("".join(t for _, t in a), "".join(t for _, t in b))
    n = weg = toene_weg = si = ti = 0
    for text, toene in a:
        if text and toene:
            n += 1
            if si in silbe_weg and all(k in ton_weg for k in range(ti, ti + len(toene))):
                weg += 1
                toene_weg += len(toene)
        si += bool(text)
        ti += len(toene)
    return n, weg, toene_weg


def messe(paare):
    """paare: Liste (Referenz, Lesung, Quelle-mit-Wortgrenzen). Fehlerraten als Anteil.

    amler, bwer: je Beispiel durch die Referenzlänge geteilt, dann gemittelt (Makro). Das Paper
    nennt die Mittelung nicht; das Skript der Autoren (efm18/AMNLT, compute_amnlt_metrics.py)
    mittelt ebenso (Abgleich in docs/Aligned_Music_Notation/README.md). amler_mikro, bwer_mikro:
    Fehlersumme durch Summe der Referenz-Token wie alle übrigen Raten hier. aler jeweils aus den
    beiden Gesamtwerten, (AMLER - bWER) / AMLER, bei AMLER = 0 gleich 0.0 (im Paper 0/0).
    Leere Referenz: fehlt im Makro-Mittel (Division durch 0), ihre Fehler zählen im Mikro-Wert.
    einheit_weg: Anteil der Einheiten (Silbe mit Tönen), die in der Lesung ganz fehlen;
    toene_weg: ihre Töne als Anteil aller Referenztöne (siehe auslassungen()).
    toene_geloescht, toene_eingefuegt, toene_ersetzt: tonlage_ser nach Fehlerart zerlegt
    (fehlender, überzähliger, falscher Ton); die drei Anteile ergeben zusammen tonlage_ser.
    """
    s = dict.fromkeys(
        "ser_f ser_n ton_f ton_n sch_f sch_n sil_f sil_n cer_f cer_n ges_f ges_n "
        "wer_f wer_n tp fp fn start_ok start_n aml_f bw_f aml_n aml_m bw_m aml_anz "
        "ein_n ein_weg ein_ton ton_del ton_ins ton_sub".split(),
        0,
    )
    for ref, hyp, wortgrenzen in paare:
        r, h = sicht(ref), sicht(hyp)
        for f, n, a, b in (
            ("ser_f", "ser_n", r["token"], h["token"]),
            ("ton_f", "ton_n", r["toene"], h["toene"]),
            ("sch_f", "sch_n", r["schl"], h["schl"]),
            ("sil_f", "sil_n", r["silben"], h["silben"]),
            ("cer_f", "cer_n", r["zeichen"], h["zeichen"]),
            ("ges_f", "ges_n", r["kompakt"], h["kompakt"]),
        ):
            s[f] += lev.distance(a, b)
            s[n] += len(a)
        if wortgrenzen:
            s["wer_f"] += lev.distance(r["woerter"], h["woerter"])
            s["wer_n"] += len(r["woerter"])
        tp, fp, fn = grenz_treffer(r, h)
        s["tp"] += tp
        s["fp"] += fp
        s["fn"] += fn
        if r["schl"]:
            s["start_n"] += 1
            s["start_ok"] += h["schl"][:1] == r["schl"][:1]
        for op in lev.editops(r["toene"], h["toene"]):  # Tonlagen-Fehler nach Art
            s[{"delete": "ton_del", "insert": "ton_ins", "replace": "ton_sub"}[op.tag]] += 1
        ein_n, ein_weg, ein_ton = auslassungen(ref, hyp)
        s["ein_n"] += ein_n
        s["ein_weg"] += ein_weg
        s["ein_ton"] += ein_ton
        rt, ht = amnlt_token(ref), amnlt_token(hyp)
        aml, bw = lev.distance(rt, ht), bwer_fehler(rt, ht)
        s["aml_f"] += aml
        s["bw_f"] += bw
        s["aml_n"] += len(rt)
        if rt:
            s["aml_m"] += aml / len(rt)
            s["bw_m"] += bw / len(rt)
            s["aml_anz"] += 1
    anteil = lambda a, b: round(s[a] / s[b], 4) if s[b] else None  # noqa: E731
    nenner = 2 * s["tp"] + s["fp"] + s["fn"]

    def aler(f, b):  # Formel 8; die Zahl der Beispiele kürzt sich im Makro-Fall heraus
        if not s["aml_n"]:
            return None
        return round((s[f] - s[b]) / s[f], 4) if s[f] else 0.0

    return {
        "noten_ser": anteil("ser_f", "ser_n"),
        "tonlage_ser": anteil("ton_f", "ton_n"),
        "schluessel_ser": anteil("sch_f", "sch_n"),
        "anfangsschluessel_richtig": anteil("start_ok", "start_n"),
        "grenzen_f1": round(2 * s["tp"] / nenner, 4) if nenner else None,
        "text_cer": anteil("cer_f", "cer_n"),
        "silben_ser": anteil("sil_f", "sil_n"),
        "text_wer_nur_mit_wortgrenzen": anteil("wer_f", "wer_n"),
        "gesamt_cer": anteil("ges_f", "ges_n"),
        "noten_token": s["ser_n"],
        "toene": s["ton_n"],
        "amler": anteil("aml_m", "aml_anz"),
        "bwer": anteil("bw_m", "aml_anz"),
        "aler": aler("aml_m", "bw_m"),
        "amler_mikro": anteil("aml_f", "aml_n"),
        "bwer_mikro": anteil("bw_f", "aml_n"),
        "aler_mikro": aler("aml_f", "bw_f"),
        "einheiten": s["ein_n"],
        "einheit_weg": anteil("ein_weg", "ein_n"),
        "toene_weg": anteil("ein_ton", "ton_n"),
        "toene_geloescht": anteil("ton_del", "ton_n"),
        "toene_eingefuegt": anteil("ton_ins", "ton_n"),
        "toene_ersetzt": anteil("ton_sub", "ton_n"),
    }


if __name__ == "__main__":
    z = "(<m>c<m>4) de(<m>g<m>h) us(<m>g<m>V<m>f) (<m>:) re(<m>e<m>x<m>f<m>+)"
    v = sicht(z)
    assert v["token"] == ["c4", "g", "h", "g", "V", "f", ":", "ex", "f+"], v["token"]
    assert v["toene"] == ["g", "h", "g", "f"], v["toene"]
    assert v["schl"] == ["c4"] and v["silben"] == ["de", "us", "re"]
    assert v["zeichen"] == "deusre" and v["kompakt"].startswith("(c4) de(gh)")
    assert sicht(z.replace("<m>", ""))["token"] == v["token"]  # mit und ohne <m> gleich
    assert sicht("(<m>;<m>6) a(<m>G)")["token"] == [";6", "G"]
    assert sicht("(<m>;<m>6) a(<m>G)")["toene"] == ["g"]
    # ausgelassene Einheiten: Silbe und Töne fehlen gemeinsam
    q = "(<m>c<m>4) a(<m>g) b(<m>h<m>i) c(<m>j)"
    assert auslassungen(q, q) == (3, 0, 0)
    assert auslassungen(q, "(c4) a(g) c(j)") == (3, 1, 2)  # Einheit b fehlt ganz
    assert auslassungen(q, "(c4) a(g) x(hi) c(j)") == (3, 0, 0)  # Silbe falsch, Töne da
    assert auslassungen(q, "(c4) a(g) (hi) c(j)") == (3, 0, 0)  # Silbe leer, Töne da
    assert auslassungen(q, "(c4) a(ghi) c(j)") == (
        3,
        0,
        0,
    )  # Töne hängen am Nachbarn davor
    assert auslassungen(q, "(c4) a(g) c(hij)") == (
        3,
        0,
        0,
    )  # Töne hängen am Nachbarn danach
    assert auslassungen(q, "") == (3, 3, 4)
    ma = messe([(q, "(c4) a(g) c(j)", False)])
    assert ma["einheiten"] == 3 and ma["einheit_weg"] == 0.3333 and ma["toene_weg"] == 0.5, ma
    assert ma["toene_geloescht"] == 0.5 and ma["toene_eingefuegt"] == ma["toene_ersetzt"] == 0.0
    mb = messe([(q, "(c4) a(g) b(hk) c(j) d(j)", False)])  # ein falscher, ein überzähliger Ton
    assert (mb["toene_geloescht"], mb["toene_eingefuegt"], mb["toene_ersetzt"]) == (0.0, 0.25, 0.25)
    assert mb["tonlage_ser"] == 0.5 and mb["einheit_weg"] == 0.0, mb
    assert messe([(z, z, True)])["noten_ser"] == 0.0
    m = messe([(z, z.replace("<m>g<m>h", "<m>g<m>i"), False)])
    assert m["tonlage_ser"] == 0.25 and m["text_wer_nur_mit_wortgrenzen"] is None, m
    assert m["grenzen_f1"] == 1.0, m  # ein falscher Ton verschiebt keine Grenze
    assert m["anfangsschluessel_richtig"] == 1.0
    assert schleife("a(f) " * 7) and not schleife("a(f) b(g) " * 2)
    assert not schleife("a(f)" + " " * 30)

    # AMNLT: AMLER, bWER, AlER
    def amnlt(*p):
        m = messe([(r, h, False) for r, h in p])
        return {k: w for k, w in m.items() if "ler" in k or "bwer" in k}

    zt = amnlt_token(z)
    assert amnlt_token("(c4) a(g)") == ["(", "<m>c", "<m>4", ")", "a", "(", "<m>g", ")"]
    assert zt == amnlt_token(z.replace("<m>", "")) and len(zt) == 25  # mit und ohne <m>
    # Leerraum und Zeilenumbruch trennen nur
    assert amnlt_token("a(g)\nb(h) c") == amnlt_token("a(g) b(h)c")
    assert amnlt_token("ab)cd(e") == ["ab", ")", "cd", "(", "<m>e"]  # schiefe Klammern
    assert bwer_fehler(list("abc"), list("ba")) == 1
    assert bwer_fehler(list("ab"), list("cde")) == 3
    assert bwer_fehler(amnlt_token("a(e)"), amnlt_token("e(a)")) == 2  # Ton a, Silbe a
    gleich = amnlt((z, z))
    assert set(gleich.values()) == {0.0}, gleich
    # reiner Inhaltsfehler: ein falscher Ton von 25 Token
    inhalt = amnlt((z, z.replace("<m>g<m>h", "<m>g<m>i")))
    assert inhalt["amler"] == inhalt["bwer"] == 0.04 and inhalt["aler"] == 0.0, inhalt
    assert inhalt == amnlt((z.replace("<m>", ""), z.replace("<m>g<m>h", "gi")))
    # reiner Ausrichtungsfehler: Ton h an der falschen Silbe, 2 Schritte auf 9 Token
    schub = amnlt(("a(gh) b(i)", "a(g) b(hi)"))
    assert schub["amler"] == 0.2222 and schub["bwer"] == 0.0, schub
    assert schub["aler"] == 1.0, schub
    assert amnlt(("a(g) b(h)", "b(g) a(h)"))["aler"] == 1.0  # Silben vertauscht
    leer = amnlt((z, ""))
    assert leer["amler"] == leer["bwer"] == leer["amler_mikro"] == 1.0, leer
    assert leer["aler"] == leer["aler_mikro"] == 0.0, leer
    beide = amnlt(("a(gh) b(i)", "a(g) b(hi)"), (z, z))  # Makro (2/9 + 0)/2, Mikro 2/34
    assert beide["amler"] == 0.1111 and beide["amler_mikro"] == 0.0588, beide
    assert beide["aler"] == beide["aler_mikro"] == 1.0
    ohne_ref = amnlt(("", "a(g)"))  # kein Makro-Wert, Mikro ohne Nenner
    assert set(ohne_ref.values()) == {None}, ohne_ref
    # leere Referenz zählt nur im Mikro-Wert: 4 + 1 Fehler auf 4 Token
    gemischt = amnlt(("", "a(g)"), ("a(g)", "a(h)"))
    assert gemischt["amler"] == 0.25 and gemischt["amler_mikro"] == 1.25, gemischt
    for name, w in (("gleich", gleich), ("inhalt", inhalt), ("schub", schub)):
        print(name, w)
    print("leer", leer)
    print("makro/mikro", beide)
    print("ok")
