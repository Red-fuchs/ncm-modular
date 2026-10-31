"""Surrogate-masking stress test on NCM codecs (codec only). usage: python src/ncm_modular/mask_stress.py {xsum|wmt14|tinystories}
xsum/wmt14: codec_mlx + HF tokenizer; tinystories: official ncm_mlx codec. CPU only. usage: mask_stress.py <xsum|wmt14|tinystories> [N]"""
import sys, os, re, json, random, difflib
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
import mlx.core as mx
from masking import *
mx.set_default_device(mx.cpu)
NAME = sys.argv[1]; NMAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 150

# ---------------- codec adapters: rt(texts)->decoded texts, ntok(text)
if NAME == "tinystories":
    from ncm_mlx.official import spec as specs
    from ncm_mlx.official.bpe import BPETokenizer
    from ncm_mlx.official.model import Codec
    D = WORK+"/ts"; task = specs.from_json(json.load(open(f"{D}/spec.json"))["task"])
    codec = Codec(task.codec); codec.load_weights(f"{D}/codec.safetensors", strict=True); codec.eval()
    bpe = BPETokenizer.load(f"{D}/bpe.json"); L, EOS, PAD = task.codec.seq_len, task.eos_id, task.pad_id
    enc = bpe.encode; dec = lambda ids: bpe.decode(ids, skip_special=True)
    logits = lambda ids: codec.decode_logits(codec.cumulative(codec.encode(ids))[-1])
else:
    import codec_mlx
    from transformers import AutoTokenizer
    codec = codec_mlx.load(f"{WORK}/mlx/{NAME}")
    tok = AutoTokenizer.from_pretrained(f"{WORK}/tok/{'gpt2' if NAME == 'xsum' else 'Qwen3-0.6B'}")
    sp = codec.cfg; L, EOS, PAD = sp["seq_len"], tok.convert_tokens_to_ids(sp["eos_token"]), sp["prefix_pad_token_id"]
    enc = tok.encode; dec = lambda ids: tok.decode(ids, skip_special_tokens=True)
    logits = lambda ids: codec.decode_logits(codec.cumulative(codec.encode(ids))[-1])
ntok = lambda t: len(enc(t))

def rt(texts, bs=40):
    outs = []
    for i in range(0, len(texts), bs):
        ids = [enc(t)[: L - 1] + [EOS] for t in texts[i:i + bs]]
        y = mx.argmax(logits(mx.array([x + [PAD] * (L - len(x)) for x in ids])), -1); mx.eval(y)
        for r in y.tolist():
            r = r[: r.index(EOS)] if EOS in r else r
            outs.append(dec([t for t in r if t != PAD]))
    return outs

# ---------------- data
POOL = dict(
 PER=["Okonkwo", "Mlauzi", "Yordanka", "Dragomir", "Siobhan", "Hristo", "Ishan", "Ngozi", "Wojciech", "Aurelio", "Sarah Collins", "Mohammed Rahman", "Priya Nair", "Olena Kovalenko", "Tomasz Nowak", "Liam O'Connor"],
 LOC=["Plovdiv", "Veliko Tarnovo", "Ouagadougou", "Szczecin", "Reykjavik", "Cardiff", "Aberdeen", "Stara Zagora", "Lyon", "Gdansk", "Nairobi", "Tbilisi"],
 NUM=["340", "17", "2,500", "4.5", "981", "63", "1,200", "7.2%", "15%", "58"],
 DATE=["3 March", "14 May 2026", "12/07/2025", "2026-10-30", "9 June", "1 January", "22 August", "5 November"],
 AMT=["£4.5m", "$12,000", "300 euros", "2.5 million dollars", "€80", "$7.50"])
DENSE = ["{P1}, {P2}, {P3} and {P4} met in {L1}, {L2} and {L3}.",
 "{N1}, {N2}, {N3} and {N4} came from {L1} on {D1} and {D2}.",
 "{P1} paid {P2} {A1}, {P3} {A2} and {P4} {A3}.",
 "{P1} met {P2} on {D1}, {D2} and {D3} in {L1}.",
 "{P1} gave {N1} to {P2}, {N2} to {P3}, and {P1} kept {N3} for {L1}.",
 "{L1}, {L2}, {L3} and {L4} lost {N1}, {N2}, {N3} and {N4}."]
SLOT = dict(P="PER", L="LOC", N="NUM", D="DATE", A="AMT")
TPL = dict(
 xsum=["{P1} said {P2} paid {A1} for a house in {L1} on {D1}, police in {L2} said.",
       "{P1} and {P2} met in {L1} on {D1}, and {P1} left {N1} days later for {L2}.",
       "{N1} people, including {P1}, were hurt in {L1} on {D1}, {P2} said, and {A1} was lost near {L2}.",
       "{P1} told {P2} that {L1} would pay {A1} by {D1}, and {L2} would send {N1} staff.",
       "{P1} and {P2} said {L1} lost {N1} jobs after {D1}, while {P1} blamed {L2} for {A1}.",
       "Police said {P1} took {A1} from {P2} in {L1} on {D1}, and {N1} others fled to {L2}."],
 wmt14=["On {D1}, {P1} asked {P2} to visit {L1}, where {N1} members would discuss {A1} for {L2}.",
        "{P1} said that {L1} and {L2} would receive {A1} from {P2} before {D1}.",
        "Mr {P1} and Mr {P2} agreed that {N1} experts from {L1} would meet in {L2} on {D1}.",
        "{P1} reported that {L1} spent {A1} in {D1}, and {P2} asked why {L2} spent {N1} more."],
 tinystories=["{P1} and {P2} went to {L1} on {D1}. {P1} found {N1} coins and gave {N2} to {P3}.",
              "{P1} and {P2} lived in {L1}. {P1} had {N1} cats, {P2} had {N2} dogs, and on {D1} they met {P3} in {L2}.",
              "{P1} said, \"Let us go to {L1} on {D1}.\" {P2} had {A1}, and {P3} had {N1} toys for {L2}.",
              "One day, {P1} saw {P2} near {L1}. {P2} gave {P1} {N1} apples and {A1} on {D1}.",
              "{P1} took {N1} birds to {L1} and {N2} to {L2}. {P2} paid {A1} on {D1}."]
)
def gen(seed, n, tries=3000):
    rnd = random.Random(seed); out = []
    for _ in range(tries):
        if len(out) >= n: break
        t = rnd.choice(TPL[NAME] + DENSE); slots = sorted(set(re.findall(r"\{(\w\d)\}", t)))
        pick = {}; ok = True
        for ty in SLOT:
            ss = [s for s in slots if s[0] == ty]
            if ss: pick.update(zip(ss, rnd.sample(POOL[SLOT[ty]], len(ss))))
        text = t.format(**pick); ents = [(pick[s], SLOT[s[0]]) for s in slots]
        sf = [e[0] for e in ents]
        if any(a != b and a in b for a in sf for b in sf) or not 5 <= len(ents) <= 8 or ntok(text) > L - 1: continue
        out.append(dict(text=text, ents=ents))
    return out

# ---------------- surrogate candidates (12 each)
CAND = dict(
 PER=dict(A="Tom Lily Sam Max Anna Ben Mia Jack Kate Paul Emma Dan".split(),
          B="John Mary David Sarah James Peter Emma Mark Lucy Paul Kate Tom".split(),
          C="Tim Sue Bob Amy Tom Ann Joe Meg Ben Liz Sam Kim".split()),
 LOC=dict(A="London Paris Berlin Rome Madrid Dublin Moscow Tokyo Vienna Oslo Cairo Delhi".split(),
          B="Edinburgh Glasgow Cardiff Belfast Leeds Bristol Oxford Cambridge Brighton York Bath Dover".split(),
          C=[f"the {w}" for w in "park school town city shop lake farm zoo beach garden river hill".split()]),
 NUM=dict(A="two three four five six seven eight nine ten eleven twelve twenty".split(),
          B="2 3 4 5 6 7 8 9 10 11 12 20".split(),
          C="10 20 30 40 50 60 70 80 90 100 200 300".split()),
 DATE=dict(A="Monday Tuesday Wednesday Thursday Friday Saturday Sunday yesterday today tomorrow tonight Easter".split(),
           B="January February March April May June July August September October November December".split(),
           C="Christmas Easter Monday Friday Sunday Saturday summer winter spring autumn morning evening".split()),
 AMT=dict(A=["ten dollars", "five pounds", "twenty euros", "two dollars", "three pounds", "four euros", "six dollars", "seven pounds", "eight euros", "nine dollars", "fifty pounds", "five euros"],
          B=["$10", "$20", "$5", "$50", "$100", "$2", "$3", "$4", "$6", "$7", "$8", "$9"],
          C=["10 pounds", "20 pounds", "5 pounds", "50 pounds", "100 pounds", "2 pounds", "3 pounds", "4 pounds", "6 pounds", "7 pounds", "8 pounds", "9 pounds"]))
tbl = lambda ch: {t: CAND[t][ch[t]] for t in TYPES}
allw = lambda table: [w for v in table.values() for w in v]

# ---------------- evaluation
def run(items, table=None, dynamic=False, vocab=None, spans=None):
    """table None = raw baseline. vocab = scan vocabulary for raw rows."""
    inps, mps = [], []
    for it in items:
        ents = (spans(it) if spans else it["ents"]) if table else []
        mp = build_map(it["text"], ents, table, dynamic) if table else {}
        mps.append(mp); inps.append(mask(it["text"], mp))
    outs = rt(inps); V = allw(table) if table else vocab
    m = dict(n=len(items), ex=0, match=0, g=0, o=0, bind=0, cnt=0, nat=0, coll=0, sim=0.0, fails=[])
    for it, mp, inp, out in zip(items, mps, inps, outs):
        g, o = occ(inp, V), occ(out, V); fin = unmask(out, mp)
        c = sum((Counter(g) & Counter(o)).values()); m["match"] += c; m["g"] += len(g); m["o"] += len(o)
        if len(g) == len(o): m["bind"] += sum(a != b for a, b in zip(g, o))
        else: m["cnt"] += 1
        ok = fin.strip() == it["text"].strip(); m["ex"] += ok
        m["sim"] += difflib.SequenceMatcher(None, it["text"], fin).ratio()
        if table:  # natural surrogate words wrongly mapped back by unmask(mask(x))
            st = mask(it["text"], {s_: "@" for s_ in mp})
            m["nat"] += len(occ(st, allw(table))); m["coll"] += len(occ(st, list(mp.values())))
        if not ok: m["fails"].append((it["text"], inp, out, fin))
    return m

def row(name, m, coll=True):
    rec = m["match"] / max(m["g"], 1); pre = m["match"] / max(m["o"], 1)
    cl = f"{m['coll']}/{m['nat']}" if coll and m["nat"] else "-"
    return f"{name:<34}| {m['ex']/m['n']:6.1%} | {rec:6.1%} | {pre:6.1%} | {m['bind']:4d} | {m['cnt']:3d} | {m['sim']/m['n']:.3f} | {cl}"
HDR = f"{'condition':<34}| exact% | recall | prec   | bind | cnt | chrsim | collisions"
LOG = []
def P(s=""): print(s); LOG.append(s)

# ---------------- main
main, calib = gen(1, NMAIN), gen(2, 60)
ne = [len(i["ents"]) for i in main]; nocc = [len(occ(i["text"], [e[0] for e in i["ents"]])) for i in main]
P(f"== codec {NAME}: seq_len={L}; main n={len(main)} texts, entities/text {min(ne)}-{max(ne)} (mean {sum(ne)/len(ne):.1f}), occurrences/text mean {sum(nocc)/len(nocc):.1f}; calib n={len(calib)}")
P("example: " + main[0]["text"])
RAWV = [s for v in POOL.values() for s in v]
res = {}

# coordinate descent on calibration for the per-slot best surrogate set
ch = {t: "A" for t in TYPES}
score = lambda c: (lambda m: m["ex"] / m["n"] + m["match"] / max(m["g"], 1))(run(calib, tbl(c)))
best = score(ch); P("\nslot-set search on calibration (score=exact+recall):")
for t in TYPES:
    sc = {}
    for o in "ABC":
        c = dict(ch, **{t: o}); sc[o] = score(c) if o != ch[t] else best
    ch[t] = max(sc, key=sc.get); best = sc[ch[t]]
    P(f"  {t:<4} " + " ".join(f"{o}={v:.3f}" for o, v in sc.items()) + f" -> {ch[t]}")
TUNED = tbl(ch)
P("\nTUNED surrogate table (first 8 per slot):")
for t in TYPES: P(f"  {t:<4} set {ch[t]}: " + ", ".join(TUNED[t][:8]))

P("\n### 1/2/5 main matrix (5-8 entities/text, repeats, mixed types)"); P(HDR)
rows = [("baseline RAW entities", run(main, vocab=RAWV), False)]
for o in "ABC": rows.append((f"masked all-{o}", run(main, tbl({t: o for t in TYPES})), True))
tm = run(main, TUNED); rows.append(("masked TUNED", tm, True))
det = lambda it: [e for e in detect(it["text"])]
rows.append(("masked TUNED, regex-detected spans", run(main, TUNED, spans=det), True))
for n_, m_, c_ in rows: P(row(n_, m_, c_))
gs = Counter(); tot = Counter()
for it in main:
    d = {e[0] for e in detect(it["text"])}; gset = {e[0] for e in it["ents"]}
    gs["tp"] += len(d & gset); gs["g"] += len(gset); gs["d"] += len(d)
P(f"regex detector alone: span recall {gs['tp']/gs['g']:.1%}, precision {gs['tp']/gs['d']:.1%} (types PER/LOC conflated)")

# ---------------- collisions
def nat_items(table, seed, n=40):
    rnd = random.Random(seed); base = gen(seed, 4 * n); out = []
    cl = dict(PER="They watched the film {W} twice.", NUM="Only {W} of them stayed.", DATE="It was cold last {W}.", LOC="We read about {W} again.")
    for it in base:
        ty = rnd.sample(list(cl), 2); ws = [rnd.choice(table[t][:4]) for t in ty]
        txt = it["text"] + " " + " ".join(cl[t].format(W=w) for t, w in zip(ty, ws))
        # entity surface must not equal a natural word; length cap
        if ntok(txt) <= L - 1 and not any(e[0] in ws for e in it["ents"]): out.append(dict(text=txt, ents=it["ents"]))
        if len(out) >= n: break
    return out
ni = nat_items(TUNED, 3)
P(f"\n### 3 collisions: {len(ni)} texts with 2 natural (non-entity) surrogate words each, e.g.\n  " + ni[0]["text"])
P(HDR)
sn, dy = run(ni, TUNED), run(ni, TUNED, dynamic=True)
P(row("static table (naive)", sn)); P(row("dynamic table (skip words in input)", dy))
P(f"  static unmask(mask(x))==x without any codec: {sum(unmask(mask(i['text'], build_map(i['text'], i['ents'], TUNED)), build_map(i['text'], i['ents'], TUNED)) == i['text'] for i in ni)}/{len(ni)}; "
  f"dynamic: {sum((lambda mp: unmask(mask(i['text'], mp), mp) == i['text'])(build_map(i['text'], i['ents'], TUNED, True)) for i in ni)}/{len(ni)}")
P(f"  the dynamic table needs >= {max(len(i['ents']) for i in ni)}+{4} spare words/slot (12 provided)")

# ---------------- German control
GER = [("Frau Müller wohnt seit 2019 in Düsseldorf.", "Müller:PER 2019:NUM Düsseldorf:LOC"),
 ("Jürgen Weiß kauft am 3. März ein Auto für 12.500 Euro.", "Jürgen Weiß:PER 3. März:DATE 12.500 Euro:AMT"),
 ("Björn und Käthe fahren nach Nürnberg.", "Björn:PER Käthe:PER Nürnberg:LOC"),
 ("Herr Köhler zahlte 340 Euro für die Reparatur in Würzburg.", "Köhler:PER 340 Euro:AMT Würzburg:LOC"),
 ("Am 14. Juni trifft Günther Schäfer seine Schwester in Zürich.", "14. Juni:DATE Günther Schäfer:PER Zürich:LOC"),
 ("Die Straße in Göttingen ist seit Montag gesperrt.", "Göttingen:LOC Montag:DATE"),
 ("Anneliese Schröder hat 17 Äpfel aus Lübeck mitgebracht.", "Anneliese Schröder:PER 17:NUM Lübeck:LOC"),
 ("Der Zug von München nach Hamburg hatte 45 Minuten Verspätung.", "München:LOC Hamburg:LOC 45:NUM"),
 ("Frau Böhm arbeitet am Freitag in Mönchengladbach.", "Böhm:PER Freitag:DATE Mönchengladbach:LOC"),
 ("Mein Onkel Jörg wurde 1962 in Osnabrück geboren.", "Jörg:PER 1962:NUM Osnabrück:LOC"),
 ("Die Schüler aus Köln und Düsseldorf zahlen 8 Euro Eintritt.", "Köln:LOC Düsseldorf:LOC 8 Euro:AMT"),
 ("Hans Müller und Grete Fröhlich haben am 12.07.2025 geheiratet.", "Hans Müller:PER Grete Fröhlich:PER 12.07.2025:DATE"),
 ("Das Konzert in Zürich beginnt am Samstag um 20 Uhr.", "Zürich:LOC Samstag:DATE 20:NUM"),
 ("Herr Gößmann überwies 1.200 Euro an Frau Jäger.", "Gößmann:PER 1.200 Euro:AMT Jäger:PER"),
 ("Die Brücke bei Lüneburg wurde im Mai 2024 für 3,5 Millionen Euro saniert.", "Lüneburg:LOC Mai 2024:DATE 3,5 Millionen Euro:AMT")]
gi = [dict(text=t, ents=[tuple(e.rsplit(":", 1)) for e in re.findall(r"(?:\S+ )*?\S+:[A-Z]+", es)]) for t, es in GER]
# the split above is greedy-safe because each entity ends with ':TYPE'
um = lambda it: [e for e in it["ents"] if re.search("[äöüßÄÖÜ]", e[0])]
GV = sorted({e[0] for i in gi for e in i["ents"]})
trunc = sum(ntok(i["text"]) > L - 1 for i in gi)
P(f"\n### 4 German control: {len(gi)} sentences, {sum(len(i['ents']) for i in gi)} entities ({sum(len(um(i)) for i in gi)} with umlaut/ss), {trunc} exceed seq_len; entities mean {sum(len(i['ents']) for i in gi)/len(gi):.1f}/sent")
P(HDR)
P(row("German RAW", run(gi, vocab=GV), False))
P(row("German masked umlaut-entities only", run(gi, TUNED, True, spans=um)))
gm = run(gi, TUNED, True); P(row("German masked all entities", gm))
P(row("English main RAW (reference)", rows[0][1], False))
for t, i, o, f in gm["fails"][:4]: P(f"  GER REF {t}\n      MASK-IN {i}\n      OUT     {f}")

# ---------------- failure examples
P("\n### failure examples (REF -> OUT after unmask)")
for n_, m_, _ in [rows[0], ("TUNED", tm, 0)]:
    P(f"[{n_}] {len(m_['fails'])}/{m_['n']} texts not exact")
    for t, i, o, f in m_["fails"][:5]: P(f"  REF {t}\n  IN  {i}\n  OUT {o}\n  UNM {f}")
P(f"\nverdict-metric: TUNED recall {tm['match']/max(tm['g'],1):.2%} (target >=99.5%), binding errors {tm['bind']}, count-err texts {tm['cnt']}, spurious surrogate outputs {tm['o']-tm['match']}")
open(f"{WORK}/mask_{NAME}.txt", "w").write("\n".join(LOG) + "\n")
