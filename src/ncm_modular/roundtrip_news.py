"""Fact-preservation round trip through the NCM codec (xsum / wmt14). MLX on CPU (needs mlx + transformers)."""
import sys, os, re, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
import numpy as np, mlx.core as mx
from transformers import AutoTokenizer
import codec_mlx
from texts import *
mx.set_default_device(mx.cpu)
task = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
c = codec_mlx.load(f"{WORK}/mlx/{task}")
tok = AutoTokenizer.from_pretrained(f"{WORK}/tok/{'gpt2' if task == 'xsum' else 'Qwen3-0.6B'}")
sp = c.cfg
L, EOS, PAD = sp["seq_len"], tok.convert_tokens_to_ids(sp["eos_token"]), sp["prefix_pad_token_id"]
print(f"== {task}: L={L} EOS={EOS} PAD={PAD} levels={c.levels}")
INDIST = NEWS if task == "xsum" else PLAIN

def roundtrip(ids_list):
    ids = [i[: L - 1] + [EOS] for i in ids_list]
    x = mx.array([i + [PAD] * (L - len(i)) for i in ids])
    zs = c.cumulative(c.encode(x))
    rows = []
    for z in zs:
        y = mx.argmax(c.decode_logits(z), -1); mx.eval(y); rows.append(y.tolist())
    acc = [sum(a == b for i, r in zip(ids, rows[k]) for a, b in zip(i, r[: len(i)])) / sum(map(len, ids)) for k in range(len(zs))]
    outs = [r[: r.index(EOS)] if EOS in r else r for r in rows[-1]]
    outs = [[t for t in o if t != PAD] for o in outs]
    return ids, outs, acc

def dec(ids): return tok.decode(ids, skip_special_tokens=True)
ENT = re.compile(r"(?<![.!?] )(?<!^)\b[A-Z][\w'-]+(?: [A-Z][\w'-]+)*|\d[\d.,%/:-]*\d%?|\d")

def report(name, refs, outs, acc, ents, exact_ids=None):
    exact = np.mean([o == r for r, o in exact_ids]) if exact_ids else np.mean([a.strip() == b.strip() for a, b in zip(outs, refs)])
    kept = [(e, e in o) if isinstance(o, str) else (e, o.count(e) >= 1) for e, o in ents]
    ent = np.mean([k for _, k in kept]) if kept else float("nan")
    print(f"\n== {task} / {name}: n={len(refs)} exact={exact:.0%} entity_kept={ent:.0%} (entities={len(kept)})")
    print("  token acc per level: " + " ".join(f"{a:.2f}" for a in acc))
    shown = 0
    for r, o in zip(refs, outs):
        if r.strip() != o.strip() and shown < 3: print(f"  REF {r}\n  OUT {o}"); shown += 1
    lost = sorted({str(e) for e, k in kept if not k})
    if lost: print("  lost:", lost[:25])

def text_case(name, texts, ents_fn):
    ids, outs, acc = roundtrip([tok.encode(t) for t in texts])
    o = [dec(x) for x in outs]
    report(name, texts, o, acc, [(e, oo) for t, oo in zip(texts, o) for e in ents_fn(t)])

def cases(pool, k):
    rnd = random.Random(0)
    return [(t, {s: rnd.choice(v) for s, v in pool.items()}) for _ in range(k) for t in TPL]

def tpl_case(name, pool, k):
    cs = cases(pool, k); refs = [t.format(**e) for t, e in cs]
    ids, outs, acc = roundtrip([tok.encode(r) for r in refs]); o = [dec(x) for x in outs]
    report(name, refs, o, acc, [(v, oo) for (t, e), oo in zip(cs, o) for s, v in e.items() if "{" + s + "}" in t])

def ph_case(name, phids):
    """Placeholders = single token ids spliced between encoded segments."""
    names = dict(zip("abpnd", phids)); cs = cases(RAW, N); ids_in = []
    for t, _ in cs:
        parts = re.split(r"\{(\w)\}", t); ids = []
        for j, p in enumerate(parts):
            if j % 2: ids.append(names[p])
            elif p: ids += tok.encode(p)
        ids_in.append(ids[: L - 1])
    _, outs, acc = roundtrip(ids_in)
    lab = lambda ids: "".join(f"[{i}]" if i in phids else tok.decode([i]) for i in ids)
    ents = [(i, o) for ii, o in zip(ids_in, outs) for i in ii if i in phids]
    report(name, [lab(i) for i in ids_in], [lab(o) for o in outs], acc, ents, exact_ids=list(zip(ids_in, outs)))
    # strict: per-sample all placeholders in same order
    ok = np.mean([[x for x in o if x in phids] == [x for x in i if x in phids] for i, o in zip(ids_in, outs)])
    print(f"  placeholder sequence identical (order+count): {ok:.0%}")

text_case("(a) in-distribution " + ("news summaries" if task == "xsum" else "plain English"), INDIST, lambda t: ENT.findall(t))
tpl_case("(b) RAW rare names/places/numbers/dates", RAW, N)
tpl_case("(c) common surrogates Tom/Lily/three", SUR, 1)
tpl_case("(d) literal tags <P1>..", TAG, 1)
if task == "xsum":
    print("\nvocab: gpt2 has only <|endoftext|> special; ids 50258..50303 are spare slots of the 50304 embedding table")
    ph_case("(e) spare slots 50258..50262 as placeholders", [50258 + i for i in range(5)])
else:
    names = ["<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>"]
    ph_case("(e1) Qwen special tokens " + ",".join(names), tok.convert_tokens_to_ids(names))
    ph_case("(e2) spare slots 151669..151673", [151669 + i for i in range(5)])
