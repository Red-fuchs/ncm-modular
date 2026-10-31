"""MASSIVE de-DE -> masked prefix / structured target pairs; round-trip targets through wmt14 codec (CPU)."""
import sys, os, re, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
import numpy as np, pyarrow.parquet as pq, mlx.core as mx
mx.set_default_device(mx.cpu)
FMT = os.environ.get("MFMT") or (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in "ABC" else "C")
SUR = "John Mary David Sarah James Peter Emma Mark Lucy Paul Kate Tom Anna Mike Linda Alex".split()
SPAN = re.compile(r"\[(\w+) : ([^\]]+)\]")
R = WORK+"/massive_raw"; OUT = WORK+"/massive_de"; os.makedirs(OUT, exist_ok=True)
feat = json.loads(pq.read_schema(f"{R}/train.parquet").metadata[b"huggingface"])["info"]["features"]
INTENTS = feat["intent"]["names"]
SLOTNAMES = set()
for _f in ('train','validation','test'):
    for _a in pq.read_table(f'{R}/{_f}.parquet').column('annot_utt').to_pylist(): SLOTNAMES |= {m.group(1) for m in re.finditer(r'\[(\w+) : ', _a)}

ILAB = {"datetime_convert": "time conversion", "datetime_query": "date and time query", "general_quirky": "quirky chat",
  "iot_hue_lightdim": "dim the lights", "iot_hue_lightoff": "lights off", "iot_hue_lighton": "lights on", "iot_hue_lightup": "brighten the lights",
  "iot_hue_lightchange": "change light color", "iot_wemo_on": "plug on", "iot_wemo_off": "plug off", "iot_cleaning": "vacuum cleaning",
  "iot_coffee": "coffee machine", "music_settings": "music preferences", "play_audiobook": "audio book", "qa_currency": "currency question",
  "qa_definition": "definition question", "qa_factoid": "fact question", "qa_maths": "math question", "qa_stock": "stock question",
  "takeaway_order": "order food", "takeaway_query": "food delivery query", "email_addcontact": "add contact", "email_querycontact": "find contact",
  "audio_volume_mute": "mute", "audio_volume_up": "volume up", "audio_volume_down": "volume down", "audio_volume_other": "volume other",
  "email_sendemail": "send email", "lists_createoradd": "list add", "play_podcasts": "play show", "social_post": "social post"}
SLAB = {"podcast_name": "show name", "weather_descriptor": "weather condition", "list_name": "list title", "audiobook_author": "book author", "audiobook_name": "book name", "timeofday": "time of day"}
def ilab(n): return ILAB.get(n, n.replace("_", " ")) if FMT == "C" else n.replace("_", " ")
def slab(n): return SLAB.get(n, n.replace("_", " ")) if FMT == "C" else n.replace("_", " ")
IMAP = {ilab(n): n for n in INTENTS}
SMAP = {}

def intent_str(i):
    n = INTENTS[i]
    return n if FMT == "A" else ilab(n)

def target_str(i, items):
    if FMT == "C": return f"The intent is {ilab(INTENTS[i])}." + "".join(f" The {slab(k)} is {v}." for k, v in items)
    return " | ".join([intent_str(i)] + [f"{k if FMT=='A' else slab(k)}={v}" for k, v in items])

def build(row):
    ann = row["annot_utt"]; parts, vals, pos = [], [], 0
    for m in SPAN.finditer(ann):
        parts.append(ann[pos:m.start()]); vals.append((m.group(1), m.group(2).strip())); pos = m.end()
    tail = ann[pos:]
    free = [s for s in SUR if not re.search(rf"(?<!\w){s.lower()}(?!\w)", row["utt"].lower())]
    mp = {}
    for _, v in vals:
        if v not in mp: mp[v] = free[len(mp)]
    pre = "".join(p + mp[v] for p, (_, v) in zip(parts, vals)) + tail
    tgt_s = target_str(row["intent"], [(s, mp[v]) for s, v in vals])
    gold = [(s, v) for s, v in vals]
    return dict(utt=row["utt"], prefix=pre, target=tgt_s, intent=INTENTS[row["intent"]], slots=gold, map={k: s for k, s in mp.items()}, id=row["id"])

def parse(t, inv):
    """decoded target string (surrogate form) -> (intent_name, [(slot, value)]) with unmask."""
    if FMT == "C":
        m = re.match(r"The intent is (.*?)\.(?: |$)", t)
        it = IMAP.get(m.group(1), m.group(1)) if m else ""
        rev = {slab(n): n for n in {s for s in SLOTNAMES}}
        sl = [(rev.get(k, k), inv.get(v, v)) for k, v in re.findall(r"The (.*?) is ([^.]*)\.", t[m.end():] if m else "")]
        return it, sl
    segs = [x.strip() for x in t.split("|")]
    it = segs[0] if FMT == "A" else IMAP.get(segs[0], segs[0])
    sl = []
    for s in segs[1:]:
        if "=" in s:
            k, v = s.split("=", 1); sl.append((k.strip(), inv.get(v.strip(), v.strip())))
    return it, sl

if __name__ == "__main__":
    data = {}
    for sp, f in [("train", "train"), ("dev", "validation"), ("test", "test")]:
        rows = pq.read_table(f"{R}/{f}.parquet").to_pylist()
        data[sp] = [build(r) for r in rows]
        with open(f"{OUT}/{sp}.jsonl", "w") as fh:
            for d in data[sp]: fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    print({k: len(v) for k, v in data.items()}); 
    for d in data["train"][:5]: print(d["prefix"], "->", d["target"])
    import codec_mlx
    from transformers import AutoTokenizer
    c = codec_mlx.load(WORK+"/mlx/wmt14"); tok = AutoTokenizer.from_pretrained(WORK+"/tok/Qwen3-0.6B")
    L, EOS, PAD = 64, c.cfg["eos_token_id"] if "eos_token_id" in c.cfg else 151645, c.cfg["prefix_pad_token_id"]
    def rt(texts, bs=64):
        U = sorted(set(texts)); res = rt0(U, bs); m = dict(zip(U, res[0])); lm = dict(zip(U, res[1]))
        return [m[t] for t in texts], [lm[t] for t in texts], res[2]
    def rt0(texts, bs=64):
        outs, lens, trunc = [], [], 0
        for i in range(0, len(texts), bs):
            ids = [tok.encode(t) for t in texts[i:i+bs]]
            trunc += sum(len(x) > L - 1 for x in ids); lens += [len(x) for x in ids]
            ids = [x[:L-1] + [EOS] for x in ids]
            x = mx.array([a + [PAD] * (L - len(a)) for a in ids])
            y = mx.argmax(c.decode_logits(c.cumulative(c.encode(x))[-1]), -1); mx.eval(y); mx.clear_cache()
            for r in y.tolist():
                r = r[:r.index(EOS)] if EOS in r else r
                outs.append(tok.decode([t for t in r if t != PAD], skip_special_tokens=True))
        return outs, lens, trunc
    rep = {"format": FMT}
    for sp in ["train", "dev", "test"]:
        D = data[sp]; outs, lens, trunc = rt([d["target"] for d in D])
        ex = [o.strip() == d["target"].strip() for o, d in zip(outs, D)]
        ok = []
        for o, d in zip(outs, D):
            inv = {v: k for k, v in d["map"].items()}
            it, sl = parse(o, inv); ok.append(it == d["intent"] and sl == [tuple(x) for x in d["slots"]])
        DS = D[:300]; po, pl, pt = rt([d["prefix"] for d in DS])
        pex = np.mean([o.strip() == d["prefix"].strip() for o, d in zip(po, DS)])
        rep[sp] = dict(n=len(D), target_exact=float(np.mean(ex)), target_unmasked_exact=float(np.mean(ok)), target_tok_mean=float(np.mean(lens)), target_trunc=trunc, prefix_exact=float(pex), prefix_tok_mean=float(np.mean(pl)), prefix_tok_p99=float(np.percentile(pl, 99)), prefix_tok_max=int(max(pl)), prefix_trunc=pt)
        bad = [(d["target"], o) for o, d, e in zip(outs, D, ex) if not e][:8]
        rep[sp]["bad_examples"] = bad
        print(sp, {k: v for k, v in rep[sp].items() if k != "bad_examples"}); [print("  ", b) for b in bad]
    json.dump(rep, open(f"{OUT}/report_{FMT}.json", "w"), ensure_ascii=False, indent=1)
