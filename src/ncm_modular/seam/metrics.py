"""entity consistency, spurious names, repetition from runs/gen_*.json (+gold). CPU only."""
import sys, os, re, json, glob, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stories import *
def ments(t): return [m.group(0) for m in re.finditer(SRE.pattern + "|" + NRE.pattern, t)]
def ngr(t, n=4): w = re.findall(r"[a-z']+", t.lower()); return [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
def sentset(t): return {s.strip().lower() for s in sents(t) if len(s.split()) > 3}
def analyse(stories, key):
    cons = legit = spur = tot = 0; st_spur = 0; leak = 0; persist = [0, 0]; rep = [0, 0]; srep = [0, 0]; wrep = [0, 0]; nstories = len(stories)
    for s in stories:
        bl = s[key]; whole = set(s["table"]); seen = set(ments(bl[0])); b0 = set(seen); hs = set(); sp_flag = 0
        grams = set(ngr(bl[0])); sset = sentset(bl[0])
        for b in bl[1:]:
            ms = ments(b)
            for m in ms:
                tot += 1
                if m in seen: cons += 1
                elif m in SUR and m in whole: legit += 1
                else:
                    spur += 1; sp_flag = 1; leak += m not in SUR
            if b0: persist[1] += 1; persist[0] += bool(b0 & set(ms))
            g = ngr(b); rep[1] += len(g); rep[0] += sum(x in grams for x in g)
            gg = set(); 
            for x in g:
                wrep[0] += x in gg; gg.add(x)
            wrep[1] += len(g)
            ss = sentset(b); srep[1] += len(ss); srep[0] += len(ss & sset)
            seen |= set(ms); grams |= set(g); sset |= ss
        st_spur += sp_flag
    return dict(mentions=tot, consistent=cons / max(tot, 1), legit_new=legit / max(tot, 1), spurious=spur / max(tot, 1), raw_name_leak=leak / max(tot, 1),
                stories_with_spurious=st_spur / nstories, b0_entity_persist=persist[0] / max(persist[1], 1),
                xblock_4gram_repeat=rep[0] / max(rep[1], 1), inblock_4gram_repeat=wrep[0] / max(wrep[1], 1), dup_sentence_rate=srep[0] / max(srep[1], 1))
out = {}
files = sorted(glob.glob(f"{RUNS}/gen_*.json"))
d0 = json.load(open(files[0])); out["gold"] = analyse([dict(gold=s["gold"], table=s["table"]) for s in d0["stories"]], "gold")
for f in files:
    d = json.load(open(f)); k = f"{d['v']}_t{d['temp']}"
    out[k] = analyse(d["stories"], "gen"); out[k].update(lat_b1_ms=d["lat_b1_ms"], batch_ms_per_block=d["batch_ms_per_block"], NFE=d["NFE"], P=d["P"])
json.dump(out, open(f"{RUNS}/metrics.json", "w"), indent=1)
for k, v in out.items(): print(k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})
