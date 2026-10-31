"""Metrics on decoded target strings (surrogate form) for MASSIVE structured output."""
import os, sys, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MFMT", "C")
import massive_build as mb

def score(preds, D):
    """preds: decoded strings. returns dict of metrics."""
    tp = fp = fn = 0; it_ok = ex = 0; rec_n = rec_hit = 0; valid = 0; inv_sur = 0
    for p, d in zip(preds, D):
        inv = {v: k for k, v in d["map"].items()}
        it, sl = mb.parse(p, inv)
        gold = [tuple(x) for x in d["slots"]]
        it_ok += it == d["intent"]; ex += (it == d["intent"] and sl == gold)
        g, q = collections.Counter(gold), collections.Counter(sl)
        c = sum((g & q).values()); tp += c; fp += sum(q.values()) - c; fn += sum(g.values()) - c
        rec_n += len(gold); rec_hit += c
        # surrogate integrity: every surrogate the output uses must exist in the prefix map
        segs = mb.re.findall(r"The .*? is ([^.]*)\.", p)
        bad = [s for s in segs[1:] if s not in inv] if len(segs) > 1 else []
        inv_sur += len(bad) > 0
    n = len(D); pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(n=n, intent_acc=it_ok / n, slot_f1=2 * pr * rc / max(pr + rc, 1e-9), slot_p=pr, slot_r=rc,
                exact_match=ex / n, fact_restoration=rec_hit / max(rec_n, 1), invalid_surrogate_rate=inv_sur / n)
