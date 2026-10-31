"""Story load/split/mask/block utilities."""
import re, json, random, os
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); from paths import *
SUR = ["John","Mary","David","Sarah","James","Peter","Emma","Mark"]
NAMES = json.load(open(f"{DATA}/names.json"))
NRE = re.compile(r"(?<![\w'])(?:" + "|".join(sorted(map(re.escape, NAMES), key=len, reverse=True)) + r")(?![\w'])")
SRE = re.compile(r"(?<![\w'])(?:" + "|".join(SUR) + r")(?![\w'])")
def load_stories(path=f"{DATA}/valid.txt"):
    out = []
    for s in open(path).read().split("<|endoftext|>"):
        s = " ".join(s.split())
        if len(s) > 400: out.append(s)
    return out
def mask_story(s):
    """consistent per-story surrogate map in first-appearance order. returns (masked, ordered surrogate list, map) or None."""
    if SRE.search(s): return None            # story already contains a surrogate word as a normal word
    mp = {}
    for m in NRE.finditer(s):
        if m.group(0) not in mp:
            if len(mp) >= len(SUR): return None
            mp[m.group(0)] = SUR[len(mp)]
    return NRE.sub(lambda m: mp[m.group(0)], s), list(mp.values()), mp
def sents(s): return re.findall(r'.+?(?:[.!?]+["”]?(?=\s|$)|$)', s)
def blocks(s, enc, L, tgt=48):
    """greedy sentence packing: token ids of each block <= L-1 (EOS added by caller). returns list of (text, ids)."""
    out, cur, curn = [], [], 0
    for se in sents(s):
        se = se.strip()
        if not se: continue
        n = len(enc(" " + se)) if cur else len(enc(se))
        if cur and curn + n > L - 1: out.append(" ".join(cur)); cur, curn = [], 0; n = len(enc(se))
        if n > L - 1: continue                 # pathological sentence: skip
        cur.append(se); curn += n
        if curn >= tgt: out.append(" ".join(cur)); cur, curn = [], 0
    if cur: out.append(" ".join(cur))
    return out
def split(stories, seed=0):
    r = random.Random(seed); idx = list(range(len(stories))); r.shuffle(idx)
    n = len(idx); te, dv = idx[:1500], idx[1500:2100]
    return idx[2100:], dv, te
