"""Surrogate masking: replace entities by frequent in-distribution words, map back after decode."""
import re

TYPES = ("PER", "LOC", "NUM", "DATE", "AMT")
_b = lambda alts: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(set(alts), key=len, reverse=True)) + r")(?!\w)")

def words(text): return {w.lower() for w in re.findall(r"\w+", text)}

def build_map(text, ents, table, dynamic=False):
    """ents: [(surface,type)] -> {surface: surrogate}. Same entity -> same surrogate; first-appearance order.
    dynamic: skip surrogates that occur in the input (case-insens., whole phrase)."""
    low = text.lower() if dynamic else None
    used, mp = set(), {}
    for s, t in sorted(set(ents), key=lambda e: text.find(e[0])):
        for c in table[t]:
            if c in used or (dynamic and re.search(r'(?<!\w)' + re.escape(c.lower()) + r'(?!\w)', low)): continue
            used.add(c); mp[s] = c; break
        else: raise ValueError(f"table exhausted for {t}")
    return mp

def mask(text, mp): return _b(mp).sub(lambda m: mp[m.group(0)], text) if mp else text

def unmask(text, mp):
    inv = {v: k for k, v in mp.items()}
    return _b(inv).sub(lambda m: inv[m.group(0)], text) if inv else text

def occ(text, vocab): return _b(vocab).findall(text) if vocab else []

# regex detector (no NER model): amounts, dates, numbers, capitalised mid-sentence runs (PER/LOC not separable)
MON = "January|February|March|April|May|June|July|August|September|October|November|December"
DET = [("AMT", r"[£$€]\d[\d,.]*(?:m|bn)?|\d[\d,.]*(?: million| billion)? (?:dollars|euros|pounds)"),
       ("DATE", rf"\d{{1,2}}(?:st|nd|rd|th)? (?:{MON})(?: \d{{4}})?|\d{{1,4}}[/.-]\d{{1,2}}[/.-]\d{{2,4}}|\d{{4}}-\d\d-\d\d"),
       ("NUM", r"\d[\d,.]*\d%?|\d%?"),
       ("PER", r"(?<![.!?\"] )(?<!^)\b[A-ZÄÖÜ][\w'’-]+(?: [A-ZÄÖÜ][\w'’-]+)*")]
def detect(text):
    out, taken = [], []
    for t, p in DET:
        for m in re.finditer(p, text):
            if any(m.start() < e and s < m.end() for s, e in taken): continue
            taken.append((m.start(), m.end())); out.append((m.group(0), t))
    return out
