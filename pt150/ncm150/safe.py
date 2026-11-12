"""Safe-token masking (port of seam/safe_eval.py): name tokens that are not in the entity table are blocked at decode time.
Pure functions on token ids / logits; no tokenizer needed except in build_nameset."""
from dataclasses import dataclass
import torch

SURROGATES = ["John", "Mary", "David", "Sarah", "James", "Peter", "Emma", "Mark", "Tom", "Lily", "Anna", "Paul", "Lucy", "Ben", "Kate", "Jack"]
NAMES = SURROGATES + """Michael Robert William Richard Thomas Charles Daniel Matthew Anthony Donald Steven Andrew Kenneth Joshua Kevin Brian George
Edward Ronald Timothy Jason Jeffrey Ryan Jacob Gary Nicholas Eric Stephen Jonathan Larry Justin Scott Brandon Frank Benjamin Gregory
Samuel Raymond Patrick Alexander Jack Dennis Jerry Tyler Aaron Henry Douglas Nathan Adam Zachary Walter Kyle Harold Carl Arthur Gerald
Roger Keith Jeremy Lawrence Sean Christian Albert Joe Ethan Austin Jesse Willie Billy Bryan Bruce Jordan Ralph Roy Noah Dylan Eugene
Patricia Jennifer Linda Elizabeth Barbara Susan Jessica Karen Nancy Lisa Betty Margaret Sandra Ashley Kimberly Emily Donna Michelle
Carol Amanda Dorothy Melissa Deborah Stephanie Rebecca Sharon Laura Cynthia Kathleen Amy Angela Shirley Brenda Pamela Nicole Samantha
Katherine Christine Helen Debra Rachel Carolyn Janet Maria Olivia Heather Diane Julie Joyce Victoria Kelly Christina Lauren Joan Evelyn
Judith Megan Cheryl Andrea Hannah Jacqueline Martha Gloria Teresa Sara Janice Julia Grace Judy Abigail Alice Sophia Charlotte Smith
Johnson Williams Brown Jones Miller Davis Wilson Anderson Taylor Moore Jackson Martin Lee Thompson White Harris Clark Lewis Walker Hall
Allen Young King Wright Scott Green Baker Adams Nelson Hill Campbell Mitchell Roberts Carter Phillips Evans Turner Parker Collins""".split()

@dataclass
class NameSet:
    ids: dict      # key -> frozenset of token ids (single-token variants: " Name", "Name")
    canon: dict    # key -> id used for the table slot in the prefix
    V: int
    def __post_init__(self):
        self.tok2key = {t: k for k, s in self.ids.items() for t in s}
        self.all = torch.zeros(self.V, dtype=torch.bool)
        for t in self.tok2key: self.all[t] = True

def build_nameset(encode, V, names=NAMES):
    """encode: str -> list[int] (GPT-2 BPE). Multi-token names are skipped (not maskable at one position)."""
    ids, canon = {}, {}
    for n in dict.fromkeys(names):
        s = {e[0] for e in (encode(" " + n), encode(n)) if len(e) == 1 and e[0] < V}
        if s: ids[n] = frozenset(s); canon[n] = encode(" " + n)[0] if len(encode(" " + n)) == 1 else min(s)
    return NameSet(ids, canon, V)

def synthetic_nameset(V, n=8, start=10):
    ids = {f"n{i}": frozenset({start + i}) for i in range(n)}
    return NameSet(ids, {k: min(s) for k, s in ids.items()}, V)

def update_table(table, toks, ns):
    """table: ordered unique name keys seen so far (= surrogate entity table); first appearance order."""
    out = list(table)
    for t in toks:
        k = ns.tok2key.get(int(t))
        if k is not None and k not in out: out.append(k)
    return out

def table_masks(tables, ns):
    """(bad, ok) both (B,V) bool. bad = name token not in that row's table; ok = name token in the table."""
    ok = torch.zeros(len(tables), ns.V, dtype=torch.bool)
    for j, t in enumerate(tables):
        for k in t: ok[j, list(ns.ids[k])] = True
    return ns.all[None] & ~ok, ok

def mask_logits(lg, bad):
    return lg.masked_fill(bad.to(lg.device).unsqueeze(1) if lg.dim() == 3 else bad.to(lg.device), -torch.inf)

def choose(lg, mode, bad, ok, pick=lambda x: x.argmax(-1)):
    """lg (B,T,V) or (B,V) -> (ids, hits) where hits = positions whose plain argmax was a disallowed name.
    none: pick(lg); mask: pick(masked); slot: pick(lg), but a disallowed name is replaced by the best allowed name
    (row has a table) or by the masked pick (empty table)."""
    bad, ok = bad.to(lg.device), ok.to(lg.device)
    b = bad.unsqueeze(1) if lg.dim() == 3 else bad
    hits = int(torch.gather(b.expand_as(lg), -1, lg.argmax(-1, keepdim=True)).sum())
    if mode == "none": return pick(lg), hits
    m = pick(mask_logits(lg, bad))
    if mode == "mask": return m, hits
    assert mode == "slot", mode
    out = pick(lg)
    isbad = torch.gather(b.expand_as(lg), -1, out.unsqueeze(-1)).squeeze(-1)
    o = ok.unsqueeze(1) if lg.dim() == 3 else ok
    best = lg.masked_fill(~o, -torch.inf).argmax(-1)
    has = ok.any(-1).view(-1, *([1] * (out.dim() - 1)))
    return torch.where(isbad, torch.where(has, best, m), out), hits
