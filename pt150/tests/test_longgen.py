import json, torch, pytest
from ncm150 import eval_long as ev
from ncm150.generate import GenCfg, build_prefix, generate_long, load_model, window_tail
from ncm150.models import build_codec, build_model
from ncm150.safe import build_nameset, choose, mask_logits, synthetic_nameset, table_masks, update_table
from conftest import tiny_cfg

V = 128
NS = synthetic_nameset(V, n=8)  # names: ids 10..17 = n0..n7

def test_build_nameset_skips_multitoken():
    enc = lambda s: {" Ann": [20], "Ann": [21], " Bob": [30, 31], "Bob": [32, 33]}.get(s, [1, 2])
    ns = build_nameset(enc, 64, ["Ann", "Bob"])
    assert ns.ids == {"Ann": frozenset({20, 21})} and ns.canon["Ann"] == 20 and ns.all.sum() == 2

def test_update_table_order_and_idempotent():
    t = update_table([], [1, 12, 3, 10, 12, 99], NS); assert t == ["n2", "n0"]
    assert update_table(t, [10, 15], NS) == ["n2", "n0", "n5"]

def test_masks_and_none_mode_is_argmax():
    bad, ok = table_masks([["n0"], []], NS)
    assert not bad[0, 10] and bad[0, 11] and ok[0, 10] and not bad[0, 5]      # non-names never bad
    assert bad[1, 10] and not ok[1].any()
    lg = torch.zeros(2, 3, V); lg[:, :, 11] = 5   # argmax = disallowed name n1
    out, hits = choose(lg, "none", bad, ok); assert (out == 11).all() and hits == 6

def test_mask_never_emits_disallowed():
    bad, ok = table_masks([["n0"], []], NS)
    lg = torch.zeros(2, 4, V); lg[:, :, 11] = 5; lg[:, :, 10] = 4
    out, _ = choose(lg, "mask", bad, ok)
    assert (out[0] == 10).all()                      # best allowed name wins by logit
    assert not bad[1][out[1]].any() and (out[1] != 11).all()

def test_slot_backlink_and_fallback():
    bad, ok = table_masks([["n0", "n3"], []], NS)
    lg = torch.zeros(2, 2, V); lg[:, :, 11] = 5; lg[:, :, 13] = 2; lg[:, :, 7] = 1   # argmax n1 (not in table)
    out, _ = choose(lg, "slot", bad, ok)
    assert (out[0] == 13).all()                      # row with table: best allowed name (n3), not the plain token 7
    assert (out[1] == 7).all()                       # empty table -> masked pick = best non-name
    lg[:, :, 11] = 0; lg[:, :, 7] = 9                # argmax is a normal token -> untouched
    assert (choose(lg, "slot", bad, ok)[0] == 7).all()

def test_mask_logits_2d_and_3d():
    bad, _ = table_masks([[]], NS)
    assert torch.isinf(mask_logits(torch.zeros(1, V), bad)[0, 10]) and torch.isinf(mask_logits(torch.zeros(1, 2, V), bad)[0, 1, 17])

def test_window_slicing_and_prefix_layout():
    h = list(range(100, 120))
    assert window_tail(h, 5) == [115, 116, 117, 118, 119] and window_tail(h, 0) == [] and window_tail(h[:3], 5) == [100, 101, 102]
    p = build_prefix(h, ["n1", "n0"], NS, 6, 2, 1, 127)
    assert p == [1, 11, 10, 116, 117, 118, 119] and len(p) == 7
    p = build_prefix([5, 6], ["n1"], NS, 6, 3, 1, 127)       # short history: table + tail padded to a fixed length
    assert p == [1, 11, 127, 127, 127, 5, 6] and len(p) == 7
    assert build_prefix([5, 6], ["n1"], NS, 6, 3, 1) == [1, 11, 5, 6]                    # AR: unpadded

def _setup(arm, tmp_path, **kw):
    c = tiny_cfg(arm, "x", tmp_path, "data.prefix=8", *[f"{k}={v}" for k, v in kw.items()])
    torch.manual_seed(0); cod = build_codec(c) if arm == "ncm" else None
    return c, build_model(c, cod).eval(), cod

PROMPT = [[5, 10, 11, 3, 4, 5, 6, 7, 8, 9, 20], [60, 61, 62, 63, 64]]   # second one is shorter than the window

@pytest.mark.parametrize("arm", ["ar", "ncm"])
def test_generate_deterministic_and_shapes(arm, tmp_path):
    c, net, cod = _setup(arm, tmp_path); g = GenCfg(n_blocks=3, seed=7)
    a = generate_long(net, cod, c, PROMPT, NS, g); b = generate_long(net, cod, c, PROMPT, NS, g)
    assert a["blocks"] == b["blocks"]
    assert all(len(r) == 3 and all(len(x) <= 16 for x in r) for r in a["blocks"])
    assert generate_long(net, cod, c, PROMPT, NS, GenCfg(n_blocks=3, seed=8))["blocks"] != a["blocks"]
    assert a["table0"][0] == ["n0", "n1"] and a["table0"][1] == []

@pytest.mark.parametrize("arm", ["ar", "ncm"])
@pytest.mark.parametrize("mode", ["mask", "slot"])
def test_safe_modes_forbid_new_names(arm, mode, tmp_path):
    c, net, cod = _setup(arm, tmp_path); ns = synthetic_nameset(V, n=60)   # many names -> 'none' invents some
    r0 = generate_long(net, cod, c, PROMPT, ns, GenCfg(n_blocks=3, seed=1, mode="none"))
    r = generate_long(net, cod, c, PROMPT, ns, GenCfg(n_blocks=3, seed=1, mode=mode, table_slots=2))
    assert sum(ev.invented(t, b, ns)[0] for t, b in zip(r0["table0"], r0["blocks"])) > 0
    assert all(ev.invented(t, b, ns)[0] == 0 for t, b in zip(r["table0"], r["blocks"]))
    assert r["tables"] == r["table0"]                                       # table cannot grow under the mask

def test_metrics_pure():
    assert ev.invented(["n0"], [[10, 12], [12, 10]], NS) == (1, 4)           # n2 invented once, then known
    assert ev.carry_over(["n0", "n1"], [[10], [3]], NS) == (1, 2)
    w = lambda ids: [str(t) for t in ids]
    assert ev.seam_doubling([1, 2], [[3, 4], [4, 5], [6]], w) == (1, 3)

def test_eval_end_to_end_json(tmp_path):
    c, net, cod = _setup("ncm", tmp_path); ref_c, ref, _ = _setup("ar", tmp_path)
    o, _ = ev.evaluate(net, cod, c, PROMPT, NS, GenCfg(n_blocks=2, mode="mask"), ref, ref_c)
    json.dumps(o); assert o["invented_rate"] == 0 and o["ppl_ref"] > 1 and 0 <= o["seam_doubling"] <= 1

def test_load_from_checkpoint_matches_train(tiny_data, tmp_path):
    from ncm150.train import train
    c = tiny_cfg("ncm", tiny_data, tmp_path, "data.prefix=0"); train(c)
    cfg, net, cod = load_model(str(tmp_path / "last.pt"))
    r = generate_long(net, cod, cfg, [[1, 2, 3]], NS, GenCfg(n_blocks=1)); assert len(r["blocks"][0][0]) <= 16
