import json, os
import numpy as np, pytest, torch
from omegaconf import OmegaConf
from ncm150 import flops as fl
from ncm150.config import load_cfg, levels_of
from ncm150.data import WindowStream
from ncm150.models import build_codec, build_model, ncm_nonembedding_params
from ncm150.train import train, lr_at
from conftest import tiny_cfg

def test_shapes(tiny_data, tmp_path):
    c = tiny_cfg("ar", tiny_data, tmp_path); ar = build_model(c).eval()
    assert ar(torch.randint(0, 128, (3, 16))).shape == (3, 16, 128)
    c = tiny_cfg("ncm", tiny_data, tmp_path); cod = build_codec(c); L = levels_of(c); net = build_model(c, cod).eval()
    idx = cod.encode(torch.randint(0, 128, (3, 16))); assert idx.shape == (3, sum(L))
    x = torch.cat([cod.encoder.wte(torch.ones(3, 1, dtype=torch.long)), cod.vq.idx_to_ncm_input(idx)], 1)
    out = net(x); assert out.shape == (3, sum(L[1:]), c.codec.vocab)

def test_bool_mask_equals_float_mask(tiny_data, tmp_path):
    c = tiny_cfg("ncm", tiny_data, tmp_path, "model.dropout=0.0"); cod = build_codec(c); net = build_model(c, cod).eval()
    x = torch.randn(2, 1 + sum(levels_of(c)[1:]), c.codec.embed_dim)
    a = net(x); m = net.attn_mask_for_masking
    net.attn_mask_for_masking = torch.where(m, 0.0, -torch.inf)
    assert torch.allclose(a, net(x), atol=1e-5)

def test_ar_causal(tiny_data, tmp_path):
    c = tiny_cfg("ar", tiny_data, tmp_path, "model.dropout=0.0"); ar = build_model(c).eval()
    a = torch.randint(0, 128, (1, 16)); b = a.clone(); b[0, 10:] = 5
    assert torch.allclose(ar(a)[0, :10], ar(b)[0, :10], atol=1e-5)

def test_tiny_param_count_matches_formula(tiny_data, tmp_path):
    m = tiny_cfg("ar", tiny_data, tmp_path).model; d, nl = m.n_embd, m.n_layer
    ar = build_model(tiny_cfg("ar", tiny_data, tmp_path))
    blocks = sum(p.numel() for p in ar.blocks.parameters())
    assert abs(blocks - nl * 12 * d * d) / blocks < 0.25  # 12d^2 per block + biases/qk-norm/LN
    assert ar.nonembedding_params() == sum(p.numel() for p in ar.parameters()) - ar.wte.weight.numel()

@pytest.mark.slow
def test_full_size_params_near_150m():
    base = ["device=cpu"]
    ar = build_model(load_cfg("ar_150m.yaml", overrides=base))
    n_ar = ar.nonembedding_params(); assert abs(n_ar / 150e6 - 1) < 0.05, n_ar
    c = load_cfg("ncm_150m.yaml", overrides=base + ["codec.random_init=true", "ncm.flex_attention=false"])
    n_ncm = ncm_nonembedding_params(build_model(c, build_codec(c))); assert abs(n_ncm / 150e6 - 1) < 0.05, n_ncm

def _pairs_from_mask(net): return int(net.attn_mask_for_masking.sum())

@pytest.mark.parametrize("arm", ["ar", "ncm"])
def test_flops_formula_matches_counter(arm, tiny_data, tmp_path):
    c = tiny_cfg(arm, tiny_data, tmp_path, "model.dropout=0.0"); m = c.model; d, nl = m.n_embd, m.n_layer
    if arm == "ar":
        net = build_model(c).eval(); x = torch.randint(0, 128, (1, 16)); T = 16
        meas = fl.measure_fwd(lambda: net(x))
        assert meas == fl.ar_fwd_flops(m, 128, T, dense_attn=True) - 4 * nl * d * T * T  # CPU counter skips SDPA
        assert fl.ar_pairs(T) == T * (T + 1) // 2
    else:
        cod = build_codec(c); net = build_model(c, cod).eval(); L = levels_of(c); S = 1 + sum(L[1:])
        x = torch.randn(1, S, c.codec.embed_dim); meas = fl.measure_fwd(lambda: net(x))
        assert meas == fl.ncm_fwd_flops(m, L, c.codec.embed_dim, c.codec.vocab, 1, True, dense_attn=True) - 4 * nl * d * S * S
        assert fl.ncm_pairs(1, L) == _pairs_from_mask(net)
        assert fl.ncm_pairs(1, L) < S * S

def test_train_flops_and_plan(tiny_data, tmp_path):
    ca, cn = tiny_cfg("ar", tiny_data, tmp_path), tiny_cfg("ncm", tiny_data, tmp_path)
    fa = fl.train_flops_per_window(ca); fn = fl.train_flops_per_window(cn, build_codec(cn))
    assert fa == 3 * fl.ar_fwd_flops(ca.model, 128, 16) and fn > 3 * fl.ncm_fwd_flops(cn.model, levels_of(cn), 16, 32)  # + codec
    ca.budget.mode = "flops"; ca.budget.flops = fa * ca.train.global_bs * 7.5
    s, fps, tps = fl.plan_steps(ca, fa); assert (s, fps, tps) == (7, fa * 4, 64)
    cn.budget.mode = "flops"; cn.budget.flops = fa * ca.train.global_bs * 7.5      # same budget: costlier arm gets fewer steps
    assert fl.plan_steps(cn, fn)[0] < s

def test_data_order_identical_across_arms(tiny_data, tmp_path):
    s1 = WindowStream(tiny_data, 16, 1234); s2 = WindowStream(tiny_data, 16, 1234)
    assert s1.n == 46
    ids = [s1.window_id(j) for j in range(2 * s1.n)]
    assert ids == [s2.window_id(j) for j in range(2 * s1.n)]
    assert sorted(ids[:46]) == list(range(46)) and ids[:46] != ids[46:]      # epoch = full permutation, reshuffled
    assert torch.equal(s1.batch(5, 6), s2.batch(5, 6))
    assert not torch.equal(s1.batch(0, 4), WindowStream(tiny_data, 16, 99).batch(0, 4))
    # the first N samples both arms consume come from the same stream regardless of micro-batching
    assert torch.equal(torch.cat([s1.batch(0, 2), s1.batch(2, 2)]), s1.batch(0, 4))

@pytest.mark.parametrize("arm", ["ar", "ncm"])
def test_resume_determinism(arm, tiny_data, tmp_path):
    N, k = 8, 3
    ex = ["budget.tokens=%d" % (N * 4 * 16), "train.ckpt_every=1", "train.eval_every=4"]
    a = train(tiny_cfg(arm, tiny_data, tmp_path / "full", *ex), stop_after=None)
    assert a["step"] == N
    b1 = train(tiny_cfg(arm, tiny_data, tmp_path / "split", *ex), stop_after=k); assert b1["step"] == k
    torch.manual_seed(999)  # clobber global RNG: resume must restore it from the checkpoint
    b2 = train(tiny_cfg(arm, tiny_data, tmp_path / "split", *ex)); assert b2["step"] == N
    sa, sb = a["model"].state_dict(), b2["model"].state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[x], sb[x]) for x in sa)
    la = [json.loads(l) for l in open(tmp_path / "full" / "log.jsonl")]; lb = [json.loads(l) for l in open(tmp_path / "split" / "log.jsonl")]
    key = lambda L: [(r["step"], r.get("loss"), r.get("val_loss")) for r in L if "step" in r and r.get("event") != "resume"]
    assert key(la) == key(lb)

def test_lr_schedules():
    tc = OmegaConf.create(dict(lr=1.0, min_lr_ratio=0.1, warmup_frac=0.1, schedule="cosine", wsd_decay_frac=0.2))
    assert lr_at(0, 100, tc) < 1 and abs(lr_at(10, 100, tc) - 1) < 1e-6 and abs(lr_at(99, 100, tc) - 0.1) < 0.01
    tc.schedule = "wsd"; assert lr_at(50, 100, tc) == 1.0 and lr_at(99, 100, tc) < 0.2

def test_ffn_extension_point_guard(tiny_data, tmp_path):
    with pytest.raises(NotImplementedError): build_model(tiny_cfg("ar", tiny_data, tmp_path, "model.ffn=moe"))
