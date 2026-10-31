"""tokenizer.pt (+config.yaml) -> w/mlx/<task>/{codec.safetensors,spec.json}."""
import json, os, sys
import numpy as np, torch, yaml
from safetensors.numpy import save_file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK
from ref_torch import load_state
from codec_mlx import cascade_ops

task = sys.argv[1]
src, dst = f"{WORK}/up/{task}", f"{WORK}/mlx/{task}"
os.makedirs(dst, exist_ok=True)
cfg = yaml.safe_load(open(f"{src}/config.yaml"))
v = cfg["tokenizer"]["vqvae"]; vq = v["vector_quantizer_config"]; pt = cfg["tokenizer"]["pre_tokenizer"]
sd = {k.removeprefix("module.").removeprefix("_orig_mod."): t.float().numpy() for k, t in load_state(f"{src}/tokenizer.pt").items()}
levels = vq["levels"]["manual_levels"]
assert vq["levels"]["use_manual_levels"] and vq["quant_resi"]["share_mode"] == 0 and vq["quant_resi"]["enabled"]
assert not vq["quant_resi"]["learnable_ratio"] and v["compression_factor"] == 1 and vq["learned_l1_cascade"] and not vq["learned_all_sampling"]
assert v["pre_quant_groupnorm"] and v.get("pre_quant_norm_type", "group") == "group"
dec_layers = 1 + len({k.split(".")[2] for k in sd if k.startswith("decoder.extra_blocks.")})
spec = dict(levels=levels, groupnorm=v["pre_quant_groupnorm"], n_head=v["n_layers"], phi_ratio=vq["quant_resi"]["ratio"],
            cascade_base=vq["learned_l1_cascade_base"], cascade_norm=vq["learned_l1_cascade_norm"],
            cascade_norm_final=vq["learned_l1_cascade_norm_final"], dec_layers=dec_layers,
            vocab=v["in_vocab_size"], seq_len=v["context_length"], dim=v["embed_dim"], codebook=v["vqvae_vocab_size"],
            model_id=pt["model_id"], eos_token=pt["special_tokens"]["eos_token"], pad_token=pt["special_tokens"]["pad_token"],
            prefix_pad_token_id=cfg["model"]["prefix_pad_token_id"])
W, used = {}, set()
def take(name, key, f=lambda a: a):
    W[name] = np.ascontiguousarray(f(sd[key])); used.add(key)
take("wte", "encoder.wte.weight"); take("wpe", "encoder.wpe.weight")
for i in range(len(levels)): take(f"cb.{i}", f"vq.embeddings.{i}.weight")
for j in range(len(levels)):
    take(f"phi.{j}.w", f"vq.quant_resi.{j}.weight", lambda a: a.transpose(0, 2, 1)); take(f"phi.{j}.b", f"vq.quant_resi.{j}.bias")
down, up = cascade_ops(levels[-1] // levels[0], spec["cascade_base"], spec["cascade_norm"], spec["cascade_norm_final"])
for op in down:
    if op[0] == "d":  # Conv1d (Co,Ci,s) -> (s*Ci, Co)
        s, i = op[1], op[2]
        take(f"down.{i}.w", f"vq.learned_down_l1.{i}.weight", lambda a: a.transpose(2, 1, 0).reshape(-1, a.shape[0])); take(f"down.{i}.b", f"vq.learned_down_l1.{i}.bias")
for op in up:
    if op[0] == "u":  # ConvT (Ci,Co,s) -> (Ci, s*Co)
        s, i = op[1], op[2]
        take(f"up.{i}.w", f"vq.learned_up_l1.{i}.weight", lambda a: a.transpose(0, 2, 1).reshape(a.shape[0], -1)); take(f"up.{i}.b", f"vq.learned_up_l1.{i}.bias")
for n in range(dec_layers):
    u = "decoder.middle_block" if n == 0 else f"decoder.extra_blocks.{n-1}"
    for a, b in [("ln1.w", "ln_1.weight"), ("ln1.b", "ln_1.bias"), ("attn.w", "attn.c_attn.weight"), ("attn.b", "attn.c_attn.bias"),
                 ("qn", "attn.q_norm.weight"), ("kn", "attn.k_norm.weight"), ("proj.w", "attn.c_proj.weight"), ("proj.b", "attn.c_proj.bias"),
                 ("ln2.w", "ln_2.weight"), ("ln2.b", "ln_2.bias"), ("fc.w", "mlp.c_fc.weight"), ("fc.b", "mlp.c_fc.bias"),
                 ("fcp.w", "mlp.c_proj.weight"), ("fcp.b", "mlp.c_proj.bias")]:
        take(f"blk{n}.{a}", f"{u}.{b}")
take("lnf.w", "decoder.ln_f.weight"); W["lnf.b"] = np.zeros_like(sd["decoder.ln_f.weight"])  # LayerNorm bias=False upstream
take("head", "decoder.output_layer.weight")
left = [k for k in sd if k not in used and not k.startswith(("vq.ema_", "vq.codebook_hit", "vq._"))]
print("unused keys:", left)
save_file(W, f"{dst}/codec.safetensors"); json.dump(spec, open(f"{dst}/spec.json", "w"), indent=1)
print(task, "ok", len(W), "tensors", sum(a.size for a in W.values()) / 1e6, "M params")
