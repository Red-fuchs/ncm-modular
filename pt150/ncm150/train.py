import argparse, contextlib, json, math, os, random, time
import numpy as np, torch, torch.nn.functional as F
from omegaconf import OmegaConf
from ncm150.config import load_cfg, levels_of
from ncm150.data import WindowStream
from ncm150.flops import plan_steps, train_flops_per_window
from ncm150.models import blocks_of, build_codec, build_model, enable_grad_ckpt, ncm_nonembedding_params

def pick_device(d):
    if d != "auto": return torch.device(d)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def lr_at(step, total, tc):
    wu = max(1, int(tc.warmup_frac * total)); lo = tc.min_lr_ratio
    if step < wu: return tc.lr * (step + 1) / wu
    p = (step - wu) / max(1, total - wu)
    if tc.schedule == "cosine": return tc.lr * (lo + (1 - lo) * 0.5 * (1 + math.cos(math.pi * p)))
    d0 = 1 - tc.wsd_decay_frac  # wsd: flat, then linear decay to lo*lr
    return tc.lr if p < d0 else tc.lr * (1 - (1 - lo) * (p - d0) / tc.wsd_decay_frac)

class Task:
    """Arm-specific loss. loss(toks) -> (scalar loss, dict of extra per-batch means)."""
    def __init__(self, cfg, dev):
        self.cfg, self.dev, self.arm = cfg, dev, cfg.arm
        self.codec = build_codec(cfg).to(dev) if self.arm == "ncm" else None
        self.net = build_model(cfg, self.codec).to(dev)
        if self.arm == "ncm":
            L = levels_of(cfg); T = sum(L[1:]); self.L0 = L[0]
            w = torch.ones(T)
            if cfg.ncm.level_weighting == "per_level_mean":
                w = torch.cat([torch.full((l,), 1.0 / l) for l in L[1:]])
            self.w = (w / w.sum()).to(dev)
            self.nonemb = ncm_nonembedding_params(self.net)
        else:
            self.nonemb = self.net.nonembedding_params()
        if cfg.train.grad_ckpt: enable_grad_ckpt(blocks_of(self.net))

    def loss(self, toks):
        cfg = self.cfg; toks = toks.to(self.dev)
        if self.arm == "ar":
            inp = torch.cat([torch.full_like(toks[:, :1], cfg.data.eos_id), toks[:, :-1]], 1)
            logits = self.net(inp)
            ce = F.cross_entropy(logits.float().flatten(0, 1), toks.flatten())
            return ce, {}
        with torch.no_grad(), torch.autocast(self.dev.type, enabled=False):
            idx = self.codec.encode(toks)
            x = torch.cat([self.codec.encoder.wte(torch.full_like(toks[:, :1], cfg.data.eos_id)),
                           self.codec.vq.idx_to_ncm_input(idx)], 1)
        logits = self.net(x)
        ce = F.cross_entropy(logits.float().flatten(0, 1), idx[:, self.L0:].flatten(), reduction="none").view(toks.shape[0], -1)
        # code CE summed over all codes of a window, per text token: comparable scale to AR nats/token (not identical quantity)
        return (ce * self.w).sum(1).mean(), {"code_nats_per_text_token": (ce.sum(1) / toks.shape[1]).mean().item()}

def save_ckpt(path, task, opt, step, cfg, keep):
    st = dict(model=task.net.state_dict(), opt=opt.state_dict(), step=step, cfg=OmegaConf.to_yaml(cfg),
              rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       py=random.getstate(), np=np.random.get_state()))
    tmp = path + ".tmp"; torch.save(st, tmp); os.replace(tmp, path)

def load_ckpt(path, task, opt):
    st = torch.load(path, map_location="cpu", weights_only=False)
    task.net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
    torch.set_rng_state(st["rng"]["cpu"])
    if st["rng"]["cuda"] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(st["rng"]["cuda"])
    random.setstate(st["rng"]["py"]); np.random.set_state(st["rng"]["np"])
    return st["step"]

@torch.no_grad()
def evaluate(task, val, n, mbs, ac):
    task.net.eval(); tot, ex, cnt = 0.0, {}, 0
    for j in range(0, n, mbs):
        with ac(): l, e = task.loss(val.batch(j, min(mbs, n - j)))
        b = min(mbs, n - j); tot += l.item() * b; cnt += b
        for k, v in e.items(): ex[k] = ex.get(k, 0.0) + v * b
    task.net.train()
    return dict(val_loss=tot / cnt, **{"val_" + k: v / cnt for k, v in ex.items()})

def train(cfg, resume=True, stop_after=None):
    dev = pick_device(cfg.device); out = cfg.out_dir; os.makedirs(out, exist_ok=True)
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    tc = cfg.train; task = Task(cfg, dev); net = task.net; net.train()
    fpw = train_flops_per_window(cfg, task.codec)
    total, fps, tps = plan_steps(cfg, fpw)
    gbs, mbs = int(tc.global_bs), int(tc.micro_bs); assert gbs % mbs == 0; accum = gbs // mbs
    stream = WindowStream(cfg.data.dir, cfg.data.window, cfg.seed, "train")
    val = WindowStream(cfg.data.dir, cfg.data.window, cfg.seed, "val")
    dec = [p for p in net.parameters() if p.requires_grad and p.ndim >= 2]; nod = [p for p in net.parameters() if p.requires_grad and p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=dec, weight_decay=tc.weight_decay), dict(params=nod, weight_decay=0.0)], lr=tc.lr,
                            betas=tuple(tc.betas), fused=dev.type == "cuda")
    ac = (lambda: torch.autocast(dev.type, dtype=torch.bfloat16)) if (cfg.precision == "bf16" and dev.type == "cuda") else contextlib.nullcontext
    fwd = torch.compile(task.net) if tc.compile else None
    step, ck = 0, os.path.join(out, "last.pt"); logp = os.path.join(out, "log.jsonl")
    if resume and os.path.exists(ck):
        step = load_ckpt(ck, task, opt)
        if os.path.exists(logp):  # drop log lines from after the checkpoint (they will be re-run)
            keep = [l for l in open(logp) if json.loads(l).get("step", 0) <= step]; open(logp, "w").writelines(keep)
    log = open(logp, "a")
    def emit(**r): log.write(json.dumps(r) + "\n"); log.flush()
    if step == 0:
        emit(event="start", arm=cfg.arm, nonembedding_params=task.nonemb, total_params=sum(p.numel() for p in net.parameters()),
             flops_per_window=fpw, flops_per_step=fps, text_tokens_per_step=tps, planned_steps=total,
             planned_flops=fps * total, planned_text_tokens=tps * total, cfg=OmegaConf.to_container(cfg))
    else: emit(event="resume", step=step)
    wb = None
    if cfg.wandb.enabled:
        import wandb; wb = wandb.init(project=cfg.wandb.project, name=f"{cfg.arm}", config=OmegaConf.to_container(cfg), resume="allow")
    t0 = time.time(); stop = total if stop_after is None else min(total, stop_after)
    while step < stop:
        for g in opt.param_groups: g["lr"] = lr_at(step, total, tc)
        ts = time.time(); acc, ex = 0.0, {}
        for a in range(accum):
            toks = stream.batch(step * gbs + a * mbs, mbs)
            with ac(): l, e = task.loss(toks)
            (l / accum).backward(); acc += l.item() / accum
            for k, v in e.items(): ex[k] = ex.get(k, 0.0) + v / accum
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), tc.grad_clip).item()
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if not math.isfinite(acc): raise FloatingPointError(f"non-finite loss at step {step}")
        dt = time.time() - ts
        if step % tc.log_every == 0 or step == total:
            r = dict(step=step, loss=acc, lr=opt.param_groups[0]["lr"], grad_norm=gn, text_tokens=step * tps,
                     flops=step * fps, step_s=dt, text_tok_s=tps / dt, tflops_s=fps / dt / 1e12, **ex)
            if dev.type == "cuda": r["mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            emit(**r); wb and wb.log(r, step=step)
        if step % tc.eval_every == 0 or step == total:
            r = dict(step=step, event="eval", **evaluate(task, val, min(tc.eval_windows, val.n), mbs, ac)); emit(**r); wb and wb.log(r, step=step)
        if step % tc.ckpt_every == 0 or step == total or step == stop: save_ckpt(ck, task, opt, step, cfg, tc.keep_ckpts)
        if tc.max_minutes and (time.time() - t0) / 60 > tc.max_minutes:
            save_ckpt(ck, task, opt, step, cfg, tc.keep_ckpts); emit(event="time_cap", step=step); break
    log.close()
    return dict(step=step, total=total, model=net)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True); ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--stop-after", type=int); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    r = train(load_cfg(*a.config, overrides=a.set), resume=not a.no_resume, stop_after=a.stop_after)
    print(json.dumps(dict(step=r["step"], total=r["total"])))

if __name__ == "__main__": main()
