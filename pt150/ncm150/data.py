"""Pre-tokenized uint16 GPT-2 shards -> deterministic window stream. Sample j of the stream is a pure function of
(seed, j): identical for every arm, and exact resume only needs the sample counter."""
import glob, os
import numpy as np, torch

class WindowStream:
    def __init__(self, data_dir, window, seed, split="train"):
        pat = "val.bin" if split == "val" else "train_*.bin"
        self.files = sorted(glob.glob(os.path.join(data_dir, pat)))
        assert self.files, f"no {pat} in {data_dir} (run prep_data.py)"
        self.mm = [np.memmap(f, dtype=np.uint16, mode="r") for f in self.files]
        self.window, self.seed, self.split = window, seed, split
        self.sizes = [len(m) // window for m in self.mm]
        self.n = sum(self.sizes); self.cum = np.cumsum([0] + self.sizes)
        self._perm = (None, None)

    def _perm_for(self, epoch):
        if self._perm[0] != epoch:
            self._perm = (epoch, np.random.default_rng([self.seed, epoch]).permutation(self.n))
        return self._perm[1]

    def window_id(self, j):
        if self.split == "val": return j % self.n
        e, k = divmod(j, self.n); return int(self._perm_for(e)[k])

    def get(self, wid):
        s = int(np.searchsorted(self.cum, wid, side="right") - 1); o = (wid - self.cum[s]) * self.window
        return np.asarray(self.mm[s][o:o + self.window], dtype=np.int64)

    def batch(self, j0, B):
        return torch.from_numpy(np.stack([self.get(self.window_id(j0 + i)) for i in range(B)]))
