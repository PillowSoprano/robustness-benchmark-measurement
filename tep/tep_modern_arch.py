"""Two modern forecasting architectures for the TEP task.

Both are adaptations, not reimplementations. This task carries 33 input
channels (22 measurements plus 11 manipulated variables), 22 output channels,
and a 20-sample history, so the published channel-independent forms do not
apply unchanged and the names below say "style".

DecompLinear follows DLinear: split the history into a moving-average trend
and a remainder, map each with its own linear layer over time, then mix
channels down to the 22 outputs. The point of including it is that a linear
model with the right inductive bias competes with deep sequence models on
long-horizon forecasting, so it belongs in a panel that claims architectural
coverage.

PatchTransformer follows PatchTST: cut the history into time patches, embed
each patch jointly across channels, attend over the patch tokens, and read
out the whole horizon. With a 20-sample history there are only four patches,
which is the honest limit of what attention can do on this task.
"""

import torch
import torch.nn as nn


class DecompLinear(nn.Module):
    """DLinear-style decomposition linear, 33 channels in, 22 out."""

    def __init__(self, hist, horizon, n_in=33, n_out=22, kernel=5):
        super().__init__()
        self.hist, self.horizon = hist, horizon
        self.n_in, self.n_out = n_in, n_out
        self.kernel = kernel
        self.trend = nn.Linear(hist, horizon)
        self.season = nn.Linear(hist, horizon)
        self.mix = nn.Linear(n_in, n_out)

    def forward(self, x):
        n = x.shape[0]
        seq = x.reshape(n, self.hist, self.n_in).transpose(1, 2)
        pad = self.kernel // 2
        padded = nn.functional.pad(seq, (pad, pad), mode="replicate")
        trend = nn.functional.avg_pool1d(padded, self.kernel, stride=1)
        trend = trend[:, :, :self.hist]
        out = self.trend(trend) + self.season(seq - trend)
        out = self.mix(out.transpose(1, 2))
        return out.reshape(n, self.horizon * self.n_out)


class PatchTransformer(nn.Module):
    """PatchTST-style patch attention, channels embedded jointly."""

    def __init__(self, hist, horizon, n_in=33, n_out=22, patch=5,
                 d_model=128, heads=4, layers=2, ff=256, drop=0.1):
        super().__init__()
        assert hist % patch == 0, "history must divide into whole patches"
        self.hist, self.horizon = hist, horizon
        self.n_in, self.n_out, self.patch = n_in, n_out, patch
        n_patch = hist // patch
        self.embed = nn.Linear(patch * n_in, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patch, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, heads, ff, drop,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Dropout(drop),
                                  nn.Linear(n_patch * d_model,
                                            horizon * n_out))

    def forward(self, x):
        n = x.shape[0]
        seq = x.reshape(n, self.hist, self.n_in)
        tok = seq.reshape(n, self.hist // self.patch, self.patch * self.n_in)
        h = self.enc(self.embed(tok) + self.pos)
        return self.head(h)
