"""
Sequence Precompute vs Per-Step Dispatch (MLX)

Demonstrates reducing kernel dispatches by precomputing linear projections
over the full sequence, then scanning the recurrent state in Python, vs
calling multiple nn.Linear ops per step.
"""

import time

import mlx.core as mx
import mlx.nn as nn


class TinySLSTM(nn.Module):
    def __init__(self, d_model=256, proj_factor=4/3):
        super().__init__()
        hidden = d_model
        proj = int(proj_factor * hidden)
        self.W_i = nn.Linear(d_model, hidden)
        self.W_f = nn.Linear(d_model, hidden)
        self.W_z = nn.Linear(d_model, hidden)
        self.W_o = nn.Linear(d_model, hidden)
        self.up = nn.Linear(hidden, 2*proj)
        self.down = nn.Linear(proj, d_model)

    def step(self, x, state):
        """

        :param x:
        :param state:
        :return:
        """
        c, n, h, m = state
        i = self.W_i(x); f = self.W_f(x)
        z = self.W_z(x); o = self.W_o(x)
        m = mx.maximum(f + m, i)
        i = mx.exp(i - m); f = mx.exp(f - m + m)
        z = mx.tanh(z); o = mx.sigmoid(o)
        c = f * c + i * z
        n = f * n + i
        h = o * (c / mx.maximum(n, mx.array(1.0)))
        y1, y2 = mx.split(self.up(h), 2, axis=-1)
        y = self.down(y1 * nn.gelu(y2))
        return y, (c, n, h, m)

    def forward_stepwise(self, x):
        """

        :param x:
        :return:
        """
        B, S, D = x.shape
        c = mx.zeros((B, D)); n = mx.ones((B, D)); h = mx.zeros((B, D)); m = mx.zeros((B, D))
        outs = []
        for t in range(S):
            y, (c, n, h, m) = self.step(x[:, t, :], (c, n, h, m))
            outs.append(y)
        return mx.stack(outs, axis=1)

    def forward_precompute(self, x):
        """

        :param x:
        :return:
        """
        B, S, D = x.shape
        # Precompute all projections once
        I = self.W_i(x); F = self.W_f(x); Z = self.W_z(x); O = self.W_o(x)
        c = mx.zeros((B, D)); n = mx.ones((B, D)); h = mx.zeros((B, D)); m = mx.zeros((B, D))
        outs = []
        for t in range(S):
            i = I[:, t, :]; f = F[:, t, :]; z = Z[:, t, :]; o = O[:, t, :]
            m = mx.maximum(f + m, i)
            i = mx.exp(i - m); f = mx.exp(f - m + m)
            z = mx.tanh(z); o = mx.sigmoid(o)
            c = f * c + i * z
            n = f * n + i
            h = o * (c / mx.maximum(n, mx.array(1.0)))
            y1, y2 = mx.split(self.up(h), 2, axis=-1)
            y = self.down(y1 * nn.gelu(y2))
            outs.append(y)
        return mx.stack(outs, axis=1)


def bench(model, x):
    """

    :param model:
    :param x:
    :return:
    """
    # Warmup
    _ = model.forward_stepwise(x); mx.eval(_)
    _ = model.forward_precompute(x); mx.eval(_)
    iters = 5
    t0 = time.time()
    for _ in range(iters):
        y = model.forward_stepwise(x); mx.eval(y)
    t_step = (time.time() - t0) / iters
    t1 = time.time()
    for _ in range(iters):
        y = model.forward_precompute(x); mx.eval(y)
    t_prec = (time.time() - t1) / iters
    return t_step, t_prec


if __name__ == "__main__":
    B, S, D = 4, 128, 256
    mx.random.seed(0)
    x = mx.random.normal((B, S, D))
    model = TinySLSTM(D)
    t_step, t_prec = bench(model, x)
    print(f"Stepwise:    {t_step*1e3:.2f} ms  (per-step Linear ops)")
    print(f"Precompute:  {t_prec*1e3:.2f} ms  (sequence-wide Linear then scan)")

