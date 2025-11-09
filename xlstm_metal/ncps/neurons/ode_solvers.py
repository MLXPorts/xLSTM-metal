"""Simple ODE solvers implemented with MLX operations."""

from __future__ import annotations

import mlx.core as mx


def euler_solve(f, y0, dt):
    """Explicit Euler step."""
    return mx.add(y0, mx.multiply(dt, f(None, y0)))


def rk4_solve(f, y0, t0, dt):
    """Fourth-order Runge–Kutta integration."""
    k1 = f(t0, y0)
    k2 = f(mx.add(t0, mx.divide(dt, 2)), mx.add(y0, mx.divide(mx.multiply(dt, k1), 2)))
    k3 = f(mx.add(t0, mx.divide(dt, 2)), mx.add(y0, mx.divide(mx.multiply(dt, k2), 2)))
    k4 = f(mx.add(t0, dt), mx.add(y0, mx.multiply(dt, k3)))
    return mx.add(
        y0,
        mx.divide(
            mx.multiply(
                dt,
                mx.add(
                    mx.add(k1, mx.multiply(2, k2)), mx.add(mx.multiply(2, k3), k4)
                ),
            ),
            6,
        ),
    )


def semi_implicit_solve(f, y0, dt):
    """Semi-implicit Euler step (Heun's method)."""
    k1 = f(None, y0)
    y_pred = mx.add(y0, mx.multiply(dt, k1))
    k2 = f(None, y_pred)
    return mx.add(y0, mx.divide(mx.multiply(dt, mx.add(k1, k2)), 2))
