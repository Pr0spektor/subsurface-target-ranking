"""
Potential-field processing (FFT-domain) + regional removal. Pure NumPy.

Conventions: grid [row=North, col=East]; component order [East, North, Down].
"""
from __future__ import annotations
import numpy as np


def _wavenumbers(shape, dx):
    ny, nx = shape
    kN = 2 * np.pi * np.fft.fftfreq(ny, d=dx)   # north (axis0)
    kE = 2 * np.pi * np.fft.fftfreq(nx, d=dx)   # east  (axis1)
    KE, KN = np.meshgrid(kE, kN)
    K = np.sqrt(KE**2 + KN**2)
    return KE, KN, K


def _apply(field, operator, dx, pad=20):
    f = np.pad(field - field.mean(), pad, mode="reflect")
    F = np.fft.fft2(f)
    out = np.real(np.fft.ifft2(F * operator(_wavenumbers(f.shape, dx))))
    return out[pad:-pad, pad:-pad]


def reduction_to_pole(tmi, dx, inc_deg, dec_deg):
    i, d = np.radians(inc_deg), np.radians(dec_deg)
    aE, aN, aZ = np.cos(i) * np.sin(d), np.cos(i) * np.cos(d), np.sin(i)
    def op(w):
        KE, KN, K = w
        K = np.where(K == 0, 1e-12, K)
        theta = aZ * K + 1j * (aE * KE + aN * KN)   # induced: field dir == magnetisation dir
        theta = np.where(np.abs(theta) < 1e-9, 1e-9, theta)
        return (K**2) / (theta * theta)
    return _apply(tmi, op, dx)


def vertical_derivative(field, dx, order=1):
    return _apply(field, lambda w: w[2]**order, dx)


def upward_continue(field, dx, height):
    return _apply(field, lambda w: np.exp(-w[2] * height), dx)


def analytic_signal(field, dx):
    gN, gE = np.gradient(field, dx)            # axis0=North, axis1=East
    gz = vertical_derivative(field, dx, 1)
    return np.sqrt(gE**2 + gN**2 + gz**2)


def tilt_angle(field, dx):
    gN, gE = np.gradient(field, dx)
    gz = vertical_derivative(field, dx, 1)
    return np.arctan2(gz, np.sqrt(gE**2 + gN**2) + 1e-12)


def polynomial_detrend(field, E, N, order=1):
    """Fit & remove a low-order 2-D polynomial regional field (planar or quadratic)."""
    e, n = E.ravel(), N.ravel()
    cols = [np.ones_like(e), e, n]
    if order >= 2:
        cols += [e * n, e**2, n**2]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, field.ravel(), rcond=None)
    regional = (A @ coef).reshape(field.shape)
    return field - regional, regional
