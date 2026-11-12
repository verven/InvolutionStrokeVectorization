# Taken from Kinross (https://gitlab.com/parclytaxel/Kinross).
# Copyright (c) 2018 Parcly Taxel / Jeremy Tan
# Licensed under the MIT License; see LICENSE-KINROSS in this directory.

# Algebraic and numerical helper functions.
import numpy as np

cc_abscissas = [(np.cos(np.arange(1, 2**k, 2) / 2**k * np.pi) + 1) / 2 for k in range(2, 11)]


def ccw_generate(n):
    """Clenshaw-Curtis weights for n+1 samples where n is a power of two.
    DFT-based algorithm from Jörg Waldvogel
    (http://www.sam.math.ethz.ch/~waldvoge/Papers/fejer.pdf)."""
    w0 = 1 / (n**2 - 1)
    v = [2 / (1 - 4 * k**2) - w0 for k in range(n // 2)]
    dft = np.fft.rfft(v + [-3 * w0] + v[:0:-1]).real / n
    # This ensures the weight array is symmetric as it should be
    return np.append(dft, dft[-2::-1])


cc_weights = [ccw_generate(2**k) for k in range(2, 11)]


def ccquad(f, a, b):
    """Clenshaw-Curtis quadrature of f in [a, b].
    f must be applicable elementwise to NumPy arrays."""
    fs = [f(a), f((a + b) / 2), f(b)]
    res = (fs[0] + 4 * fs[1] + fs[2]) / 3
    for q in range(9):
        fs = np.insert(fs, range(1, len(fs)), f((b - a) * cc_abscissas[q] + a))
        prev, res = res, np.dot(cc_weights[q], fs)
        if abs(res - prev) <= 1e-12 * abs(prev):
            break
    return (b - a) / 2 * res


def fproot(f, a, b):
    """Given that a root of f is bracketed in (a, b), find that root
    using false position with the Anderson–Björk multiplier."""
    fa, fb = f(a), f(b)
    while abs(b - a) > 1e-14:
        z = (a * fb - b * fa) / (fb - fa)
        fz = f(z)
        if abs(fz) <= 1e-14:
            return z
        if fb * fz < 0:
            a, fa = b, fb
        else:
            m = fz / fb
            fa *= 1 - m if m < 1 else 0.5
        b, fb = z, fz
    return (a + b) / 2
