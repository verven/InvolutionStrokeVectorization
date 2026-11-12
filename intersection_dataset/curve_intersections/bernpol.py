# Taken from Kinross (https://gitlab.com/parclytaxel/Kinross).
# Copyright (c) 2018 Parcly Taxel / Jeremy Tan
# Licensed under the MIT License; see LICENSE-KINROSS in this directory.

# Methods dealing with polynomials expressed in the Bernstein basis,
# (1-t)^n term first. These usually achieve better numerical stability.

# "Sederberg" below refers to his book Computer-Aided Geometric Design,
# available at http://scholarsarchive.byu.edu/facpub/1
import numpy as np

from .algebra import fproot

bnm = [
    np.array([1]),
    np.array([1, 1]),
    np.array([1, 2, 1]),
    np.array([1, 3, 3, 1]),
    np.array([1, 4, 6, 4, 1]),
    np.array([1, 5, 10, 10, 5, 1]),
    np.array([1, 6, 15, 20, 15, 6, 1]),
    np.array([1, 7, 21, 35, 35, 21, 7, 1]),
    np.array([1, 8, 28, 56, 70, 56, 28, 8, 1]),
    np.array([1, 9, 36, 84, 126, 126, 84, 36, 9, 1]),
]


def berneval(p, t):
    """Evaluate the polynomial p at real parameters t using an adapted
    Horner scheme (Sederberg p. 42)."""
    n = len(p) - 1
    t = np.array(t)
    u = 1 - t
    tn = np.ones_like(t)
    out = tn * p[0]
    for i in range(n):
        tn *= t
        out = out * u + bnm[n][i + 1] * tn * p[i + 1]
    return out


def bernmul(p, q):
    """Multiply two Bernstein basis polynomials directly (Sederberg p. 107)."""
    m, n = len(p) - 1, len(q) - 1
    return np.convolve(p * bnm[m], q * bnm[n]) / bnm[m + n]


def bernraise(p, n):
    """Raise the degree of p by n without changing the polynomial represented."""
    return bernmul(p, np.ones(n + 1))


def bernder(p):
    """Differentiate (or find the hodograph of) p (Sederberg p. 26)."""
    return np.diff(p) * (len(p) - 1)


def bernsplit(p, t):
    """Splits p at a parameter ratio of t, returning (before, after),
    using de Casteljau's algorithm (Sederberg p. 20)."""
    D = len(p) - 1
    L = 2 * D + 1
    work = np.empty(L)
    work[::2] = p
    u = 1 - t
    for i in range(D):
        work[i + 1 : L - 1 - i : 2] = u * work[i : L - i - 2 : 2] + t * work[i + 2 : L - i : 2]
    return (work[: D + 1], work[D:])


def bernhalf(p):
    """Equivalent to bernsplit(p, 0.5)."""
    D = len(p) - 1
    L = 2 * D + 1
    work = np.empty(L)
    work[::2] = p
    for i in range(D):
        work[i + 1 : L - 1 - i : 2] = (work[i : L - i - 2 : 2] + work[i + 2 : L - i : 2]) * 0.5
    return (work[: D + 1], work[D:])


def bernroots(p):
    """Find roots of p in [0, 1] using root isolation (Sederberg p. 103)."""
    # Deflate roots at zero and one out to condition the isolation process
    zeros, ones = 0, 0
    found = []
    while p[0] == 0:
        zeros += 1
        p = (len(p) - 1) / np.arange(1, len(p)) * p[1:]
    while p[-1] == 0:
        ones += 1
        p = (len(p) - 1) / np.arange(1, len(p))[::-1] * p[:-1]
    stack = [(p, 0, 1)]
    pt = lambda t: berneval(p, t)
    while stack:
        q, start, end = stack.pop()
        # If the interval is very small, temporarily assume no roots
        if end - start < 0.001:
            continue

        sc = sum(np.diff(np.signbit(q)))  # number of sign changes
        if sc == 0:  # guaranteed no roots
            continue
        elif sc == 1:  # guaranteed one root
            found.append(fproot(pt, start, end))
        else:  # split in half and try again
            mid = (start + end) * 0.5
            if pt(mid) == 0:
                found.append(mid)
            pre, post = bernhalf(q)
            stack.extend([(post, mid, end), (pre, start, mid)])
    return [0] * zeros + sorted(found) + [1] * ones


def bernvdm(x, y, n):
    """Assemble a Vandermonde-like matrix whose rows are powers of
    x and y up to nth order, where n is from 1 to 3.
    Rows are ordered 1, x, y, x², xy, y², x³, x²y, xy², y³."""
    D = len(x) - 1  # degree of x and y
    if n == 1:
        return np.array([np.ones(D + 1), x, y])
    x2 = bernmul(x, x)
    xy = bernmul(x, y)
    y2 = bernmul(y, y)
    if n == 2:
        return np.array([np.ones(2 * D + 1), bernraise(x, D), bernraise(y, D), x2, xy, y2])
    return np.array(
        [
            np.ones(3 * D + 1),
            bernraise(x, 2 * D),
            bernraise(y, 2 * D),
            bernraise(x2, D),
            bernraise(xy, D),
            bernraise(y2, D),
            bernmul(x2, x),
            bernmul(xy, x),
            bernmul(xy, y),
            bernmul(y2, y),
        ]
    )


def pen_dgc(p, q):
    """Find a t in [0, 1] such that det((1 - t) * p + t * q) = 0 and return
    the corresponding degenerate conic, or None if no such conic exists."""
    lins = np.stack([p, q], axis=2)
    quads = [
        bernmul(lins[0, i - 2], lins[1, i - 1]) - bernmul(lins[1, i - 2], lins[0, i - 1])
        for i in (0, 1, 2)
    ]
    cubic = sum(bernmul(quads[i], lins[2, i]) for i in (0, 1, 2))
    roots = bernroots(cubic)
    if roots:
        t = roots[0]
        return (1 - t) * p + t * q
    return None


def split_dgc(c, l):
    """Split the real degenerate conic c into its constituent lines by adding
    some multiple of an antisymmetric matrix formed from line l. Return the lines
    expressed by their homogeneous coefficients, or () if they are complex."""
    k = abs(l).argmax()
    disc = c[k - 2, k - 1] ** 2 - c[k - 2, k - 2] * c[k - 1, k - 1]
    if disc < 0:
        return ()
    ansym = np.array([[0, l[2], -l[1]], [-l[2], 0, l[0]], [l[1], -l[0], 0]])
    c += np.sqrt(disc) / l[k] * ansym
    best_i, best_j = np.unravel_index(abs(c).argmax(), (3, 3))
    return (c[best_i], c[:, best_j])


def x_conic(p, q):
    """p and q are two real symmetric 3×3 matrices representing conic sections.
    Return their points of intersection as complex numbers."""
    # Find a degenerate conic c and non-degenerate conic ndg
    if abs(np.linalg.det(p)) <= 1e-8:
        c, ndg = p, q
    elif abs(np.linalg.det(q)) <= 1e-8:
        c, ndg = q, p
    else:
        c = pen_dgc(p, q)
        if c is None:
            c = pen_dgc(p, -q)
        if c is None:
            return []
        ndg = p
    tr_c, c2 = np.trace(c), c @ c
    adj = (tr_c * tr_c - np.trace(c2)) / 2 * np.eye(3) - tr_c * c + c2
    v = adj[abs(adj).argmax() % 3]

    res = []
    for l in split_dgc(c, v):
        res.extend(split_dgc(np.cross(l, np.cross(l, ndg).T), l))
    res = np.array(res).reshape(-1, 3).T
    return (res[0] + 1j * res[1]) / res[2]
