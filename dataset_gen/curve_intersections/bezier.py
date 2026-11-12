# Taken from Kinross (https://gitlab.com/parclytaxel/Kinross).
# Copyright (c) 2018 Parcly Taxel / Jeremy Tan
# Licensed under the MIT License; see LICENSE-KINROSS in this directory.

from functools import lru_cache

import numpy as np

from .algebra import ccquad, fproot
from .bernpol import bernder, berneval, bernmul, bernroots, bernvdm


class bezier:
    powbasis = (
        np.array([[1, 0], [-1, 1]]),
        np.array([[1, 0, 0], [-2, 2, 0], [1, -2, 1]]),
        np.array([[1, 0, 0, 0], [-3, 3, 0, 0], [3, -6, 3, 0], [-1, 3, -3, 1]]),
    )
    bernbasis = (
        np.array([[1, 0], [1, 1]]),
        np.array([[1, 0, 0], [1, 1 / 2, 0], [1, 1, 1]]),
        np.array([[1, 0, 0, 0], [1, 1 / 3, 0, 0], [1, 2 / 3, 1 / 3, 0], [1, 1, 1, 1]]),
    )

    def __init__(self, *cpoints):
        """Initialise a bezier instance from the given control points,
        a sequence of 2 to 4 complex numbers. The control points are accessed
        as self.v."""
        if not (2 <= len(cpoints) <= 4):
            raise ValueError("bezier constructor requires two to four control points")
        self.v = np.array(cpoints)
        self.deg = len(cpoints) - 1
        # We also compute derivatives of v for faster retrieval
        self.dv = bernder(self.v)
        self.d2v = bernder(self.dv)

    def __str__(self):
        """Return a string representation of this curve."""
        return f"bezier({','.join([str(z).strip('()') for z in self.v])})"

    def __repr__(self):
        return str(self)

    def __call__(self, t):
        """Evaluate this curve at real parameter(s) t."""
        return berneval(self.v, t)

    def __getitem__(self, k):
        """If called with a number, return the control point with that index.
        If called with a slice [a:b], return the curve reparametrised such that
        t = 0, 1 in the new curve are t = a, b in the old curve,
        where a, b default to 0, 1."""
        if type(k) == slice:
            a = 0 if k.start is None else k.start
            b = 1 if k.stop is None else k.stop
            # Define the parameter transformation matrix
            ptm = np.zeros((self.deg + 1, self.deg + 1))
            for i in range(self.deg + 1):
                ptm[: i + 1, i] = np.polynomial.polynomial.polypow([a, b - a], i)
            newv = bezier.bernbasis[self.deg - 1] @ ptm @ bezier.powbasis[self.deg - 1] @ self.v
            return bezier(*newv)
        return self.v[k]

    def __neg__(self):
        """Reverse this curve and return it."""
        return bezier(*self.v[::-1])

    def __rmatmul__(self, m):
        """Transform this curve using the given mt instance."""
        return bezier(*(m @ self.v))

    @property
    @lru_cache()
    def infl(self):
        """Compute the t-values of a cubic curve's inflections, where the curvature
        vanishes or x'y'' - x''y' = 0. The property and cache are defined since
        they do not depend on any input."""
        if self.deg != 3:
            return []
        # CQ is the pseudoinverse of the quadratic-to-cubic degree elevation matrix,
        # which yields the least-squares solution to the inverse problem.
        # infl_pol is calculated as a cubic, but is really a quadratic,
        # so we left-multiply by CQ.
        CQ = np.array(
            [[0.95, 0.15, -0.15, 0.05], [-0.25, 0.75, 0.75, -0.25], [0.05, -0.15, 0.15, 0.95]]
        )
        infl_pol = CQ @ (
            bernmul(self.dv.real, self.d2v.imag) - bernmul(self.dv.imag, self.d2v.real)
        )
        return bernroots(infl_pol)

    @property
    @lru_cache()
    def cx(self):
        """For cubic curves containing a loop, compute the parameters t, u where
        B(t) = B(u), i.e. the parameters of self-intersection.
        The method name is both a phonetic abbreviation ("c"elf "x")
        and an ASCII depiction of a loopy curve.

        As Bézier curve geometry is invariant under affine transformations,
        we can restrict ourselves to some canonical form of the curve.
        Then, consider the coordinate polynomials
        x(t) = at³+bt²+ct+p
        y(t) = dt³+et²+ft+q
        and parameters of self-intersection as λ and μ. By definition,
        x(λ) - x(μ) = a(λ³-μ³)+b(λ²-μ²)+c(λ-μ) = 0
        y(λ) - y(μ) = d(λ³-μ³)+e(λ²-μ²)+f(λ-μ) = 0
        Dividing by the trivial solution λ = μ and expanding we get
        a(λ²+λμ+μ²)+b(λ+μ)+c = 0
        d(λ²+λμ+μ²)+e(λ+μ)+f = 0
        In the canonical form chosen here
        (https://pomax.github.io/bezierinfo/#canonical) we have
        (x-3)(λ²+λμ+μ²)+3(λ+μ) = 0
        y(λ²+λμ+μ²)-3(λ+μ)+3 = 0
        whereby eliminating λ²+λμ+μ² gives (-3-3y/(x-3))(λ+μ) + 3 = 0
        or λ+μ = (x-3)/(x+y-3), followed by λμ = (λ+μ)²+3/(x+y-3).
        λ and μ can now be found by Viète's formulas."""
        if self.deg != 3:
            return []
        vx, vy, vz = self[2] - self[1], self[1] - self[0], self[3] - self[0]
        try:
            x, y = np.linalg.solve([[vx.real, vy.real], [vx.imag, vy.imag]], [vz.real, vz.imag])
        except np.linalg.LinAlgError:
            return []
        if (
            x > 1
            or 4 * y > (x + 1) * (3 - x)
            or x > 0
            and 2 * y + x < np.sqrt(3 * x * (4 - x))
            or 3 * y < x * (3 - x)
        ):
            return []
        rs = (x - 3) / (x + y - 3)
        rp = rs * rs + 3 / (x + y - 3)
        x1 = (rs - np.sqrt(rs * rs - 4 * rp)) / 2
        return sorted([x1, rp / x1])

    @property
    @lru_cache()
    def length_integrand(self):
        """Integrand for the arc length function."""
        dx, dy = self.dv.real, self.dv.imag
        norm = bernmul(dx, dx) + bernmul(dy, dy)
        return lambda t: np.sqrt(berneval(norm, t))

    def length(self, t2=1, t1=0):
        """Measure this curve's length over [t1, t2]."""
        if self.deg == 1:
            return abs(self[1] - self[0]) * (t2 - t1)
        return ccquad(self.length_integrand, t1, t2)

    def t_length(self, target):
        """Compute t with self.length(t) = target. This works even for lengths
        beyond the curve proper and negative lengths."""
        # Struzik search starting from t = ±1 to find a bound
        bound = np.copysign(1, target)
        while abs(self.length(bound)) < abs(target):
            bound *= 2
        return fproot(lambda t: self.length(t) - target, 0, bound)

    def project(self, p):
        """Project p onto this curve and return the projection's t-value.
        Besides the endpoints, the relevant parameters satisfy (x-p)x' + (y-p)y' = 0."""
        foot = self.v - p  # the constant 1 is a vector of ones in any Bernstein basis
        feeteq = bernmul(foot.real, self.dv.real) + bernmul(foot.imag, self.dv.imag)
        cand_t = bernroots(feeteq) + [0, 1]
        return cand_t[abs(self(cand_t) - p).argmin()]

    @property
    @lru_cache()
    def t_inv(self):
        """Derive an inversion formula for this curve,
        t = (a_1 x + b_1 y + c_1) / (a_2 x + b_2 y + c_2).
        Return np.array([[a_1, b_1, c_1], [a_2, b_2, c_2]]) (Sederberg p. 205)."""
        if self.deg == 3:
            C = np.stack([self.v.real, self.v.imag, np.ones(4)], axis=1)
            # Multiplication in the Bernstein basis by a fixed polynomial can be
            # expressed as a matrix multiplication on the coefficients.
            # U and T are the matrices corresponding to multiplication by
            # <1, 1> and <0, 1> respectively where the variable polynomial is cubic.
            U = np.array(
                [
                    [1, 0, 0, 0],
                    [0.25, 0.75, 0, 0],
                    [0, 0.5, 0.5, 0],
                    [0, 0, 0.75, 0.25],
                    [0, 0, 0, 1],
                ]
            )
            T = np.diag([0.25, 0.5, 0.75, 1])
            A = np.zeros((5, 6))
            A[:, :3] = -(U @ C)
            A[1:, 3:] = T @ C
            return np.linalg.qr(A.T, "complete")[0][:, -1].reshape(2, 3)
        if self.deg == 2:
            Q = np.stack([self.v.real, self.v.imag, np.ones(3)], axis=1)
            return np.array([np.linalg.solve(Q, [0, 0.5, 1]), [0, 0, 1]])
        # the curve is linear, use the projection formula
        delta = self[1] - self[0]
        res = np.array(
            [delta.real, delta.imag, -delta.real * self[0].real - delta.imag * self[0].imag]
        )
        return np.array([res / abs(delta) ** 2, [0, 0, 1]])

    @property
    @lru_cache()
    def implicit(self):
        """Compute an implicit representation of this curve. Coefficients
        correspond to monomials in the same order as returned by bernvdm()."""
        A = bernvdm(self.v.real, self.v.imag, self.deg)
        return np.linalg.qr(A, "complete")[0][:, -1]

    def x_bez(self, other):
        """Compute the intersection between this curve and curve other using
        implicitisation and inversion. Return an np.array with two columns,
        each row corresponding to an intersection on [0,1]×[0,1] and listing first
        this curve's parameter, then the other curve's."""
        # If one of the curves has smaller degree, that curve is implicitised
        # (ic) and the other one's x(t) and y(t) are subbed into it (tc)
        swap = self.deg > other.deg  # swapping from ic = self and tc = other?
        ic, tc = (other, self) if swap else (self, other)
        tc_pol = bernvdm(tc.v.real, tc.v.imag, ic.deg).T @ ic.implicit
        tc_t = np.array(bernroots(tc_pol))
        xy = tc(tc_t)
        nd = ic.t_inv @ [np.real(xy), np.imag(xy), np.ones_like(xy, dtype=float)]
        ic_t = nd[0] / nd[1]
        vind = np.nonzero((0 <= ic_t) & (ic_t <= 1))[0]
        tc_t, ic_t = tc_t[vind], ic_t[vind]
        bundle = (tc_t, ic_t) if swap else (ic_t, tc_t)
        return np.stack(bundle, axis=1)

    def x_ell(self, other):
        """Similar to x_bez(), but other is an elliptical arc. This is solved
        easily by deriving the implicit form of other and substituting."""
        Q = other.qmat
        ici = np.array([Q[2, 2], 2 * Q[0, 2], 2 * Q[1, 2], Q[0, 0], 2 * Q[0, 1], Q[1, 1]])
        tc_pol = bernvdm(self.v.real, self.v.imag, 2).T @ ici
        tc_t = np.array(bernroots(tc_pol))
        ic_t = other.inv(self(tc_t))
        vind = np.nonzero((0 <= ic_t) & (ic_t <= 1))[0]
        tc_t, ic_t = tc_t[vind], ic_t[vind]
        return np.stack([tc_t, ic_t], axis=1)
