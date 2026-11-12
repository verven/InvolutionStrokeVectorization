# NumPy port of Jake Low's JavaScript implementation of Hobby's algorithm
# (https://www.jakelow.com/blog/hobby-curves/hobby.js), described in
# https://www.jakelow.com/blog/hobby-curves. The original code is distributed
# under the following license:
#
# ISC License
#
# Copyright 2020 Jake Low
#
# Permission to use, copy, modify, and/or distribute this software for any purpose
# with or without fee is hereby granted, provided that the above copyright notice
# and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER
# TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
# THIS SOFTWARE.

import numpy as np

def hobby(points, omega=0.0):
    """
    Hobby's algorithm: given a set of points, fit a Bézier spline to them.
    
    Parameters:
      - points: an array of points as [x, y] pairs (or a numpy array)
      - omega: a number between 0 and 1 (inclusive); controls how much curl
        there will be at the endpoints of the curve
    
    Returns: A numpy array of shape (3n-2, 2) containing the spline points.
             The sequence is [Knot, Control1, Control2, Knot, Control1...].
    """
    points = np.array(points, dtype=float)
    
    # solving is only possible if there are at least two points
    assert len(points) >= 2, "Must have at least 2 points"

    n = len(points) - 1

    # chords[i] is the vector from P[i] to P[i+1].
    # d[i] is the length of the ith chord.
    # We can compute all chords and lengths in one go using numpy vectorization.
    chords = points[1:] - points[:-1]
    d = np.linalg.norm(chords, axis=1)

    # no chord can be zero-length (i.e. no two successive points can be the same)
    assert np.all(d > 0), "Duplicate successive points detected"

    # gamma[i] is the signed turning angle at P[i].
    # We calculate the angle of every chord, then find the difference.
    # np.arctan2(y, x) gives the angle of a vector.
    chord_angles = np.arctan2(chords[:, 1], chords[:, 0])
    
    # Initialize gamma. Size is n + 1 so indices match P[0]...P[n]
    gamma = np.zeros(n + 1)
    
    # Calculate turning angles for internal points (1 to n-1).
    # This represents vAngleBetween(chords[i-1], chords[i])
    # The angle between two vectors is the difference of their absolute angles.
    # We wrap to (-pi, pi) to ensure correct turning direction, though 
    # for typical splines simple subtraction usually works.
    diffs = chord_angles[1:] - chord_angles[:-1]
    gamma[1:n] = np.arctan2(np.sin(diffs), np.cos(diffs))
    
    # gamma[0] is undefined (stays 0); gamma[n] is artificially defined to be zero
    gamma[n] = 0.0

    # Set up the system of linear equations (tridiagonal matrix).
    A = np.zeros(n + 1)
    B = np.zeros(n + 1)
    C = np.zeros(n + 1)
    D = np.zeros(n + 1)

    # Boundary condition at i=0
    B[0] = 2.0 + omega
    C[0] = 2.0 * omega + 1.0
    D[0] = -1.0 * C[0] * gamma[1]

    # Fill internal points
    # Using slicing to vectorize the loop: for i in range(1, n)
    # Note: d indices are shifted compared to matrix indices i.
    # d[i-1] maps to d_prev, d[i] maps to d_curr
    d_prev = d[0:n-1]
    d_curr = d[1:n]
    
    A[1:n] = 1.0 / d_prev
    B[1:n] = (2.0 * d_prev + 2.0 * d_curr) / (d_prev * d_curr)
    C[1:n] = 1.0 / d_curr
    
    g_curr = gamma[1:n]
    g_next = gamma[2:n+1]
    
    D[1:n] = (-1.0 * (2.0 * g_curr * d_curr + g_next * d_prev)) / (d_prev * d_curr)

    # Boundary condition at i=n
    A[n] = 2.0 * omega + 1.0
    B[n] = 2.0 + omega
    D[n] = 0.0

    # Solve for alpha angles
    alpha = thomas(A, B, C, D)

    # Solve for beta angles
    beta = np.zeros(n)
    # for i in range(0, n - 1): beta[i] = -gamma[i+1] - alpha[i+1]
    beta[0:n-1] = -gamma[1:n] - alpha[1:n]
    beta[n-1] = -alpha[n]

    # Compute handle (control point) positions
    c0 = np.zeros((n, 2))
    c1 = np.zeros((n, 2))

    # Vectorized calculation of handle lengths using rho function
    # rho inputs are arrays, returns array
    a_len = (rho(alpha[:n], beta) * d) / 3.0
    b_len = (rho(beta, alpha[:n]) * d) / 3.0

    # Calculate control points.
    # Instead of vRot, we simply add the computed angles to the base chord angles.
    
    # Angle of handle departing P[i]
    theta0 = chord_angles + alpha[:n]
    # Angle of handle arriving at P[i+1] (from P[i] perspective, reversed)
    theta1 = chord_angles - beta
    
    # Convert polar (length, angle) to cartesian (x, y) and add/sub from knots
    c0[:, 0] = points[:n, 0] + a_len * np.cos(theta0)
    c0[:, 1] = points[:n, 1] + a_len * np.sin(theta0)
    
    c1[:, 0] = points[1:, 0] - b_len * np.cos(theta1)
    c1[:, 1] = points[1:, 1] - b_len * np.sin(theta1)

    # Gather results into a single list
    res = []
    for i in range(n):
        res.append(points[i])
        res.append(c0[i])
        res.append(c1[i])
    res.append(points[n])

    return np.array(res)

def rho(alpha, beta):
    """
    Velocity function that computes the length of the handles.
    Works with scalars or numpy arrays.
    """
    c = 2.0 / 3.0
    return 2.0 / (1.0 + c * np.cos(beta) + (1.0 - c) * np.cos(alpha))

def thomas(a, b, c, d):
    """
    Solves a tridiagonal system Ax = d using the Thomas algorithm (TDMA).
    
    Parameters:
    a: lower diagonal (a[0] is not used)
    b: main diagonal
    c: upper diagonal (c[-1] is not used)
    d: right-hand side vector
    """
    n = len(d)
    # Make copies to avoid modifying inputs
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    # Forward elimination
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        temp = b[i] - a[i] * c_prime[i-1]
        if i < n - 1:
            c_prime[i] = c[i] / temp
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / temp
        
    # Backward substitution
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x