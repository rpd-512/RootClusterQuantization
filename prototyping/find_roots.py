"""
Sweep every vector v in DOMAIN^n against a random m x n matrix A.

For each of the 15^n vectors v:   b = A v  stored with --bbits bits per component
(default 4, i.e. 16 levels; --bbits 0 keeps b exact). The roots of v are every x in
DOMAIN^n whose A x gives the same stored b, and the program measures the distance
from v to each of them. v itself is always one of its roots.

Instead of solving 15^n systems one by one, it uses the fact that the roots of
A x = A v are exactly the other vectors in DOMAIN^n that map to the same b. So it
computes b for all 15^n vectors, groups the vectors by b, and measures distances
inside each group.

Roots are counted only inside the domain, so every system has finitely many roots.

Python:
    from root_sweep import run_experiment, print_report
    res = run_experiment(n=6, m=3, seed=1)
    print_report(res)

Command line:
    python root_sweep.py 6 3
    python root_sweep.py 5 2 --trials 10 --seed 1
"""
import argparse
import time
import numpy as np

DOMAIN = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=float)
SCALE = 2          # DOMAIN * SCALE must be integers so that b can be keyed exactly


# ----------------------------------------------------------------------------- helpers
def decode(idx, domain, n):
    """Index 0..len(domain)**n-1  ->  vectors (len(idx) x n) with entries from `domain`."""
    base = len(domain)
    X = np.empty((len(idx), n))
    rem = np.asarray(idx).copy()
    for i in range(n):
        rem, d = np.divmod(rem, base)
        X[:, i] = domain[d]
    return X


def quantize_b(B, bmax, bbits):
    """Quantize each column of B (rows of A) to 2**bbits levels spread evenly over [-bmax_i, +bmax_i].
    Returns integer level indices 0 .. 2**bbits - 1."""
    levels = 2 ** bbits
    step = 2.0 * bmax / (levels - 1)
    step = np.where(step == 0, 1.0, step)
    return np.clip(np.rint((B + bmax) / step), 0, levels - 1).astype(np.int64)


def _keys_for_all_vectors(A, domain, chunk=1_000_000, bbits=None):
    """One int64 key per vector in DOMAIN^n identifying its b = A v
    (exactly if bbits is None, otherwise b rounded to `bbits` bits per component)."""
    m, n = A.shape
    base = len(domain)
    N = base ** n
    if bbits is None:
        bmax = int(np.ceil(np.abs(A).sum(axis=1).max() * np.abs(domain).max() * SCALE))
        radix, offset = 2 * bmax + 1, bmax
    else:
        radix, offset = 2 ** bbits, 0
        bmax_rows = np.abs(A).sum(axis=1) * np.abs(domain).max()      # largest possible |b_i|
    if radix ** m >= 2 ** 62:
        raise ValueError("m is too large for the exact key encoding; use a smaller m or amax")
    keys = np.empty(N, dtype=np.int64)
    for s in range(0, N, chunk):
        idx = np.arange(s, min(s + chunk, N))
        B = decode(idx, domain, n) @ A.T
        if bbits is None:
            Q = np.rint(B * SCALE).astype(np.int64) + offset
        else:
            Q = quantize_b(B, bmax_rows, bbits)
        key = np.zeros(len(idx), dtype=np.int64)
        for j in range(m):
            key = key * radix + Q[:, j]
        keys[s:s + len(idx)] = key
    return keys


# ----------------------------------------------------------------------------- one matrix
def sweep_matrix(A, domain=DOMAIN, max_pairs=500_000_000, chunk=1_000_000, rng=None, bbits=None):
    """Sweep all vectors in DOMAIN^n for one matrix A. Returns sums and counts."""
    A = np.asarray(A, dtype=float)
    domain = np.asarray(domain, dtype=float)
    m, n = A.shape
    if not np.allclose(domain * SCALE, np.rint(domain * SCALE)):
        raise ValueError("domain values must be multiples of 0.5")
    if not np.allclose(A, np.rint(A)):
        raise ValueError("A must have integer entries")
    N = len(domain) ** n

    keys = _keys_for_all_vectors(A, domain, chunk, bbits)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    del keys

    starts = np.r_[0, np.flatnonzero(np.diff(sk)) + 1]
    sizes = np.diff(np.r_[starts, N])
    pairs = int((sizes * (sizes - 1) // 2).sum())
    if pairs > max_pairs:
        raise ValueError(f"{pairs:,} vector pairs share a b value (limit {max_pairs:,}); "
                         "use a larger m, a smaller n, or raise --max-pairs")
    gsize = np.repeat(sizes, sizes)                      # roots per vector, in sorted order

    rowsum = np.zeros(N)                                 # sum of distances from each vector to its other roots
    cand = np.flatnonzero(sk[:-1] == sk[1:])             # positions p with p and p+d in the same group
    d = 1
    while cand.size:
        for c0 in range(0, cand.size, chunk):
            p = cand[c0:c0 + chunk]
            dist = np.linalg.norm(decode(order[p], domain, n) - decode(order[p + d], domain, n), axis=1)
            rowsum[p] += dist
            rowsum[p + d] += dist
        d += 1
        cand = cand[cand + d < N]
        cand = cand[sk[cand] == sk[cand + d]]

    multi = gsize > 1
    sum_incl = float((rowsum / gsize).sum())             # root set includes v itself (distance 0)
    sum_excl = float((rowsum[multi] / (gsize[multi] - 1)).sum())

    # Roots are counted INSIDE the domain only, so every system has finitely many roots.
    # A vector's roots are the members of its group; v itself is always one of them.
    n_unsolvable = int((gsize < 1).sum())                # no root in the domain (cannot happen: v is a root)
    n_unique = int((gsize == 1).sum())                   # v is the only root in the domain
    n_multi = int(multi.sum())                           # v plus at least one other root in the domain
    return {"N": N, "rank": int(np.linalg.matrix_rank(A)),
            "sum_incl": sum_incl, "n_multi": n_multi, "n_unique": n_unique,
            "n_unsolvable": n_unsolvable, "sum_excl": sum_excl,
            "max_roots": int(sizes.max())}


# ----------------------------------------------------------------------------- experiment
def run_experiment(n, m, trials=1, seed=None, amax=3, domain=DOMAIN, max_pairs=500_000_000, bbits=4):
    """Random integer A (m x n, entries in [-amax, amax]) swept over all vectors in domain^n.
    With trials > 1 a fresh A is drawn each time and the tallies are added together."""
    rng = np.random.default_rng(seed)
    tot = {"combos": 0, "unique": 0, "multi": 0, "unsolvable": 0,
           "sum_incl": 0.0, "n_multi": 0, "sum_excl": 0.0, "max_roots": 0}
    matrices, ranks = [], []
    t0 = time.time()
    for _ in range(trials):
        A = rng.integers(-amax, amax + 1, size=(m, n)).astype(float)
        r = sweep_matrix(A, domain, max_pairs, rng=rng, bbits=bbits or None)
        matrices.append(A)
        ranks.append(r["rank"])
        tot["combos"] += r["N"]
        tot["unique"] += r["n_unique"]
        tot["multi"] += r["n_multi"]
        tot["unsolvable"] += r["n_unsolvable"]
        tot["max_roots"] = max(tot["max_roots"], r["max_roots"])
        tot["sum_incl"] += r["sum_incl"]
        tot["n_multi"] += r["n_multi"]
        tot["sum_excl"] += r["sum_excl"]
    tot.update(n=n, m=m, trials=trials, bbits=bbits or None, seed=seed, domain=np.asarray(domain), matrices=matrices,
               ranks=ranks, seconds=time.time() - t0)
    return tot


def print_report(res):
    n, m, T = res["n"], res["m"], res["trials"]
    base = len(res["domain"])
    print(f"n = {n} (vector size), m = {m} (equations), domain size = {base}")
    print(f"Combinations of v per matrix: {base}^{n} = {base ** n:,}   matrices: {T}   "
          f"systems swept: {res['combos']:,}")
    if T == 1:
        print("\nA =")
        print(res["matrices"][0].astype(int))
    print(f"rank(A) per matrix: {res['ranks']}")
    if res["bbits"]:
        print(f"b is stored with {res['bbits']} bits per component ({2 ** res['bbits']} levels over its full range)")
        print("A vector x is a root of v if A x rounds to the same b as A v.")
    else:
        print("b is exact (no rounding); x is a root of v if A x = A v.")

    avg_incl = res["sum_incl"] / res["combos"]
    avg_excl = res["sum_excl"] / res["n_multi"] if res["n_multi"] else float("nan")
    print("\n" + "=" * 66)
    print(f"Average distance from v to its roots, all combinations: {avg_incl:.6f}")
    print("   (each v's roots include v itself at distance 0)")
    print(f"Same, counting only the other roots, over the {res['n_multi']:,} combinations")
    print("   that have another root in the domain:                "
          + (f"{avg_excl:.6f}" if res["n_multi"] else "n/a (none)"))
    print("-" * 66)
    finite = res["unique"] + res["multi"]
    print(f"Roots finite     (at least one root in the domain):     {finite:,} times")
    print(f"   only v itself is a root in the domain:               {res['unique']:,}")
    print(f"   v plus other roots in the domain:                    {res['multi']:,}")
    print(f"Roots infinite   (not possible: the domain is finite):  0 times")
    print(f"Unsolvable       (no root in the domain):               {res['unsolvable']:,} times")
    print(f"(most roots any one v had in the domain: {res['max_roots']:,})")
    print("=" * 66)
    print(f"({res['seconds']:.1f} s)")


# ----------------------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description="Sweep all domain vectors against a random m x n matrix.")
    ap.add_argument("n", type=int, nargs="?", help="size of the vector v")
    ap.add_argument("m", type=int, nargs="?", help="number of equations (rows of A)")
    ap.add_argument("--trials", type=int, default=1, help="number of random matrices (default 1)")
    ap.add_argument("--seed", type=int, default=None, help="random seed")
    ap.add_argument("--amax", type=int, default=3, help="entries of A are integers in [-amax, amax] (default 3)")
    ap.add_argument("--bbits", type=int, default=4,
                    help="bits per component used to store b (default 4; 0 = exact b)")
    ap.add_argument("--max-pairs", type=int, default=500_000_000, help="refuse sweeps with more pairs than this")
    args = ap.parse_args()

    n = args.n if args.n is not None else int(input("n (size of v): "))
    m = args.m if args.m is not None else int(input("m (number of equations): "))
    if n < 1 or m < 1:
        raise SystemExit("n and m must be at least 1")
    if len(DOMAIN) ** n > 30_000_000:
        raise SystemExit(f"{len(DOMAIN)}^{n} = {len(DOMAIN) ** n:,} combinations is too many "
                         "(limit 30,000,000, i.e. n <= 6)")
    try:
        res = run_experiment(n, m, args.trials, args.seed, args.amax, max_pairs=args.max_pairs, bbits=args.bbits)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")
    print_report(res)


if __name__ == "__main__":
    main()