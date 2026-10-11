"""
All roots of  A x = b  whose entries lie in a chosen domain, given an original v.

You supply A and an original v (every entry of v in the domain). Then b = A v, so v
is itself one root, and the program lists EVERY root x with A x = b and every entry
of x in the domain, with its distance from v.

Python:
    from all_roots import find_roots, print_roots
    out = find_roots(A, v)                          # default domain
    out = find_roots(A, v, domain=[0, 1, 2, 3])     # any finite domain you like
    print_roots(out, v)

Command line:
    python all_roots.py --demo
    python all_roots.py --A "1 2 -1 0 3 1; 0 1 1 1 -1 2; 2 0 1 -1 1 0; 1 -1 0 2 0 1" --v "1 1 1 1 1 1"
    python all_roots.py --A "1 1 1; 0 1 2" --v "1 0 2" --domain "0 1 2 3"
    python all_roots.py                             # prompts for A and v
"""
import argparse
import numpy as np

# Default domain: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
DOMAIN = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=float)


# ----------------------------------------------------------------------------- domain helpers
def make_domain(domain):
    """Sorted array of the distinct allowed values."""
    return np.unique(np.asarray(domain, dtype=float))


def in_domain(x, domain, tol=1e-9):
    """True where each entry of x is one of the domain values."""
    x = np.asarray(x, dtype=float)
    return np.min(np.abs(x[..., None] - domain), axis=-1) <= tol


def snap_to_domain(x, domain):
    """Replace each entry by the nearest domain value."""
    x = np.asarray(x, dtype=float)
    return domain[np.argmin(np.abs(x[..., None] - domain), axis=-1)]


def _pivot_columns(A):
    """Indices of a maximal set of linearly independent columns of A."""
    cols = []
    for j in range(A.shape[1]):
        if np.linalg.matrix_rank(A[:, cols + [j]]) > len(cols):
            cols.append(j)
    return cols


# ----------------------------------------------------------------------------- core search
def domain_roots(A, b, domain=DOMAIN, tol=1e-7, chunk=500_000, max_combos=50_000_000):
    """Every x (columns of the result) with A x = b and all entries in `domain`.

    Pick k = n - rank(A) free variables, try every domain value for each
    (len(domain)**k combinations), solve for the remaining variables, and keep a
    candidate only if those also land on domain values.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    domain = make_domain(domain)
    m, n = A.shape
    base = len(domain)

    piv = _pivot_columns(A)
    free = [j for j in range(n) if j not in piv]
    k = len(free)
    total = base ** k
    if total > max_combos:
        raise ValueError(f"{base}**{k} = {total:,} combinations is too many to enumerate; "
                         "use a smaller domain or a matrix with higher rank")

    A_P, A_F = A[:, piv], A[:, free]
    pinv_P = np.linalg.pinv(A_P) if piv else np.zeros((0, m))

    found = []
    for start in range(0, total, chunk):
        rem = np.arange(start, min(start + chunk, total))
        X_F = np.empty((k, rem.size))
        for i in range(k):
            rem, d = np.divmod(rem, base)
            X_F[i] = domain[d]
        rhs = b[:, None] - A_F @ X_F                                   # m x c
        X_P = pinv_P @ rhs                                             # r x c
        resid = np.linalg.norm((A_P @ X_P if piv else 0) - rhs, axis=0)
        ok = resid <= tol
        if piv:
            ok &= in_domain(X_P, domain, tol).all(axis=0)
        if not ok.any():
            continue
        X = np.empty((n, int(ok.sum())))
        if free:
            X[free] = X_F[:, ok]
        if piv:
            X[piv] = snap_to_domain(X_P[:, ok], domain)
        found.append(X)
    return np.hstack(found) if found else np.empty((n, 0))


def find_roots(A, v, domain=DOMAIN, tol=1e-7):
    """All roots of A x = A v with every entry in `domain`, nearest to v first.

    A      : m x n matrix
    v      : original vector, n entries, each in `domain` (so v is one of the roots)
    domain : any finite collection of allowed values

    Returns a dict:
      domain, b (= A v), rank, k (= n - rank)
      roots     : n x r array, columns are the roots (column 0 is v itself)
      offsets   : roots - v
      distances : |x - v| for each root
    """
    A = np.asarray(A, dtype=float)
    v = np.asarray(v, dtype=float)
    domain = make_domain(domain)
    m, n = A.shape
    if v.shape != (n,):
        raise ValueError(f"v must have {n} entries to match A's {n} columns, got {v.size}")
    bad = ~in_domain(v, domain)
    if bad.any():
        raise ValueError("v has entries outside the domain: "
                         + ", ".join(f"v{i + 1}={v[i]:g}" for i in np.where(bad)[0]))

    b = A @ v
    R = domain_roots(A, b, domain, tol)

    dist = np.linalg.norm(R - v[:, None], axis=0)
    order = np.argsort(dist, kind="stable")
    R, dist = R[:, order], dist[order]
    if R.shape[1] == 0 or dist[0] > 1e-9:
        raise RuntimeError("internal error: the original v was not among the roots found")

    return {"domain": domain, "b": b, "rank": int(np.linalg.matrix_rank(A)),
            "k": n - int(np.linalg.matrix_rank(A)),
            "roots": R, "offsets": R - v[:, None], "distances": dist}


# ----------------------------------------------------------------------------- printing
def _fmt(a):
    return "(" + ", ".join(f"{x:g}" for x in a) + ")"


def print_system(A, v, b):
    """Print  A v = b  as a matrix equation."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    rows = max(m, n)
    aw = max(len(f"{x:g}") for x in A.ravel())
    bw = max(len(f"{x:g}") for x in b)
    vw = max(len(f"{x:g}") for x in v)
    print("A v = b:\n")
    for i in range(rows):
        a = "[ " + " ".join(f"{x:>{aw}g}" for x in A[i]) + " ]" if i < m else " " * (n * (aw + 1) + 3)
        vv = f"[ {v[i]:>{vw}g} ]"
        bb = f"[ {b[i]:>{bw}g} ]" if i < m else ""
        mid = "  x  " if i == rows // 2 else "     "
        eq = "  =  " if i == rows // 2 else "     "
        print(f"{a}{mid}{vv}{eq}{bb}")
    print()


def print_roots(out, v, limit=None):
    """Print every root, its offset from v, and its distance from the original v."""
    R, D = out["roots"], out["distances"]
    r = R.shape[1]
    print(f"Domain ({len(out['domain'])} values): {out['domain'].tolist()}")
    print(f"rank(A) = {out['rank']}, free variables k = {out['k']}")
    print(f"\nRoots with every entry in the domain: {r}  (the original v is one of them)\n")

    show = r if limit is None else min(limit, r)
    rows = [(_fmt(R[:, j]), _fmt(out["offsets"][:, j]), D[j]) for j in range(show)]
    w1 = max(len("root x"), max(len(x[0]) for x in rows))
    w2 = max(len("x - v"), max(len(x[1]) for x in rows))
    print(f"{'#':>3}  {'root x':<{w1}}   {'x - v':<{w2}}   distance from v")
    print("-" * (w1 + w2 + 28))
    for j, (xs, os_, d) in enumerate(rows):
        tag = "   <- original v" if d < 1e-12 else ""
        print(f"{j:>3}  {xs:<{w1}}   {os_:<{w2}}   {d:.4f}{tag}")
    if show < r:
        print(f"... {r - show} more roots not shown (raise --limit)")
    others = D[D > 1e-12]
    if others.size:
        print(f"\nNearest other root: {others.min():.4f}    farthest root: {D.max():.4f}")
    else:
        print("\nThe original v is the only root in this domain.")


# ----------------------------------------------------------------------------- input helpers
def read_matrix(text=None):
    """`text` uses ';' between rows, e.g. "1 2 3; 4 5 6". Without text, prompts row by row."""
    if text is None:
        print("Enter A one row per line (spaces or commas between numbers). Blank line to finish:")
        lines = []
        while True:
            line = input("  row: ").strip()
            if not line:
                break
            lines.append(line)
    else:
        lines = [r for r in text.split(";") if r.strip()]
    try:
        rows = [[float(x) for x in r.replace(",", " ").split()] for r in lines]
    except ValueError:
        raise SystemExit("A contains something that is not a number.")
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        raise SystemExit("A must be non-empty, with the same number of entries in every row.")
    return np.array(rows)


def read_vector(n, text=None, name="v"):
    raw = text if text is not None else input(f"Enter {name} ({n} numbers): ")
    try:
        vals = [float(x) for x in raw.replace(",", " ").split()]
    except ValueError:
        raise SystemExit(f"{name} contains something that is not a number.")
    if len(vals) != n:
        raise SystemExit(f"{name} needs {n} numbers, got {len(vals)}.")
    return np.array(vals)


# ----------------------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description="All roots of A x = A v with entries in a chosen domain.")
    ap.add_argument("--A", help='matrix, rows split by ";" e.g. "1 2 3; 4 5 6"')
    ap.add_argument("--v", help='original vector, e.g. "1 0 2"; every entry must be in the domain')
    ap.add_argument("--domain", help='allowed values, e.g. "0 1 2 3" (default: 0, +-0.5, +-1, +-1.5, +-2, +-3, +-4, +-6)')
    ap.add_argument("--limit", type=int, default=50, help="max roots to print (default 50)")
    ap.add_argument("--demo", action="store_true", help="run the built-in 4x6 example")
    args = ap.parse_args()

    if args.demo:
        A = np.array([[1, 2, -1, 0, 3, 1],
                      [0, 1, 1, 1, -1, 2],
                      [2, 0, 1, -1, 1, 0],
                      [1, -1, 0, 2, 0, 1]], dtype=float)
        v = np.ones(6)
    else:
        A = read_matrix(args.A)
        v = read_vector(A.shape[1], args.v)

    domain = DOMAIN if args.domain is None else make_domain(read_vector(len(args.domain.replace(",", " ").split()),
                                                                          args.domain, "domain"))
    try:
        out = find_roots(A, v, domain)
    except (ValueError, RuntimeError) as e:
        raise SystemExit(f"Error: {e}")

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print_system(A, v, out["b"])
    print_roots(out, v, limit=args.limit)


if __name__ == "__main__":
    main()