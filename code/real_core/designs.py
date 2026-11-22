"""Covariate-only probability designs for the BUD study.

Every decision here is a function of Phase-I X and the fixed sampling budget.
GLP selection minimizes only mixture discrepancy (MD), never an outcome score.
BUD capacities are exact. SRS is without replacement independently by stratum.
"""
from functools import lru_cache
from math import gcd
from time import perf_counter
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, qmc
from sklearn.cluster import KMeans


def standardized_pca(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('X must be a finite two-dimensional array with N >= 2.')
    sd = np.std(x, axis=0, ddof=0)
    safe_sd = np.where(sd > 1e-12, sd, 1.0)
    z = (x - np.mean(x, axis=0)) / safe_sd
    u, singular, vt = np.linalg.svd(z, full_matrices=False)
    # Fix the otherwise arbitrary PCA orientation, without looking at outcomes.
    orientations = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
    orientations[orientations == 0] = 1
    scores = (u * singular) * orientations
    eigenvalues = singular**2 / len(x)
    if eigenvalues.sum() <= 1e-14:
        q = 1
    else:
        q = int(np.searchsorted(np.cumsum(eigenvalues) / eigenvalues.sum(), .85)) + 1
    return z, scores, eigenvalues, q


@lru_cache(maxsize=128)
def glp_uniform_design(h, q):
    """A deterministic leave-one-out rank-one lattice selected by MD.

    Power generators are used when available. If none has q distinct powers,
    use a fixed finite list of rank-one generators with coprime coordinates.
    This maintains exact one-dimensional Latin projections in every case.
    The return is immutable to protect the shared cache.
    """
    h, q = int(h), int(q)
    if h < 1 or q < 1:
        raise ValueError('Positive h and q are required.')
    if h == 1:
        out = np.full((1, q), .5)
        out.setflags(write=False)
        return out
    modulus = h + 1
    unit = np.array([a for a in range(1, modulus) if gcd(a, modulus) == 1])
    generators = []
    for a in unit:
        g = np.array([pow(int(a), j, modulus) for j in range(q)], dtype=int)
        if len(np.unique(g)) == q:
            generators.append(g)
    if not generators:
        rng = np.random.default_rng(781211 + 997*h + q)
        # With q above the number of coprime residues, repeated coordinates
        # are unavoidable for this lattice family and explicitly recorded.
        for _ in range(32):
            g = rng.choice(unit, size=q, replace=(q > len(unit)))
            generators.append(g)
    best, best_md = None, np.inf
    index = np.arange(1, h + 1)[:, None]
    for gen in generators:
        design = ((index * gen[None, :]) % modulus - .5) / h
        md = qmc.discrepancy(design, method='MD')
        if md < best_md:
            best, best_md = design, md
    best.setflags(write=False)
    return best


def balanced_assignment(coordinates, anchors):
    """Exact minimum quadratic transport to equal-capacity fixed anchors."""
    coordinates = np.asarray(coordinates, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    n, h = len(coordinates), len(anchors)
    if n % h:
        raise ValueError('Exact BUD requires N divisible by the number of strata.')
    k = n // h
    distances = cdist(coordinates, anchors, metric='sqeuclidean')
    # Repeated columns turn capacitated transport into an assignment problem.
    row, col = linear_sum_assignment(np.repeat(distances, k, axis=1))
    labels = np.empty(n, dtype=int)
    labels[row] = col // k
    return labels, float(np.mean(distances[np.arange(n), labels]))


def equal_quantile_labels(v, h):
    n = len(v)
    if n % h:
        raise ValueError('Equal-capacity PC1 strata require N divisible by H.')
    order = np.argsort(v, kind='stable')
    labels = np.empty(n, dtype=int)
    labels[order] = np.repeat(np.arange(h), n // h)
    return labels


def inverse_quantile_anchors(scores, design):
    return np.column_stack([
        np.quantile(scores[:, j], design[:, j], method='linear')
        for j in range(scores.shape[1])
    ])


def _merge_tiny_kmeans_clusters(z, labels):
    """Merge singleton clusters to their nearest existing cluster centroid."""
    labels = np.asarray(labels, dtype=int).copy()
    while True:
        values, sizes = np.unique(labels, return_counts=True)
        small = values[sizes < 2]
        if len(small) == 0:
            break
        if len(values) == 1:
            raise ValueError('N < 2 leaves no legal bootstrap stratum.')
        h = int(small[0])
        others = values[values != h]
        centroids = np.array([z[labels == j].mean(axis=0) for j in others])
        c = z[labels == h].mean(axis=0)
        nearest = others[np.argmin(np.sum((centroids - c)**2, axis=1))]
        labels[labels == h] = nearest
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def proportional_allocation(sizes, m, minimum=2):
    """Integer capped proportional allocation with at least two per stratum.

    Begin with the minimum, then repeatedly allocate to the largest shortfall
    from m*N_h/N among noncensused strata. Exactly m samples are allocated.
    """
    sizes = np.asarray(sizes, dtype=int)
    if np.any(sizes < minimum) or not minimum*len(sizes) <= m <= sizes.sum():
        raise ValueError('Infeasible minimum/capacity/sample-size constraints.')
    ideal = m * sizes / sizes.sum()
    allocation = np.full(len(sizes), minimum, dtype=int)
    while allocation.sum() < m:
        deficit = ideal - allocation
        deficit[allocation >= sizes] = -np.inf
        allocation[int(np.argmax(deficit))] += 1
    return allocation


def _within_scatter(z, labels):
    total = 0.0
    for h in np.unique(labels):
        zh = z[labels == h]
        total += np.sum((zh - zh.mean(axis=0))**2)
    return float(total / len(z))


def make_design(x, m, variant):
    """Return (labels, integer allocation, diagnostics).

    Primary methods:
      SRS, PC1-X, KMEANS-X,
      BUD-rank4, BUD-rank2, BUD-inverse4, BUD-inverse2,
      BUD-lloyd2, BUD-sliced2.
    Sensitivities: PC1-X4, KMEANS-X2, BALANCED-KMEANS-X2.
    KMEANS-X always fits all standardized raw X columns, with no PCA input.
    It uses H=m//4 so minimum-two inference can coexist with proportional
    allocation. KMEANS-X2 illustrates the allocation penalty at H=m//2.
    """
    start = perf_counter()
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    m = int(m)
    if not 2 <= m <= n:
        raise ValueError('Require 2 <= sample size <= population size.')
    z, scores, eigenvalues, q = standardized_pca(x)
    extra = {}
    objective = np.nan
    if variant == 'SRS':
        labels = np.zeros(n, dtype=int)
        allocation = np.array([m], dtype=int)
    elif variant in ('PC1-X', 'PC1-X4'):
        s = 4 if variant.endswith('4') else 2
        if m % s:
            raise ValueError('Sample size must be divisible by samples per cell.')
        h = m // s
        labels = equal_quantile_labels(scores[:, 0], h)
        allocation = np.full(h, s, dtype=int)
    elif variant in ('KMEANS-X', 'KMEANS-X2', 'BALANCED-KMEANS-X2'):
        s = 2 if variant.endswith('2') else 4
        if m % s:
            raise ValueError('Sample size must be divisible by target samples per cell.')
        h = m // s
        fit = KMeans(n_clusters=h, n_init=10, random_state=91317,
                     algorithm='lloyd', max_iter=150).fit(z)
        if variant == 'BALANCED-KMEANS-X2':
            labels, objective = balanced_assignment(z, fit.cluster_centers_)
            for _ in range(3):
                centers = np.array([z[labels == j].mean(axis=0) for j in range(h)])
                labels, objective = balanced_assignment(z, centers)
            allocation = np.full(h, 2, dtype=int)
        else:
            labels = _merge_tiny_kmeans_clusters(z, fit.labels_)
            counts = np.bincount(labels)
            allocation = proportional_allocation(counts, m)
            extra['merged_clusters'] = int(h - len(counts))
    elif variant.startswith('BUD-'):
        s = 4 if variant.endswith('4') else 2
        if m % s:
            raise ValueError('Sample size must be divisible by samples per cell.')
        h = m // s
        if n % h:
            raise ValueError('Require N divisible by H for equal BUD capacities.')
        if variant == 'BUD-sliced2':
            if q == 1:
                labels = equal_quantile_labels(scores[:, 0], h)
                extra['coarse_slabs'] = h
            else:
                if h % 4:
                    raise ValueError('Sliced BUD requires H divisible by four.')
                slabs = h // 4
                coarse = equal_quantile_labels(scores[:, 0], slabs)
                design = glp_uniform_design(4, q-1)
                labels = np.empty(n, dtype=int)
                costs = []
                for b in range(slabs):
                    ix = np.flatnonzero(coarse == b)
                    coords = scores[ix, 1:q]
                    anchors = inverse_quantile_anchors(coords, design)
                    local, cost = balanced_assignment(coords, anchors)
                    labels[ix] = 4*b + local
                    costs.append(cost)
                objective = float(np.mean(costs))
                extra['coarse_slabs'] = slabs
        else:
            design = glp_uniform_design(h, q)
            if variant in ('BUD-rank2', 'BUD-rank4'):
                coords = np.column_stack([(rankdata(scores[:, j])-.5)/n for j in range(q)])
                anchors = design
            elif variant in ('BUD-inverse2', 'BUD-inverse4', 'BUD-lloyd2'):
                coords = scores[:, :q]
                anchors = inverse_quantile_anchors(coords, design)
            else:
                raise ValueError(f'Unknown BUD variant: {variant}')
            labels, objective = balanced_assignment(coords, anchors)
            if variant == 'BUD-lloyd2':
                extra['within_pc_before_lloyd'] = _within_scatter(coords, labels)
                centers = np.array([coords[labels == j].mean(axis=0) for j in range(h)])
                labels, objective = balanced_assignment(coords, centers)
                extra['within_pc_after_lloyd'] = _within_scatter(coords, labels)
            extra['ud_mixture_discrepancy'] = float(qmc.discrepancy(design, method='MD'))
        allocation = np.full(h, s, dtype=int)
    else:
        raise ValueError(f'Unknown design: {variant}')

    sizes = np.bincount(labels)
    if allocation.sum() != m or len(sizes) != len(allocation):
        raise RuntimeError('Allocation failed budget/label consistency.')
    if np.any(allocation < 2) or np.any(allocation > sizes):
        raise RuntimeError('Illegal stratum sample sizes.')
    diagnostics = dict(
        variant=variant, N=n, m=m, p=p, q=q, H=len(sizes),
        eigenvalue_1=float(eigenvalues[0]),
        retained_fraction=float(eigenvalues[:q].sum()/max(eigenvalues.sum(),1e-30)),
        min_population_cell=int(sizes.min()), max_population_cell=int(sizes.max()),
        min_sample_cell=int(allocation.min()), max_sample_cell=int(allocation.max()),
        within_full_x=_within_scatter(z, labels),
        within_retained_pc=_within_scatter(scores[:, :q], labels),
        assignment_cost=objective, seconds=perf_counter()-start,
        **extra,
    )
    return labels, allocation, diagnostics


def design_variance(scores, labels, allocation):
    """Exact finite-population design variance of a stratified mean of scores."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    allocation = np.asarray(allocation, dtype=int)
    n = len(scores)
    result = 0.0
    for h, sh in enumerate(allocation):
        vh = scores[labels == h]
        nh = len(vh)
        result += (nh/n)**2 * (1-sh/nh) * np.var(vh, ddof=1) / sh
    return float(result)
