import numpy as np

def _validate_feature_inputs(feature_indices, selected_scores):
    if feature_indices is None or selected_scores is None:
        raise ValueError("feature_indices and selected_scores cannot be None.")

    fi = np.asarray(feature_indices)
    sc = np.asarray(selected_scores)

    if fi.shape != sc.shape:
        raise ValueError(
            f"Length mismatch: {fi.shape=} vs {sc.shape=}."
        )

    if not np.issubdtype(fi.dtype, np.integer):
        raise TypeError("feature_indices must be integers.")

    if not np.issubdtype(sc.dtype, np.number):
        raise TypeError("selected_scores must be numeric.")

    if not np.isfinite(sc).all():
        raise ValueError("selected_scores must be finite (no NaN/inf).")


def split_into_axes(feature_indices, selected_scores, n_axes=3):
 
    _validate_feature_inputs(feature_indices, selected_scores)

    fi = np.asarray(feature_indices, dtype=np.int32)
    sc = np.asarray(selected_scores, dtype=np.float32)

    if fi.size == 0:
        return [[] for _ in range(n_axes)], [[] for _ in range(n_axes)]

    order = np.argsort(sc)[::-1]
    fi = fi[order]
    sc = sc[order]

    groups = [[] for _ in range(n_axes)]
    scores = [[] for _ in range(n_axes)]

    for i, (f, s) in enumerate(zip(fi, sc)):
        ax = i % n_axes
        groups[ax].append(int(f))
        scores[ax].append(float(s))

    return groups, scores


def compute_axes(X, groups, group_scores):

    if len(groups) != len(group_scores):
        raise ValueError("groups and group_scores length mismatch.")

    X = np.asarray(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features).")

    n_samples, n_features = X.shape
    n_axes = len(groups)

    axis_vals = np.zeros((n_samples, n_axes), dtype=np.float32)

    if n_samples == 0:
        return axis_vals

    for axis_i in range(n_axes):
        g = groups[axis_i]
        if not g:
            continue

        g_idx = np.asarray(g, dtype=np.int32)

        if (g_idx < 0).any() or (g_idx >= n_features).any():
            raise IndexError(
                f"Axis {axis_i} contains invalid feature indices: {g_idx}"
            )

        w = np.asarray(group_scores[axis_i], dtype=np.float32)

        if w.shape != g_idx.shape:
            raise ValueError(
                f"Axis {axis_i} weight mismatch: {w.shape=} vs {g_idx.shape=}"
            )

        if not np.isfinite(w).all():
            raise ValueError(
                f"Non-finite weights at axis {axis_i}: {w}"
            )

        s = w.sum()
        if s <= 1e-9:
            w = np.full_like(w, 1.0 / w.size)
        else:
            w = w / s

        axis_vals[:, axis_i] = X[:, g_idx] @ w

    return axis_vals
