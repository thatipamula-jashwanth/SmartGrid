import numpy as np


def _validate_feature_inputs(feature_indices, selected_scores):
    if feature_indices is None or selected_scores is None:
        raise ValueError("feature_indices and selected_scores cannot be None.")

    fi = np.asarray(feature_indices)
    sc = np.asarray(selected_scores)

    if fi.shape != sc.shape:
        raise ValueError(f"Length mismatch: {fi.shape=} vs {sc.shape=}.")

    if not np.issubdtype(fi.dtype, np.integer):
        raise TypeError("feature_indices must be integers.")

    if not np.issubdtype(sc.dtype, np.number):
        raise TypeError("selected_scores must be numeric.")

    if not np.isfinite(sc).all():
        raise ValueError("selected_scores must be finite.")


def split_into_axes(
    feature_indices,
    selected_scores,
    X=None,
    n_axes=3,
    *,
    max_features_corr=512,
    score_clip=5.0,
    X_standardized=False,
    cached_stats=None,
):
    

    _validate_feature_inputs(feature_indices, selected_scores)

    fi = np.asarray(feature_indices, dtype=np.int32)
    sc = np.asarray(selected_scores, dtype=np.float32)

    if fi.size == 0:
        return [[] for _ in range(n_axes)], [[] for _ in range(n_axes)]

    n_axes = min(n_axes, fi.size)

    order = np.argsort(sc)[::-1]
    fi = fi[order]
    sc = sc[order]

    groups = [[] for _ in range(n_axes)]
    scores = [[] for _ in range(n_axes)]

    if X is None:
        axis_load = np.zeros(n_axes, dtype=np.float32)
        for f, s in zip(fi, sc):
            ax = np.argmin(axis_load)
            groups[ax].append(int(f))
            scores[ax].append(float(s))
            axis_load[ax] += abs(s)
        return groups, scores

    Xf = X[:, fi].astype(np.float32, copy=False)

    if not X_standardized:
        if cached_stats is None:
            mean = Xf.mean(axis=0)
            std = Xf.std(axis=0)
            std[std < 1e-8] = 1.0
        else:
            mean, std = cached_stats
        Xs = (Xf - mean) / std
    else:
        Xs = Xf

    Xs = np.ascontiguousarray(Xs)

    N, F = Xs.shape

    if F > max_features_corr:
        axis_load = np.zeros(n_axes, dtype=np.float32)
        for f, s in zip(fi, sc):
            ax = np.argmin(axis_load)
            groups[ax].append(int(f))
            scores[ax].append(float(s))
            axis_load[ax] += abs(s)
        return groups, scores

    denom = max(N - 1, 1)
    corr = (Xs.T @ Xs) / denom
    corr = np.abs(corr)
    np.fill_diagonal(corr, 0.0)

    assigned = np.zeros(F, dtype=bool)

    axis_members = [[] for _ in range(n_axes)]

    axis_corr_means = [None] * n_axes
    axis_sizes = [0] * n_axes

    for ax in range(n_axes):
        remaining = np.where(~assigned)[0]
        if remaining.size == 0:
            break

        idx = remaining[np.argmax(sc[remaining])]
        assigned[idx] = True

        axis_members[ax].append(idx)
        axis_corr_means[ax] = corr[:, idx].copy()
        axis_sizes[ax] = 1

        groups[ax].append(int(fi[idx]))
        scores[ax].append(float(sc[idx]))

    scale = np.percentile(sc, 75) + 1e-6
    sc_norm = np.clip(sc / scale, 0.0, score_clip)

    axis_score_sums = [0.0] * n_axes
    axis_score_means = [0.0] * n_axes

    for ax in range(n_axes):
        if axis_members[ax]:
            idx = axis_members[ax][0]
            axis_score_sums[ax] = sc_norm[idx]
            axis_score_means[ax] = sc_norm[idx]

    remaining = np.where(~assigned)[0]

    for i in remaining:
        best_ax = 0
        best_score = -np.inf

        for ax in range(n_axes):
            m = axis_corr_means[ax]
            if m is None:
                continue

            score = m[i] * axis_score_means[ax]
            if score > best_score:
                best_score = score
                best_ax = ax

        assigned[i] = True
        axis_members[best_ax].append(i)

        k = axis_sizes[best_ax]
        axis_corr_means[best_ax] = (m * k + corr[:, i]) / (k + 1)
        axis_sizes[best_ax] = k + 1

        axis_score_sums[best_ax] += sc_norm[i]
        axis_score_means[best_ax] = (
            axis_score_sums[best_ax] / axis_sizes[best_ax]
        )

        groups[best_ax].append(int(fi[i]))
        scores[best_ax].append(float(sc[i]))

    return groups, scores


def compute_axes(X, groups, group_scores):
    if len(groups) != len(group_scores):
        raise ValueError("groups and group_scores length mismatch.")

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be 2D.")

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
            raise IndexError(f"Axis {axis_i} has invalid indices.")

        w = np.asarray(group_scores[axis_i], dtype=np.float32)
        if w.shape != g_idx.shape:
            raise ValueError("Weight / index mismatch.")

        w = np.sign(w) * np.sqrt(np.abs(w))
        w /= (np.sum(np.abs(w)) + 1e-9)

        axis_vals[:, axis_i] = X[:, g_idx] @ w

    return axis_vals
