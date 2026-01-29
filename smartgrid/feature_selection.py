import numpy as np


def select_top_k_features(
    X,
    y,
    k,
    eps=1e-6,
    min_std=1e-8,
    max_score=1e6,
    class_loop_threshold=32,
):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be 2D.")

    n_samples, n_features = X.shape
    if y.shape[0] != n_samples:
        raise ValueError("y must match X.")

    if n_features == 0 or k <= 0:
        return np.array([], dtype=np.int32), np.zeros(n_features, dtype=np.float32)

    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = classes.size
    if n_classes < 2:
        raise ValueError("Requires at least 2 classes.")

    stds = X.std(axis=0)
    valid_mask = stds > min_std
    if not valid_mask.any():
        raise ValueError("All features near-constant.")

    Xv = X[:, valid_mask]
    kept_features = np.nonzero(valid_mask)[0]

    Xv = (Xv - Xv.mean(axis=0)) / (Xv.std(axis=0) + eps)

    n_kept = Xv.shape[1]

    class_counts = np.bincount(y_idx, minlength=n_classes).astype(np.float32)
    if np.any(class_counts == 0):
        raise ValueError("Empty class detected.")

    class_weights = 1.0 / class_counts
    class_weights /= class_weights.sum()

    class_sum = np.zeros((n_classes, n_kept), dtype=np.float32)
    class_sqsum = np.zeros((n_classes, n_kept), dtype=np.float32)

    if n_classes <= class_loop_threshold:
        for c in range(n_classes):
            Xc = Xv[y_idx == c]
            if Xc.size == 0:
                continue
            class_sum[c] = Xc.sum(axis=0)
            class_sqsum[c] = (Xc * Xc).sum(axis=0)
    else:
        np.add.at(class_sum, y_idx, Xv)
        np.add.at(class_sqsum, y_idx, Xv * Xv)

    class_means = class_sum / class_counts[:, None]
    overall_mean = class_sum.sum(axis=0) / n_samples

    diff = class_means - overall_mean
    between_var = (class_weights[:, None] * diff * diff).sum(axis=0)

    within_var = (
        class_weights[:, None]
        * (
            class_sqsum
            - 2.0 * class_means * class_sum
            + class_counts[:, None] * class_means * class_means
        )
    ).sum(axis=0)

    scores_kept = between_var / (within_var + eps)
    scores_rank = np.clip(scores_kept, 0.0, max_score)

    k = min(k, n_kept)
    idx = np.argpartition(scores_rank, -k)[-k:]

    order = np.lexsort((within_var[idx], -scores_rank[idx]))
    top_local = idx[order]
    top_features = kept_features[top_local]

    full_scores = np.zeros(n_features, dtype=np.float32)
    full_scores[kept_features] = scores_kept

    return top_features.astype(np.int32), full_scores
