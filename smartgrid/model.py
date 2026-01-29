import numpy as np
import warnings

from .feature_selection import select_top_k_features
from .axis_mapping import split_into_axes
from .grid_builder import build_grid
from .predictor import predict_batch


class SmartGrid:
    def __init__(self, k_features=30, bins=20, radius=1, alpha=None, n_grids=2):
        self.k_features = k_features
        self.bins = bins
        self.radius = radius
        self.alpha = alpha
        self.n_grids = n_grids

        self.feature_idx = None
        self.feature_scores_full = None

        self._projs = []
        self._ranges_list = []

        self.majority_class = None
        self._n_features_train = None
        self._label_inverse = None
        self._dtype = np.float32
        self._fitted = False

    def fit(self, X, y):
        X = np.asarray(X, dtype=self._dtype)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must align.")

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y)

        self._n_features_train = X.shape[1]

        classes, y_encoded = np.unique(y, return_inverse=True)
        y = y_encoded.astype(np.int32)
        self._label_inverse = classes

        feature_idx, full_scores = select_top_k_features(X, y, self.k_features)
        full_scores = np.nan_to_num(full_scores, nan=0.0)

        valid = full_scores[feature_idx] != 0
        feature_idx = feature_idx[valid]
        selected_scores = full_scores[feature_idx]

        if feature_idx.size == 0:
            if np.all(full_scores == 0):
                best = 0
                score = 0.0
            else:
                best = int(np.argmax(full_scores))
                score = float(full_scores[best])

            feature_idx = np.array([best], dtype=np.int32)
            selected_scores = np.array([score], dtype=np.float32)

        order = np.argsort(selected_scores)[::-1]
        feature_idx = feature_idx[order]
        selected_scores = selected_scores[order]

        self.feature_idx = feature_idx
        self.feature_scores_full = full_scores

        uniq, counts = np.unique(y, return_counts=True)
        freq = counts.astype(np.float32)
        self.majority_class = int(uniq[np.argmax(freq)])

        if self.alpha is None:
            imbalance_ratio = freq.max() / (freq.min() + 1e-9)
            alpha = 0.5 + 0.8 * np.tanh(imbalance_ratio / 10.0)
            alpha = float(np.clip(alpha, 0.3, 1.2))
        else:
            alpha = float(self.alpha)

        class_weights = ((freq.max() / freq) ** alpha).astype(np.float32)
        class_weights = np.clip(class_weights, 0.1, 10.0)

        est_bytes = self.bins**3 * len(freq) * 4 * self.n_grids
        if est_bytes > 500_000_000:
            warnings.warn(
                f"SmartGrid may use ~{est_bytes / 1e6:.1f} MB of memory "
                f"(bins={self.bins}, n_grids={self.n_grids}).",
                RuntimeWarning,
            )

        self._projs.clear()
        self._ranges_list.clear()


        X_sel = X[:, feature_idx]

        local_idx = np.arange(feature_idx.size, dtype=np.int32)

        for _ in range(self.n_grids):
            groups_local, group_scores = split_into_axes(
                local_idx,
                selected_scores,
                X=X_sel,
                n_axes=3,
            )

            proj = np.zeros((3, self._n_features_train), dtype=np.float32)

            for ax in range(3):
                local = np.asarray(groups_local[ax], dtype=np.int32)
                w = np.asarray(group_scores[ax], dtype=np.float32)

                if w.size:
                    w /= (w.sum() + 1e-9)

                    global_idx = feature_idx[local]
                    proj[ax, global_idx] = w

            XYZ = X @ proj.T
            grid, ranges = build_grid(XYZ, y, bins=self.bins, r=self.radius)

            ranges["majority_class"] = self.majority_class
            ranges["class_weights"] = class_weights
            ranges["num_classes"] = len(freq)

            self._projs.append(proj)
            self._ranges_list.append(ranges)

        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict().")

        X = np.asarray(X, dtype=self._dtype)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self._n_features_train:
            raise ValueError("Feature mismatch.")

        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        n = X.shape[0]
        num_classes = self._ranges_list[0]["num_classes"]
        votes = np.zeros((n, num_classes), dtype=np.int32)

        for proj, ranges in zip(self._projs, self._ranges_list):
            XYZ = X @ proj.T
            preds = predict_batch(ranges, XYZ)

            preds = np.clip(preds, 0, num_classes - 1)
            np.add.at(votes, (np.arange(n), preds), 1)

        final = votes.argmax(axis=1)
        return self._label_inverse[final]
