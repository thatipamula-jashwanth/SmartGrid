import numpy as np
from .feature_selection import select_top_k_features
from .axis_mapping import split_into_axes
from .grid_builder import build_grid
from .predictor import predict_batch


class SmartGrid:
    def __init__(self, k_features=30, bins=20, radius=1, alpha=None):
        self.k_features = k_features
        self.bins = bins
        self.radius = radius
        self.alpha = alpha

        self.feature_idx = None
        self.feature_scores_full = None
        self.groups = None
        self.group_scores = None
        self.grid = None
        self.ranges = None
        self.majority_class = None

        self._proj = None
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

  
        X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
        y = np.nan_to_num(y)

        self._n_features_train = X.shape[1]

        classes, y_encoded = np.unique(y, return_inverse=True)
        y = y_encoded.astype(np.int32)
        self._label_inverse = classes


        feature_idx, full_scores = select_top_k_features(X, y, self.k_features)

  
        full_scores = np.nan_to_num(full_scores, nan=0.0, posinf=1e9, neginf=-1e9)

        valid_idx = np.where(full_scores[feature_idx] != 0)[0]
        feature_idx = feature_idx[valid_idx]
        selected_scores = full_scores[feature_idx]

       
        if len(feature_idx) == 0:
            feature_idx = np.array([np.argmax(full_scores)])
            selected_scores = np.array([full_scores[feature_idx[0]]])

        order = np.argsort(selected_scores)[::-1]
        feature_idx = feature_idx[order]
        selected_scores = selected_scores[order]

        self.feature_idx = feature_idx
        self.feature_scores_full = full_scores

        groups, group_scores = split_into_axes(feature_idx, selected_scores)
        self.groups = groups
        self.group_scores = group_scores


        proj = np.zeros((len(groups), self._n_features_train), dtype=np.float32)
        for ax in range(len(groups)):
            idx = np.asarray(groups[ax], dtype=np.int32)
            w = np.asarray(group_scores[ax], dtype=np.float32)
            if w.size:
                w /= w.sum() + 1e-9
                proj[ax, idx] = w
        self._proj = proj


        XYZ = X @ proj.T

 
        grid, ranges = build_grid(XYZ, y, bins=self.bins, r=self.radius)
        self.grid = grid
        self.ranges = ranges


        uniq, counts = np.unique(y, return_counts=True)
        freq = counts.astype(np.float32)
        inv_freq = 1.0 / (freq + 1e-9)
        self.majority_class = int(uniq[np.argmax(freq)])

        ranges["majority_class"] = self.majority_class
        ranges["freq"] = freq
        ranges["inv_freq"] = inv_freq

        if self.alpha is None:
            imbalance_ratio = freq.max() / (freq.min() + 1e-9)
            scaled = np.tanh(imbalance_ratio / 10.0)
            alpha = float(0.5 + 0.8 * scaled)
        else:
            alpha = float(self.alpha)

        ranges["alpha"] = alpha

        major = freq.max()
        ranges["class_weights"] = ((major / freq) ** alpha).astype(np.float32)

        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Must call fit() before predict().")

        X = np.asarray(X, dtype=self._dtype)
        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if X.shape[1] != self._n_features_train:
            raise ValueError(
                f"Prediction feature mismatch: got {X.shape[1]}, "
                f"expected {self._n_features_train}"
            )

        X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)

        XYZ = X @ self._proj.T

        preds = np.asarray(predict_batch(self.ranges, XYZ), dtype=np.int32)

        num_classes = self.ranges["num_classes"]
        invalid = (preds < 0) | (preds >= num_classes)
        preds[invalid] = self.majority_class

        return self._label_inverse[preds]
