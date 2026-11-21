import numpy as np
from sklearn.utils import resample, shuffle
import smote_variants as sv


def balance_with_smote(X, y, random_state=42):

    original_size = len(y)
    sampler = sv.Safe_Level_SMOTE(random_state=random_state)
    X_res, y_res = sampler.sample(X, y)
    if len(y_res) > original_size:
        classes = np.unique(y_res)
        target_per_class = original_size // len(classes)
        remainder = original_size % len(classes)
        X_list = []
        y_list = []
        for i, cls in enumerate(classes):
            X_cls = X_res[y_res == cls]
            y_cls = y_res[y_res == cls]
            target = target_per_class + 1 if i < remainder else target_per_class
            if len(y_cls) > target:
                X_cls, y_cls = resample(X_cls, y_cls, replace=False, n_samples=target, random_state=random_state)
            elif len(y_cls) < target:
                X_cls, y_cls = resample(X_cls, y_cls, replace=True, n_samples=target, random_state=random_state)
            X_list.append(X_cls)
            y_list.append(y_cls)
        X_res = np.vstack(X_list)
        y_res = np.hstack(y_list)
    return X_res


