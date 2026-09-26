"""Model training, the candidate filter threshold, the official metric and the match decision rule."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold


def lgb_params(seed, n_threads, small=False):
    """LightGBM parameters; `small` is the lighter model used for the stage-2 candidate filter."""
    return dict(objective='binary', learning_rate=0.1 if small else 0.05, num_leaves=31 if small else 63,
                min_child_samples=20, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                seed=seed, verbose=-1, num_threads=n_threads)


def train_cv(X, y, groups, n_folds, params, rounds, log, name):
    """GroupKFold (grouped by Source-1 entity, so no entity leaks across folds) with early stopping.

    Returns out-of-fold probabilities and the list of fold models (averaged at test time).
    """
    oof = np.zeros(len(X), np.float32)
    models = []
    y = y.astype(np.int32)
    for fold, (a, b) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        dtr = lgb.Dataset(X.iloc[a], y[a])
        dva = lgb.Dataset(X.iloc[b], y[b], reference=dtr)
        m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
        models.append(m)
        pos, neg = oof[b][y[b] == 1], oof[b][y[b] == 0]
        log(f'  {name} fold {fold}: {m.best_iteration} trees | mean p true pairs '
            f'{pos.mean() if len(pos) else float("nan"):.3f} | non-matches {neg.mean() if len(neg) else float("nan"):.4f}')
    return oof, models


def predict_avg(models, X):
    """Mean probability of the fold models."""
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)


def rank_within(l, p):
    """1-based rank of each pair's score among the candidates of the same Source-1 record."""
    return pd.Series(p).groupby(l).rank(ascending=False, method='first').to_numpy()


def choose_filter_threshold(p, y, l, target_recall, max_k):
    """Largest stage-2 threshold that keeps `target_recall` of the true pairs found by stage 1 (top-`max_k` cap)."""
    rk = rank_within(l, p)
    pos = np.sort(p[(y == 1) & (rk <= max_k)])[::-1]
    need = int(np.ceil(target_recall * max(int(y.sum()), 1)))
    if len(pos) == 0:
        return 0.0
    return float(pos[min(need, len(pos)) - 1])


def filter_mask(p, l, thr, max_k):
    """Stage-2 survivors: score >= thr and within the top-max_k of its Source-1 record."""
    return (p >= thr) & (rank_within(l, p) <= max_k)


def macro_f05(l_idx, lab, mask, n_true, n_left):
    """Official metric: F0.5 per Source-1 entity, macro-averaged, singletons included.

    n_true counts ALL true matches of each entity (including those lost in blocking/filtering).
    An entity with no true matches scores 1 if nothing is predicted, else 0.
    """
    n_pred = np.bincount(l_idx[mask], minlength=n_left)
    tp = np.bincount(l_idx[mask & lab], minlength=n_left)
    P_ = np.divide(tp, n_pred, out=np.zeros(n_left), where=n_pred > 0)
    R_ = np.divide(tp, n_true, out=np.zeros(n_left), where=n_true > 0)
    F = np.where(tp > 0, 1.25 * P_ * R_ / np.maximum(0.25 * P_ + R_, 1e-12), 0.0)
    F = np.where((n_true == 0) & (n_pred == 0), 1.0, F)
    return float(F.mean()), F


def decision_inputs(l, r, p):
    """Best score of each pair's Source-1 group, and whether the pair is the best Source-1 for its Source-2/3 record."""
    d = pd.DataFrame({'l': l, 'r': r, 'p': p})
    lmax = d.groupby('l')['p'].transform('max').to_numpy()
    rbest = p >= d.groupby('r')['p'].transform('max').to_numpy() - 1e-9
    return lmax, rbest


def apply_rule(p, lmax, rbest, t, ratio, o2o):
    """Match if p >= t and p >= ratio * best-of-group; optionally only the best Source-1 per Source-2/3 record."""
    mask = p >= t
    if ratio > 0:
        mask &= p >= ratio * lmax - 1e-9
    if o2o:
        mask &= rbest
    return mask


def tune_decision(l, r, p, lab, n_true, n_left, o2o):
    """Grid-search (threshold, ratio) on out-of-fold scores for the best macro F0.5 (one-to-one fixed)."""
    lmax, rbest = decision_inputs(l, r, p)
    res = []
    for ratio in (0.0, 0.5, 0.7, 0.8, 0.9, 1.0):
        for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
            f, _ = macro_f05(l, lab, apply_rule(p, lmax, rbest, t, ratio, o2o), n_true, n_left)
            res.append((f, float(t), ratio, o2o))
    res = pd.DataFrame(res, columns=['f05', 't', 'ratio', 'o2o']).sort_values('f05', ascending=False)
    return res.iloc[0].to_dict(), res
