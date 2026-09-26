"""Model training, the candidate filter threshold, the official metric and the match decision rule."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False


def lgb_params(seed, n_threads, small=False):
    """LightGBM parameters; `small` is the lighter model used for the stage-2 candidate filter."""
    if small:
        return dict(objective='binary', learning_rate=0.08, num_leaves=63,
                    min_child_samples=15, feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
                    lambda_l2=0.5, lambda_l1=0.1,
                    seed=seed, verbose=-1, num_threads=n_threads)
    return dict(objective='binary', learning_rate=0.03, num_leaves=127,
                min_child_samples=10, feature_fraction=0.9, bagging_fraction=0.85, bagging_freq=1,
                lambda_l2=0.3, lambda_l1=0.1, max_depth=8,
                seed=seed, verbose=-1, num_threads=n_threads)


def catboost_params(seed, n_threads):
    return dict(iterations=5000, learning_rate=0.03, depth=8, l2_leaf_reg=3,
                random_seed=seed, verbose=0, eval_metric='Logloss',
                early_stopping_rounds=150, thread_count=n_threads if n_threads > 0 else -1)


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


def train_cv_catboost(X, y, groups, n_folds, seed, n_threads, log, name):
    """CatBoost GroupKFold CV. Returns OOF probabilities and fold models. No-op if catboost not installed."""
    if not _HAS_CATBOOST:
        log(f'  {name}: catboost not installed, skipping')
        return np.zeros(len(X), np.float32), []
    oof = np.zeros(len(X), np.float32)
    models = []
    y_int = y.astype(np.int32)
    params = catboost_params(seed, n_threads)
    for fold, (a, b) in enumerate(GroupKFold(n_splits=n_folds).split(X, y_int, groups)):
        m = CatBoostClassifier(**params)
        m.fit(X.iloc[a], y_int[a], eval_set=(X.iloc[b], y_int[b]), verbose=False)
        oof[b] = m.predict_proba(X.iloc[b])[:, 1].astype(np.float32)
        models.append(m)
        pos, neg = oof[b][y_int[b] == 1], oof[b][y_int[b] == 0]
        log(f'  {name} catboost fold {fold}: {m.best_iteration_} trees | mean p true pairs '
            f'{pos.mean() if len(pos) else float("nan"):.3f} | non-matches {neg.mean() if len(neg) else float("nan"):.4f}')
    return oof, models


def predict_avg(models, X):
    """Mean probability of the LightGBM fold models."""
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0).astype(np.float32)


def predict_catboost(models, X):
    """Mean probability of the CatBoost fold models."""
    if not models:
        return np.zeros(len(X), np.float32)
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0).astype(np.float32)


def ensemble_predict(bundle, X):
    """Weighted average of LightGBM multi-seed ensemble and CatBoost (50/50 when both present)."""
    p_lgb = predict_avg(bundle['stage3'], X)
    cb_models = bundle.get('stage3_catboost', [])
    if cb_models:
        p_cb = predict_catboost(cb_models, X)
        return (0.5 * p_lgb + 0.5 * p_cb).astype(np.float32)
    return p_lgb


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
    """Grid-search (threshold, ratio) on out-of-fold scores for the best macro F0.5.
    Fine-grained grid (0.005 step) with wider ratio range for better F0.5 precision optimisation."""
    lmax, rbest = decision_inputs(l, r, p)
    res = []
    for ratio in (0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
        for t in np.round(np.arange(0.05, 0.99, 0.005), 3):
            f, _ = macro_f05(l, lab, apply_rule(p, lmax, rbest, t, ratio, o2o), n_true, n_left)
            res.append((f, float(t), ratio, o2o))
    res = pd.DataFrame(res, columns=['f05', 't', 'ratio', 'o2o']).sort_values('f05', ascending=False)
    return res.iloc[0].to_dict(), res
