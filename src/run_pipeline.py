"""End-to-end pipeline: data -> normalisation -> blocking -> candidate filter -> matcher -> output files.

Usage (from code/business_entity_resolution/):
    python src/run_pipeline.py --data-dir /path/to/student_resource                 # full run
    python src/run_pipeline.py --data-dir /path/to/student_resource --nrows 200000  # quick trial on the first rows
    python src/run_pipeline.py --data-dir ... --models runs/<run_id>/models.pkl     # skip training, reuse models

Writes <package>/output/matching_results.tsv and candidate_pairs.tsv, validates them, and keeps the outputs, log,
settings, scores and trained models in runs/<run_id>/ (version history).
"""
import argparse
import dataclasses
import datetime
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config                                                              # noqa: E402
from data_io import read_tsv, parse_ids, write_outputs, check_outputs, run_official_validator, count_lines  # noqa
from records import Records, CountryCodes, load_source, make_pool                      # noqa: E402
from blocking import CandidateIndex                                                    # noqa: E402
from features import stage2_features, stage3_features                                 # noqa: E402
from model import (lgb_params, train_cv, predict_avg, choose_filter_threshold,         # noqa: E402
                   filter_mask, macro_f05, decision_inputs, apply_rule, tune_decision)


# ------------------------------------------------------------------------------------------------ helpers
class Logger:
    """Prints timestamped progress lines (with free RAM) and appends them to runs/<run_id>/log.txt."""

    def __init__(self, path):
        self.path, self.t0 = path, time.time()

    def __call__(self, msg=''):
        line = f'[{(time.time() - self.t0) / 60:6.1f} min | RAM free {ram_free()}] {msg}'
        print(line, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def ram_free():
    """Available RAM in GB (psutil if installed, else /proc/meminfo, else '?')."""
    try:
        import psutil
        return f'{psutil.virtual_memory().available / 1e9:.1f}GB'
    except Exception:
        try:
            info = dict(line.split(':', 1) for line in open('/proc/meminfo'))
            return f"{int(info['MemAvailable'].split()[0]) / 1e6:.1f}GB"
        except Exception:
            return '?'


def parse_args():
    """Build the command line from the Config dataclass (every field becomes --field-name)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in dataclasses.fields(Config):
        ap.add_argument('--' + f.name.replace('_', '-'), type=type(f.default), default=f.default)
    cfg = Config(**vars(ap.parse_args()))
    if cfg.workers <= 0:
        cfg.workers = max(1, min(6, (os.cpu_count() or 2) - 1))
    return cfg


def find_data_root(path):
    """Return the folder that contains dataset/train/train_source1.tsv (searching below `path`)."""
    path = os.path.abspath(os.path.expanduser(path))
    for dp, _, files in os.walk(path):
        if 'train_source1.tsv' in files and os.path.basename(dp) == 'train':
            return os.path.dirname(os.path.dirname(dp))
    raise SystemExit(f'Could not find dataset/train/train_source1.tsv under {path} - check --data-dir '
                     '(in Git Bash use forward slashes: C:/Users/...)')


def add_frequencies(L, R):
    """Per-100k frequency of each record's core name / address among Source 2+3 (chains, shared buildings)."""
    scale = np.float32(1e5 / max(len(R), 1))
    for h, name in (('hn', 'nfreq'), ('ha', 'afreq')):
        uniq, cnt = np.unique(R.num[h], return_counts=True)
        for D in (L, R):
            pos = np.clip(np.searchsorted(uniq, D.num[h]), 0, len(uniq) - 1)
            D.num[name] = np.where(uniq[pos] == D.num[h], cnt[pos], 0).astype(np.float32) * scale


def load_split(folder, prefix, cfg, countries, pool, log, s1_filter=None):
    """Stream-load and normalise the three sources of a split -> (Source-1 Records, Source-2+3 Records)."""
    L = load_source(os.path.join(folder, f'{prefix}_source1.tsv'), 1, cfg, countries, pool, s1_filter)
    log(f'{prefix} Source 1 loaded: {len(L):,} records')
    R2 = load_source(os.path.join(folder, f'{prefix}_source2.tsv'), 2, cfg, countries, pool)
    log(f'{prefix} Source 2 loaded: {len(R2):,} records')
    R3 = load_source(os.path.join(folder, f'{prefix}_source3.tsv'), 3, cfg, countries, pool)
    log(f'{prefix} Source 3 loaded: {len(R3):,} records')
    R = Records.concat([R2, R3])
    del R2, R3
    add_frequencies(L, R)
    return L, R


def batched(n, size):
    """(start, stop) ranges covering 0..n."""
    return [(s, min(s + size, n)) for s in range(0, n, size)]


def rows_by_l(l, lo, hi):
    """Slice of pairs (sorted by l) whose Source-1 row is in [lo, hi)."""
    return slice(np.searchsorted(l, lo), np.searchsorted(l, hi))


# ------------------------------------------------------------------------------------------------ training
def train(cfg, train_dir, countries, pool, log, run_dir):
    """Train the candidate filter and the matcher on a sample of Source-1 entities; returns the model bundle."""
    gt = read_tsv(os.path.join(train_dir, 'train_ground_truth.tsv'), nrows=cfg.nrows or None)
    ids = gt['source1_entity_id'].str.strip().to_numpy()
    rng = np.random.RandomState(cfg.seed)
    pick = np.sort(rng.choice(len(ids), min(cfg.max_train_s1, len(ids)), replace=False))
    truth = {ids[i]: parse_ids(gt['matched_entity_ids'].iat[i]) for i in pick}
    log(f'ground truth: {len(ids):,} Source-1 entities, training on a random sample of {len(truth):,}')
    del gt, ids

    L, R = load_split(train_dir, 'train', cfg, countries, pool, log, s1_filter=set(truth))
    index = CandidateIndex(L, R, cfg, log)
    nR = len(R)

    # labels: map the sample's true Source-2/3 ids to row numbers by scanning the id column once
    wanted = {m for ms in truth.values() for m in ms}
    row_of = {}
    for lo, hi in batched(nR, 1_000_000):
        for j, x in enumerate(R.ids.take(np.arange(lo, hi))):
            if x in wanted:
                row_of[x] = lo + j
    l_ids = L.ids.take(np.arange(len(L)))
    n_true = np.array([len(truth.get(s, ())) for s in l_ids])
    true_keys = np.array(sorted(i * nR + row_of[m] for i, s in enumerate(l_ids) for m in truth.get(s, ()) if m in row_of),
                         dtype=np.int64)
    lost = int(n_true.sum()) - len(true_keys)
    tl, tr = true_keys // nR, true_keys % nR
    same_c = float(np.mean(L.num['country'][tl] == R.num['country'][tr])) if len(true_keys) else 1.0
    log(f'sample: {len(L):,} S1 | {int(n_true.sum()):,} true pairs ({lost} ids not found in S2/S3) | singletons '
        f'{np.mean(n_true == 0):.1%} | true pairs with the same country label {same_c:.4f}')

    P1 = index.query(np.arange(len(L)), cfg)
    y1 = np.isin(P1['l'].to_numpy() * nR + P1['r'].to_numpy(), true_keys)
    n_pos = max(int(n_true.sum()), 1)
    st = {'train_stage1_per_s1': len(P1) / len(L), 'train_stage1_recall': y1.sum() / n_pos,
          'key_cap_name': index.caps['n'], 'key_cap_addr': index.caps['a']}
    log(f'STAGE 1 blocking: {len(P1):,} pairs ({st["train_stage1_per_s1"]:.1f} per S1) | recall '
        f'{st["train_stage1_recall"]:.4f} | name view alone {y1[(P1.m.to_numpy() & 1) > 0].sum() / n_pos:.4f} | '
        f'address view alone {y1[(P1.m.to_numpy() & 2) > 0].sum() / n_pos:.4f}')

    lcol = P1['l'].to_numpy()
    X2 = pd.concat([stage2_features(L, R, P1.iloc[rows_by_l(lcol, lo, hi)].reset_index(drop=True))
                    for lo, hi in batched(len(L), cfg.test_batch_s1)], ignore_index=True)
    FEATS2 = list(X2.columns)
    oof2, models2 = train_cv(X2, y1, lcol, cfg.stage2_folds, lgb_params(cfg.seed, cfg.n_threads, True), 1000, log,
                             'stage2')
    thr = choose_filter_threshold(oof2, y1, lcol, cfg.prune_recall, cfg.prune_max_k)
    keep = filter_mask(oof2, lcol, thr, cfg.prune_max_k)
    P, X2k, p2, y = P1[keep].reset_index(drop=True), X2[keep].reset_index(drop=True), oof2[keep], y1[keep]
    del X2, P1, oof2, y1
    st.update({'filter_threshold': thr, 'train_final_per_s1': len(P) / len(L), 'train_final_recall': y.sum() / n_pos,
               'train_s1_no_candidates': float(np.mean(np.bincount(P['l'], minlength=len(L)) == 0))})
    log(f'STAGE 2 filter (threshold {thr:.4f}): {len(P):,} pairs ({st["train_final_per_s1"]:.2f} per S1) | recall '
        f'{st["train_final_recall"]:.4f} | S1 with an empty candidate list {st["train_s1_no_candidates"]:.1%}')

    X3 = stage3_features(L, R, P, X2k, p2)
    FEATS3 = list(X3.columns)
    oof3, models3 = train_cv(X3, y, P['l'].to_numpy(), cfg.folds, lgb_params(cfg.seed, cfg.n_threads), 5000, log,
                             'stage3')
    o2o = bool(cfg.one_to_one)
    best, table = tune_decision(P['l'].to_numpy(), P['r'].to_numpy(), oof3, y, n_true, len(L), o2o)
    lmax, rbest = decision_inputs(P['l'].to_numpy(), P['r'].to_numpy(), oof3)
    mask = apply_rule(oof3, lmax, rbest, best['t'], best['ratio'], o2o)
    f_oof, F = macro_f05(P['l'].to_numpy(), y, mask, n_true, len(L))
    f_ceil, _ = macro_f05(P['l'].to_numpy(), y, y.copy(), n_true, len(L))
    f_none, _ = macro_f05(P['l'].to_numpy(), y, np.zeros_like(y), n_true, len(L))
    cn = countries.names
    log(f'VALIDATION (out-of-fold) macro F0.5 = {f_oof:.4f}  [threshold {best["t"]}, ratio {best["ratio"]}, '
        f'one-to-one {o2o}] | ceiling with these candidates {f_ceil:.4f} | predict-nothing {f_none:.4f}')
    log(f'  singletons {F[n_true == 0].mean():.4f} | entities with matches {F[n_true > 0].mean():.4f} | per country: '
        + ', '.join(f'{cn[c]}={F[L.num["country"] == c].mean():.4f}' for c in np.unique(L.num['country'])))
    imp = pd.Series(np.mean([m.feature_importance('gain') for m in models3], axis=0), index=FEATS3)
    top = list(imp.sort_values(ascending=False).head(15).index)
    log('top features: ' + ', '.join(top))
    st.update({'validation_macro_f05': f_oof, 'ceiling_f05': f_ceil, 'predict_nothing_f05': f_none})
    bundle = {'stage2': models2, 'stage3': models3, 'filter_threshold': thr, 'decision': best, 'one_to_one': o2o,
              'features2': FEATS2, 'features3': FEATS3, 'stats': st, 'top_features': top, 'config': dataclasses.asdict(cfg)}
    with open(os.path.join(run_dir, 'models.pkl'), 'wb') as f:
        pickle.dump(bundle, f)
    log(f'models saved -> {os.path.join(run_dir, "models.pkl")} (reuse with --models if the test phase fails)')
    return bundle


# ------------------------------------------------------------------------------------------------ inference
def predict_test(cfg, test_dir, countries, pool, log, bundle):
    """Blocking, filtering and matching for every test Source-1 entity. Returns everything needed for output."""
    L, R = load_split(test_dir, 'test', cfg, countries, pool, log)
    if pool is not None:
        pool.shutdown()
    index = CandidateIndex(L, R, cfg, log)
    thr, FEATS2, FEATS3 = bundle['filter_threshold'], bundle['features2'], bundle['features3']
    Ps, X2s, p2s, n1 = [], [], [], 0
    for lo, hi in batched(len(L), cfg.test_batch_s1):
        P1 = index.query(np.arange(lo, hi), cfg)
        n1 += len(P1)
        if len(P1):
            X2 = stage2_features(L, R, P1)[FEATS2]
            p2 = predict_avg(bundle['stage2'], X2)
            k = filter_mask(p2, P1['l'].to_numpy(), thr, cfg.prune_max_k)
            Ps.append(P1[k].reset_index(drop=True)); X2s.append(X2[k].reset_index(drop=True)); p2s.append(p2[k])
        log(f'  test S1 {lo:,}-{hi:,} of {len(L):,}: {n1 / hi:.1f} -> {sum(map(len, Ps)) / hi:.2f} candidates per S1')
    del index
    P = pd.concat(Ps, ignore_index=True)
    X2 = pd.concat(X2s, ignore_index=True)
    p2 = np.concatenate(p2s) if p2s else np.empty(0, np.float32)
    del Ps, X2s, p2s
    lcol = P['l'].to_numpy()
    p3 = np.empty(len(P), np.float32)
    for lo, hi in batched(len(L), cfg.test_batch_s1):
        sl = rows_by_l(lcol, lo, hi)
        if sl.stop > sl.start:
            X3 = stage3_features(L, R, P.iloc[sl].reset_index(drop=True), X2.iloc[sl], p2[sl])[FEATS3]
            p3[sl] = predict_avg(bundle['stage3'], X3)
    best = bundle['decision']
    lmax, rbest = decision_inputs(lcol, P['r'].to_numpy(), p3)
    mask = apply_rule(p3, lmax, rbest, best['t'], best['ratio'], bundle['one_to_one'])
    n_pred = np.bincount(lcol[mask], minlength=len(L))
    st = {'test_stage1_per_s1': n1 / len(L), 'test_final_per_s1': len(P) / len(L), 'test_matches': int(mask.sum()),
          'test_share_with_match': float((n_pred > 0).mean()),
          'test_s1_no_candidates': float(np.mean(np.bincount(lcol, minlength=len(L)) == 0))}
    log(f'TEST: {st["test_stage1_per_s1"]:.1f} -> {st["test_final_per_s1"]:.2f} candidates per S1 | '
        f'{mask.sum():,} matches | S1 with >=1 match {st["test_share_with_match"]:.1%}')
    for c in np.unique(L.num['country']):
        sel = L.num['country'] == c
        log(f'    country {countries.names[c]!r}: {sel.sum():,} S1 | with >=1 match {(n_pred[sel] > 0).mean():.1%}')
    return L, R, P, p3, mask, st


# ------------------------------------------------------------------------------------------------ main
def main():
    cfg = parse_args()
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_root = os.path.dirname(os.path.dirname(code_dir))
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ('_trial' if cfg.nrows else '')
    run_dir = os.path.join(cfg.runs_dir or os.path.join(code_dir, 'runs'), run_id)
    os.makedirs(run_dir, exist_ok=True)
    out_dir = os.path.join(run_dir, 'output') if cfg.nrows else (cfg.out_dir or os.path.join(pkg_root, 'output'))
    log = Logger(os.path.join(run_dir, 'log.txt'))

    data_root = find_data_root(cfg.data_dir or '.')
    train_dir, test_dir = os.path.join(data_root, 'dataset', 'train'), os.path.join(data_root, 'dataset', 'test')
    log(f'run {run_id} | data {data_root} | workers {cfg.workers}' + (f' | TRIAL on first {cfg.nrows:,} rows' if cfg.nrows else ''))
    for d in (train_dir, test_dir):
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.tsv'):
                p = os.path.join(d, fn)
                log(f'  {fn:28s} {count_lines(p) - 1:>12,} rows  {os.path.getsize(p) / 1e6:8.1f} MB')

    countries = CountryCodes()
    pool = make_pool(cfg)
    if cfg.models:
        with open(cfg.models, 'rb') as f:
            bundle = pickle.load(f)
        log(f'loaded trained models from {cfg.models} (training skipped)')
    else:
        bundle = train(cfg, train_dir, countries, pool, log, run_dir)

    L, R, P, p3, mask, st_test = predict_test(cfg, test_dir, countries, pool, log, bundle)

    lcol, rcol = P['l'].to_numpy(), P['r'].to_numpy()
    order = np.lexsort((-p3, lcol))
    r_ids = R.ids.take(rcol[order])
    cand, match = {}, {}
    for li, rid, mk in zip(lcol[order].tolist(), r_ids, mask[order].tolist()):
        cand.setdefault(li, []).append(rid)
        if mk:
            match.setdefault(li, []).append(rid)
    s1_ids = L.ids.take(np.arange(len(L)))
    del R, P
    mp, cp = write_outputs(s1_ids, cand, match, out_dir)
    check_outputs(mp, cp, s1_ids)
    log(f'local format check: PASS -> {mp}')
    if not cfg.nrows:
        log('official validator: ' + run_official_validator(data_root, mp, cp))
        shutil.copytree(out_dir, os.path.join(run_dir, 'output'), dirs_exist_ok=True)

    st = dict(bundle['stats'], **st_test)
    info = {'run_id': run_id, 'trial_rows': cfg.nrows,
            'validation_macro_f05': round(float(st.get('validation_macro_f05', float('nan'))), 5),
            'decision': {'t': bundle['decision']['t'], 'ratio': bundle['decision']['ratio'],
                         'one_to_one': bundle['one_to_one']},
            'stats': {k: round(float(v), 6) for k, v in st.items()}, 'config': dataclasses.asdict(cfg),
            'top_features': bundle['top_features'], 'public_leaderboard': None}
    with open(os.path.join(run_dir, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    with open(os.path.join(run_dir, 'environment.txt'), 'w') as f:
        f.write(environment())
    with open(os.path.join(run_dir, 'methodology_draft.md'), 'w') as f:
        f.write(methodology(cfg, st, bundle, len(bundle['features3'])))
    log(f'run history saved in {run_dir}')
    log('TRIAL finished - pipeline works; now run without --nrows for the real submission.' if cfg.nrows
        else f'DONE. Upload {mp} to the portal.')


def environment():
    """Exact versions used in this run, in requirements.txt format."""
    from importlib.metadata import version, PackageNotFoundError
    lines = [f'# python {sys.version.split()[0]}']
    for pkg in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'rapidfuzz', 'lightgbm', 'psutil'):
        try:
            lines.append(f'{pkg}=={version(pkg)}')
        except PackageNotFoundError:
            pass
    return '\n'.join(lines) + '\n'


def methodology(cfg, st, bundle, n_feats):
    """Auto-generated write-up with this run's numbers, to paste into Documentation_template.md."""
    best, g = bundle['decision'], st.get
    return f"""# Methodology (auto-generated draft - paste into Documentation_template.md)

## Methodology
Three-stage supervised entity resolution built to scale to ~12M records per split on a laptop:
(1) streaming normalisation into a compact store, (2) candidate generation with a hashed, frequency-capped inverted
index followed by a lightweight learned candidate filter, (3) a gradient-boosted matcher whose decision rule is tuned
directly for the official macro F0.5 (singletons included) on out-of-fold predictions.

## Normalisation
Accent stripping; lower-casing; '&' -> 'and'; abbreviation canonicalisation for names (Corporation/Corp,
Private/Pvt, Limited/Ltd, Technologies/Tech ...) and addresses (Road/Rd, Street/St, Nagar/Ngr, French street words
...); legal-suffix extraction (Inc, LLC, Pvt, Ltd, SARL, SAS ...); DBA / trade-name alias splitting; landmark phrase
removal (Near/Opp/Behind ...); postal code (ZIP / PIN incl. '700 032' / code postal) and house-number extraction;
a transliteration-tolerant consonant skeleton (Lakshmi ~ Laxmi, Shree ~ Shri). Country is an open set of labels,
used only as an agreement feature (never one-hot), so the unseen France label is handled like any other.
Text is stored as UTF-8 blobs with offsets (no per-string Python objects) and decoded only for candidate pairs.

## Candidate generation / blocking
Stage 1 - hashed inverted index (2^22 hashed keys). Name keys: tokens, skeleton tokens, bigrams, compact name, and
composite keys token@postal-code and token#house-number (these stay rare even for very common names). Address keys:
tokens, bigrams, postal code, postal code + house number. Keys are IDF-weighted on Source 2+3 and keys carried by
more than {g('key_cap_name')} (name) / {g('key_cap_addr')} (address) records are dropped, so the sparse product
A @ B^T only links records sharing a rare key - no all-pairs comparison, near-linear cost. Top-{cfg.k_name} (name)
and top-{cfg.k_addr} (address) neighbours per Source-1 record are unioned.
Train sample: {g('train_stage1_per_s1', 0):.1f} candidates per S1, recall {g('train_stage1_recall', 0):.4f}.

Stage 2 - learned candidate filter: a small LightGBM model on cheap features (index cosines, RapidFuzz token-set /
ratio on name, skeleton and address, postal agreement, rank and gap to the best candidate) keeps at most
{cfg.prune_max_k} candidates per S1 with score >= {g('filter_threshold', 0):.4f}; the threshold is set on
out-of-fold predictions to keep {cfg.prune_recall:.1%} of the true pairs found by stage 1. Its output is
candidate_pairs.tsv, exactly the set the matcher scores.
Train sample: {g('train_final_per_s1', 0):.2f} candidates per S1, recall {g('train_final_recall', 0):.4f}.
Test: {g('test_stage1_per_s1', 0):.1f} -> {g('test_final_per_s1', 0):.2f} candidates per S1;
{g('test_s1_no_candidates', 0):.1%} of S1 entities get an empty list.

## Model architecture and feature engineering
Matcher: LightGBM, {cfg.folds}-fold GroupKFold grouped by Source-1 entity, early stopping, test scores averaged
over the fold models; trained on a random sample of {cfg.max_train_s1:,} Source-1 entities blocked against the full
Source 2+3 pool (so negatives are realistic). {n_feats} features per pair: RapidFuzz ratio / partial / token-sort /
token-set / Jaro-Winkler / Levenshtein on name, compact name, skeleton and address; token Jaccard and containment;
acronym match; DBA-alias similarity; legal-suffix agreement; postal code state and prefix; house-number agreement /
conflict; landmark flags; name and address frequency (chains, shared buildings); index cosines; the stage-2 score;
rank / gap-to-best features within each Source-1 entity's candidates.
Most important features: {', '.join(bundle['top_features'])}.

## Decision rule and results
Match if p >= {best['t']:.2f} and p >= {best['ratio']:.1f} x (best p of that S1){', and the pair is the best Source-1 entity for its Source-2/3 record (one-to-one, since Source 1 is deduplicated)' if bundle['one_to_one'] else ''}.
Tuned on out-of-fold predictions with the official per-entity macro F0.5 (matches lost in blocking/filtering count
as misses). Validation macro F0.5 = {g('validation_macro_f05', 0):.4f} (ceiling with these candidates
{g('ceiling_f05', 0):.4f}; predict-nothing baseline {g('predict_nothing_f05', 0):.4f}).

## Other
Only the provided files are used - no external data, APIs, geocoding or lookups, no pretrained models.
Libraries: scikit-learn, SciPy, NumPy, pandas, RapidFuzz (MIT), LightGBM (MIT).
"""


if __name__ == '__main__':
    main()
