"""Pairwise features.

stage2_features : cheap features for every stage-1 candidate (input of the learned candidate filter)
stage3_features : the full feature set for the filtered candidates (input of the matcher)
Text is decoded from the compact store only for the rows in the current batch.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

try:
    from rapidfuzz.process import cpdist        # vectorised, multi-threaded pairwise scoring (rapidfuzz >= 3.6)
except ImportError:
    cpdist = None


def S(scorer, a, b):
    """Element-wise similarity of two equal-length string lists."""
    if cpdist is not None:
        return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    return np.fromiter((scorer(x, y) for x, y in zip(a, b)), np.float32, len(a))


def tok_jaccard(a, b):
    """Jaccard of whitespace-token sets (NaN when both are empty)."""
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = set(x.split()), set(y.split())
        u = len(x | y)
        out[i] = len(x & y) / u if u else np.nan
    return out


def tok_overlap_min(a, b):
    """|A n B| / min(|A|, |B|): 1.0 when one token set contains the other."""
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = set(x.split()), set(y.split())
        out[i] = len(x & y) / min(len(x), len(y)) if (x and y) else np.nan
    return out


def trigram_jacc(a, b):
    """Character trigram Jaccard similarity (handles transliteration noise better than token metrics)."""
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        sx = set(x[j:j+3] for j in range(max(0, len(x)-2))) if len(x) >= 3 else set(x)
        sy = set(y[j:j+3] for j in range(max(0, len(y)-2))) if len(y) >= 3 else set(y)
        u = len(sx | sy)
        out[i] = len(sx & sy) / u if u else (1.0 if x == y else 0.0)
    return out


def _soundex(s):
    """Simple Soundex for a single token."""
    if not s:
        return ''
    s = s.upper()
    table = str.maketrans('BFPVCGJKQSXZDTLMNR', '111122222222334556')
    keep = s[0]
    coded = keep + s[1:].translate(table).replace('0', '')
    # collapse consecutive identical digits
    prev, out = '', keep
    for c in coded[1:]:
        if c != prev and c != keep:
            out += c
            if len(out) == 4:
                break
        prev = c
    return out.ljust(4, '0')[:4]


def soundex_match(a, b):
    """1.0 if first tokens share Soundex code, 0.5 if partial overlap, 0.0 otherwise."""
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        xt = x.split()
        yt = y.split()
        if not xt or not yt:
            out[i] = np.nan
            continue
        sx = {_soundex(t) for t in xt if len(t) > 1}
        sy = {_soundex(t) for t in yt if len(t) > 1}
        u = len(sx | sy)
        out[i] = len(sx & sy) / u if u else 0.0
    return out


def name_len_ratio(a, b):
    """min(len(a),len(b)) / max(len(a),len(b)) on compact (no-space) name. Flags length mismatches."""
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        cx, cy = x.replace(' ', ''), y.replace(' ', '')
        mn, mx = min(len(cx), len(cy)), max(len(cx), len(cy))
        out[i] = mn / mx if mx else 1.0
    return out


def postal_state(p1, p2):
    """0 both missing, 1 one missing, 2 equal, 3 one edit apart (typo), 4 different."""
    out = np.empty(len(p1), np.float32)
    for i, (x, y) in enumerate(zip(p1, p2)):
        out[i] = 0 if not (x or y) else 1 if not (x and y) else 2 if x == y else \
            3 if Levenshtein.distance(x, y) <= 1 else 4
    return out


def legal_rel(a, b):
    """Legal-suffix agreement: 0 none, 1 one side only, 2 identical, 3 overlapping, 4 conflicting (Inc vs LLC)."""
    out = np.empty(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        x, y = set(x.split()), set(y.split())
        out[i] = 0 if not (x or y) else 1 if not (x and y) else 2 if x == y else 3 if x & y else 4
    return out


def group_features(l, X, cols):
    """Rank and gap-to-best of each column among the candidates of the same Source-1 record."""
    G = pd.DataFrame({c: X[c] for c in cols})
    g = G.groupby(l, sort=False)
    out = {}
    for c in cols:
        out[f'{c}_rk_l'] = g[c].rank(ascending=False, method='min').to_numpy(np.float32)
        out[f'{c}_d_lmax'] = (G[c] - g[c].transform('max')).to_numpy(np.float32)
    out['n_cand_l'] = g[cols[0]].transform('size').to_numpy(np.float32)
    return out


STAGE2_GROUP = ['cos_n', 'cos_a', 'q_name_tset', 'q_addr_tset', 'q_score']


def stage2_features(L, R, P):
    """Cheap features for stage-1 pairs P (columns l, r, m, cos_n, cos_a). Returns a float32 DataFrame."""
    l, r = P['l'].to_numpy(), P['r'].to_numpy()
    m = P['m'].to_numpy()
    X = {'blk_n': ((m & 1) > 0).astype(np.float32), 'blk_a': ((m & 2) > 0).astype(np.float32),
         'cos_n': P['cos_n'].to_numpy(np.float32), 'cos_a': P['cos_a'].to_numpy(np.float32)}
    nl, nr = L.col('n_core', l), R.col('n_core', r)
    X['q_name_tset'] = S(fuzz.token_set_ratio, nl, nr)
    X['q_name_ratio'] = S(fuzz.ratio, nl, nr)
    X['q_skel_tset'] = S(fuzz.token_set_ratio, L.col('n_skel', l), R.col('n_skel', r))
    X['q_addr_tset'] = S(fuzz.token_set_ratio, L.col('a_core', l), R.col('a_core', r))
    X['q_postal'] = postal_state(L.col('postal', l), R.col('postal', r))
    X['q_same_country'] = (L.num['country'][l] == R.num['country'][r]).astype(np.float32)
    X['q_is_s3'] = (R.num['src'][r] == 3).astype(np.float32)
    n_empty = (L.num['n_ntok'][l] == 0) | (R.num['n_ntok'][r] == 0)
    a_empty = (L.num['a_ntok'][l] == 0) | (R.num['a_ntok'][r] == 0)
    for k in ('q_name_tset', 'q_name_ratio', 'q_skel_tset'):
        X[k][n_empty] = np.nan
    X['q_addr_tset'][a_empty] = np.nan
    with np.errstate(all='ignore'):
        X['q_score'] = np.nanmean(np.vstack([X['cos_n'], X['cos_a'], X['q_name_tset'] / 100,
                                             X['q_addr_tset'] / 100]), axis=0).astype(np.float32)
    X.update(group_features(l, X, STAGE2_GROUP))
    return pd.DataFrame(X)


STAGE3_GROUP = ['p2', 'cos_n', 'cos_a', 'n_tset', 'a_tset', 'hscore', 'n_trigram_jacc', 'n_alias_tset']


def stage3_features(L, R, P, X2, p2):
    """Full feature set for filtered pairs P given their stage-2 features X2 and stage-2 score p2."""
    l, r = P['l'].to_numpy(), P['r'].to_numpy()
    base = X2[[c for c in X2.columns if '_rk_' not in c and '_d_' not in c and c != 'n_cand_l']].reset_index(drop=True)
    X = {c: base[c].to_numpy(np.float32) for c in base.columns}
    X.update(_pairwise(L, R, l, r))
    X['p2'] = np.asarray(p2, np.float32)
    with np.errstate(all='ignore'):
        X['hscore'] = np.nanmean(np.vstack([X['cos_n'], X['cos_a'], X['n_tset'] / 100, X['a_tset'] / 100,
                                            X['n_trigram_jacc'], X['n_alias_tset'] / 100]),
                                 axis=0).astype(np.float32)
    X.update(group_features(l, X, STAGE3_GROUP))
    return pd.DataFrame(X)


def _pairwise(L, R, l, r):
    """Heavy per-pair string features."""
    n = len(l)
    X = {}
    a, b = L.col('n_core', l), R.col('n_core', r)
    X['n_ratio'] = S(fuzz.ratio, a, b)
    X['n_partial'] = S(fuzz.partial_ratio, a, b)
    X['n_tsort'] = S(fuzz.token_sort_ratio, a, b)
    X['n_tset'] = S(fuzz.token_set_ratio, a, b)
    X['n_jw'] = S(JaroWinkler.normalized_similarity, a, b)
    ca, cb = [x.replace(' ', '') for x in a], [y.replace(' ', '') for y in b]
    X['n_compact_lev'] = S(Levenshtein.normalized_similarity, ca, cb)
    X['n_compact_partial'] = S(fuzz.partial_ratio, ca, cb)
    X['n_jacc'] = tok_jaccard(a, b)
    X['n_overlap_min'] = tok_overlap_min(a, b)
    acr_a = [''.join(t[0] for t in x.split()) if ' ' in x else '' for x in a]
    acr_b = [''.join(t[0] for t in y.split()) if ' ' in y else '' for y in b]
    X['n_acronym'] = np.fromiter(((x != '' and x == yc) or (y != '' and y == xc)
                                  for x, y, xc, yc in zip(acr_a, acr_b, ca, cb)), np.float32, n)
    X['n_first_eq'] = np.fromiter((x.split()[:1] == y.split()[:1] for x, y in zip(a, b)), np.float32, n)
    X['n_len_diff'] = np.fromiter((abs(len(x) - len(y)) for x, y in zip(ca, cb)), np.float32, n)
    X['n_skel_ratio'] = S(fuzz.ratio, L.col('n_skel', l), R.col('n_skel', r))
    X['n_trigram_jacc'] = trigram_jacc(a, b)
    X['n_soundex'] = soundex_match(a, b)
    X['n_len_ratio'] = name_len_ratio(a, b)
    # skel trigram for cross-script robustness
    sk_l, sk_r = L.col('n_skel', l), R.col('n_skel', r)
    X['n_skel_trigram'] = trigram_jacc(sk_l, sk_r)
    xa = [x or y for x, y in zip(L.col('n_alt', l), a)]            # DBA / trade-name aliases
    xb = [x or y for x, y in zip(R.col('n_alt', r), b)]
    X['n_alias_tset'] = np.maximum.reduce([X['n_tset'], S(fuzz.token_set_ratio, xa, b),
                                           S(fuzz.token_set_ratio, a, xb), S(fuzz.token_set_ratio, xa, xb)])
    X['legal_rel'] = legal_rel(L.col('n_legal', l), R.col('n_legal', r))
    X['ntok_l'] = L.num['n_ntok'][l].astype(np.float32)
    X['ntok_r'] = R.num['n_ntok'][r].astype(np.float32)
    for c in ('nfreq', 'afreq'):
        X[f'{c}_l'] = L.num[c][l]
        X[f'{c}_r'] = R.num[c][r]

    a1, a2 = L.col('a_core', l), R.col('a_core', r)
    X['a_ratio'] = S(fuzz.ratio, a1, a2)
    X['a_tset'] = S(fuzz.token_set_ratio, a1, a2)
    X['a_tsort'] = S(fuzz.token_sort_ratio, a1, a2)
    X['a_partial'] = S(fuzz.partial_ratio, a1, a2)
    X['a_jacc'] = tok_jaccard(a1, a2)
    X['a_overlap_min'] = tok_overlap_min(a1, a2)
    X['a_trigram_jacc'] = trigram_jacc(a1, a2)
    X['a_jw'] = S(JaroWinkler.normalized_similarity, a1, a2)
    p1, p2 = L.col('postal', l), R.col('postal', r)
    X['postal_pref3'] = np.fromiter(((x[:3] == y[:3]) if (x and y) else np.nan for x, y in zip(p1, p2)), np.float32, n)
    u1, u2 = L.col('nums', l), R.col('nums', r)
    X['num_jacc'] = tok_jaccard(u1, u2)
    X['num_conflict'] = np.fromiter((float(bool(x) and bool(y) and not (set(x.split()) & set(y.split())))
                                     for x, y in zip(u1, u2)), np.float32, n)
    f1, f2 = L.col('first_num', l), R.col('first_num', r)
    X['first_num_eq'] = np.fromiter(((x == y) if (x and y) else np.nan for x, y in zip(f1, f2)), np.float32, n)
    X['a_ntok_l'] = L.num['a_ntok'][l].astype(np.float32)
    X['a_ntok_r'] = R.num['a_ntok'][r].astype(np.float32)
    X['landmark_l'] = L.num['landmark'][l].astype(np.float32)
    X['landmark_r'] = R.num['landmark'][r].astype(np.float32)

    n_empty = (L.num['n_ntok'][l] == 0) | (R.num['n_ntok'][r] == 0)
    a_empty = (L.num['a_ntok'][l] == 0) | (R.num['a_ntok'][r] == 0)
    for k in X:
        if k.startswith('n_'):
            X[k][n_empty] = np.nan
        elif k.startswith('a_') and k not in ('a_ntok_l', 'a_ntok_r'):
            X[k][a_empty] = np.nan
    return X
