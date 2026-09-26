"""Stage 1 - scalable blocking with a hashed, document-frequency-capped inverted index.

Every record gets two sets of hashed keys (see normalize.name_keys / addr_keys):
  name view : name tokens, consonant-skeleton tokens, bigrams, compact name, token@postal, token#house-number
  addr view : address tokens and bigrams, postal code, postal code + house number
Keys are weighted by IDF computed on Source 2+3; keys carried by more than `cap` records ('ltd', 'road', a city
name ...) get weight 0, i.e. are removed from the index. The product A @ B.T over the remaining (rare) keys is
genuinely sparse: each Source-1 record is only compared with Source-2/3 records that share a rare key, so the cost
grows roughly linearly with the data. For every Source-1 record the top-K of each view are kept (union).
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize as l2_normalize

from normalize import N_HASH
from records import merge_keys

VIEWS = (('n', 1), ('a', 2))      # view key, bit in the candidate bitmask


def document_frequency(chunks):
    """Number of records carrying each hashed key."""
    df = np.zeros(N_HASH, np.int64)
    for ind, _ in chunks:
        df += np.bincount(ind, minlength=N_HASH)
    return df


def make_idf(df, n_docs, cfg):
    """IDF weights with the frequency cap applied (weight 0 = key not indexed). Returns (idf, cap)."""
    cap = int(np.clip(cfg.max_df_frac * n_docs, cfg.max_df_min, cfg.max_df_max))
    idf = (np.log((n_docs + 1) / (df + 1)) + 1).astype(np.float32)
    idf[df > cap] = 0
    return idf, cap


def weighted_matrix(chunks, idf):
    """Build the L2-normalised TF-IDF CSR matrix (records x N_HASH) from key chunks, dropping capped keys."""
    ind, ptr = merge_keys(chunks)
    n = len(ptr) - 1
    w = idf[ind]
    keep = w > 0
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))[keep]
    X = sp.csr_matrix((w[keep], (rows, ind[keep])), shape=(n, N_HASH), dtype=np.float32)
    return l2_normalize(X, copy=False)


class CandidateIndex:
    """Transposed Source-2/3 matrices (key -> records) for both views, plus the Source-1 query matrices."""

    def __init__(self, L, R, cfg, log):
        self.A, self.BT, self.caps = {}, {}, {}
        self.nR = len(R)
        for v, _ in VIEWS:
            df = document_frequency(R.keys[v])
            idf, cap = make_idf(df, len(R), cfg)
            B = weighted_matrix(R.keys[v], idf)
            self.BT[v] = B.tocsc().T          # CSC transposed == CSR of shape (N_HASH, nR); no extra copy
            del B
            self.A[v] = weighted_matrix(L.keys[v], idf)
            self.caps[v] = cap
            log(f'  index view {v}: {self.BT[v].nnz:,} Source-2/3 key entries | key frequency cap {cap}')
        R.keys = {'n': [], 'a': []}              # raw keys no longer needed
        L.keys = {'n': [], 'a': []}

    def query(self, l_idx, cfg):
        """Stage-1 candidates for Source-1 rows l_idx -> DataFrame[l, r, m, cos_n, cos_a]."""
        ks = {'n': cfg.k_name, 'a': cfg.k_addr}
        out = []
        for s in range(0, len(l_idx), cfg.chunk_rows):
            li = l_idx[s:s + cfg.chunk_rows]
            Smat, keys, bits = {}, [], []
            for v, bit in VIEWS:
                Smat[v] = (self.A[v][li] @ self.BT[v]).tocsr()
                a, b = topk_rows(Smat[v], ks[v])
                keys.append(a * self.nR + b)
                bits.append(np.full(len(a), bit, np.int8))
            key = np.concatenate(keys)
            if len(key) == 0:
                continue
            uniq, inv = np.unique(key, return_inverse=True)
            m = np.zeros(len(uniq), np.int8)
            np.bitwise_or.at(m, inv, np.concatenate(bits))
            rows, cols = uniq // self.nR, uniq % self.nR
            out.append(pd.DataFrame({
                'l': li[rows], 'r': cols, 'm': m,
                'cos_n': np.asarray(Smat['n'][rows, cols]).ravel().astype(np.float32),
                'cos_a': np.asarray(Smat['a'][rows, cols]).ravel().astype(np.float32)}))
        if not out:
            return pd.DataFrame({'l': np.empty(0, np.int64), 'r': np.empty(0, np.int64), 'm': np.empty(0, np.int8),
                                 'cos_n': np.empty(0, np.float32), 'cos_a': np.empty(0, np.float32)})
        return pd.concat(out, ignore_index=True)


def topk_rows(S, k):
    """Top-k columns per row of a sparse score matrix -> (row, col) int64 arrays."""
    S.eliminate_zeros()
    if S.nnz == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    rows = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
    order = np.lexsort((-S.data, rows))                 # by row, then score descending
    pos = np.arange(S.nnz) - S.indptr[rows[order]]      # rank of each entry inside its row
    keep = order[pos < k]
    return rows[keep].astype(np.int64), S.indices[keep].astype(np.int64)
