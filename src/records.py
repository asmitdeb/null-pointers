"""Compact, streaming storage for millions of records.

Python str objects cost ~50 bytes of overhead each, so 10M records x 10 text fields would need >10 GB.
StrCol stores a whole column as one UTF-8 byte blob plus an int64 offset array (text bytes + 8 bytes per value),
and decodes only the rows that are actually needed (candidate pairs).
Files are read and normalised in chunks, in parallel worker processes, so the raw text is never all in memory.
"""
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from normalize import STR_COLS, process_chunk, encode_strs

SRC_COLS = ['entity_id', 'business_name', 'business_address', 'country']


class StrCol:
    """A column of strings stored as one UTF-8 blob + offsets; `take(idx)` decodes only the requested rows."""

    def __init__(self):
        self._blobs, self._lens = [], []
        self.blob, self.off = b'', np.zeros(1, np.int64)

    def append_encoded(self, blob, lens):
        """Add a chunk that was already encoded with normalize.encode_strs."""
        self._blobs.append(blob)
        self._lens.append(lens)

    def append(self, strs):
        """Add a chunk of Python strings."""
        self.append_encoded(*encode_strs(strs))

    def finalize(self):
        """Merge the appended chunks into one blob (call once, after the last append)."""
        if self._blobs:
            lens = np.concatenate(self._lens).astype(np.int64)
            self.blob = b''.join(self._blobs)
            self.off = np.zeros(len(lens) + 1, np.int64)
            np.cumsum(lens, out=self.off[1:])
        self._blobs, self._lens = [], []
        return self

    def __len__(self):
        return len(self.off) - 1

    def take(self, idx):
        """Decode rows `idx` -> list of str."""
        b, o = self.blob, self.off
        return [b[o[i]:o[i + 1]].decode('utf-8') for i in np.asarray(idx, dtype=np.int64).tolist()]

    def subset(self, idx):
        """New StrCol with only rows `idx`."""
        s = StrCol()
        s.append(self.take(idx))
        return s.finalize()

    @staticmethod
    def concat(cols):
        """Concatenate finalized StrCols."""
        s = StrCol()
        for c in cols:
            s.append_encoded(c.blob, np.diff(c.off))
        return s.finalize()


class Records:
    """Normalised records of one source (or several concatenated sources)."""

    NUM_COLS = ['src', 'country', 'n_ntok', 'a_ntok', 'landmark', 'hn', 'ha']

    def __init__(self):
        self.ids = StrCol()
        self.s = {c: StrCol() for c in STR_COLS}
        self.num = {c: [] for c in self.NUM_COLS}
        self.keys = {'n': [], 'a': []}          # per chunk: (indices int32, indptr int64) of hashed blocking keys

    def __len__(self):
        return len(self.ids)

    def finalize(self):
        """Merge chunks after loading."""
        self.ids.finalize()
        for c in self.s.values():
            c.finalize()
        self.num = {k: (np.concatenate(v) if len(v) else np.empty(0)) if isinstance(v, list) else v
                    for k, v in self.num.items()}
        return self

    @staticmethod
    def concat(parts):
        """Concatenate finalized Records (e.g. Source 2 + Source 3)."""
        r = Records()
        r.ids = StrCol.concat([p.ids for p in parts])
        r.s = {c: StrCol.concat([p.s[c] for p in parts]) for c in STR_COLS}
        r.num = {k: np.concatenate([p.num[k] for p in parts]) for k in Records.NUM_COLS}
        r.keys = {v: sum((p.keys[v] for p in parts), []) for v in ('n', 'a')}
        return r

    def subset(self, idx):
        """Records restricted to rows idx (keys are re-sliced chunk by chunk)."""
        idx = np.asarray(idx, dtype=np.int64)
        r = Records()
        r.ids = self.ids.subset(idx)
        r.s = {c: self.s[c].subset(idx) for c in STR_COLS}
        r.num = {k: v[idx] for k, v in self.num.items()}
        for v in ('n', 'a'):
            ind, ptr = merge_keys(self.keys[v])
            starts, ends = ptr[idx], ptr[idx + 1]
            lens = ends - starts
            sel = np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens) + np.arange(lens.sum())
            new_ptr = np.zeros(len(idx) + 1, np.int64)
            np.cumsum(lens, out=new_ptr[1:])
            r.keys[v] = [(ind[sel], new_ptr)]
        return r

    def col(self, name, idx):
        """Decoded strings (text columns) or numpy values (numeric columns) for rows idx."""
        return self.s[name].take(idx) if name in self.s else self.num[name][idx]


def merge_keys(chunks):
    """Concatenate per-chunk (indices, indptr) CSR structure into one."""
    if len(chunks) == 1:
        return chunks[0]
    inds, ptrs, base = [], [np.zeros(1, np.int64)], 0
    for ind, ptr in chunks:
        inds.append(ind)
        ptrs.append(ptr[1:] + base)
        base += ptr[-1]
    return np.concatenate(inds), np.concatenate(ptrs)


def iter_rows(path, chunk, nrows=0):
    """Yield (ids, names, addresses, countries) lists, `chunk` rows at a time.

    Plain line splitting (no quote handling - names may contain quotes); extra tabs are folded into the address.
    """
    with open(path, encoding='utf-8', errors='replace', newline='') as f:
        header = [h.strip() for h in f.readline().rstrip('\r\n').split('\t')]
        pos = {c: header.index(c) for c in SRC_COLS if c in header}
        missing = set(SRC_COLS) - set(pos)
        if missing:
            raise SystemExit(f'{path}: missing columns {missing}')
        n_cols = len(header)
        ids, names, addrs, ctry = [], [], [], []
        n = 0
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            p = line.split('\t')
            if len(p) > n_cols and n_cols == 4 and header == SRC_COLS:
                p = [p[0], p[1], ' '.join(p[2:-1]), p[-1]]
            elif len(p) < n_cols:
                p += [''] * (n_cols - len(p))
            ids.append(p[pos['entity_id']].strip())
            names.append(p[pos['business_name']])
            addrs.append(p[pos['business_address']])
            ctry.append(p[pos['country']].strip().lower())
            n += 1
            if len(ids) >= chunk:
                yield ids, names, addrs, ctry
                ids, names, addrs, ctry = [], [], [], []
            if nrows and n >= nrows:
                break
        if ids:
            yield ids, names, addrs, ctry


class CountryCodes:
    """Maps country labels (an open set - France included) to small integer codes."""

    def __init__(self):
        self.names, self.code = [], {}

    def encode(self, labels):
        """Labels -> int16 codes, adding new labels as they appear."""
        out = np.empty(len(labels), np.int16)
        for i, c in enumerate(labels):
            k = self.code.get(c)
            if k is None:
                k = self.code[c] = len(self.names)
                self.names.append(c)
            out[i] = k
        return out


class _Done:
    """Stand-in for a Future when running without worker processes."""

    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v


def load_source(path, src, cfg, countries, pool, row_filter=None):
    """Stream one source file through the normaliser -> Records.

    row_filter: optional set of entity ids to keep (used to load only the sampled training Source-1 entities).
    """
    rec = Records()
    pending = deque()

    def drain(block):
        while pending and (block or len(pending) > 2 * max(cfg.workers, 1)):
            ids, ctry, fut = pending.popleft()
            enc, nums, kn, ka = fut.result()
            rec.ids.append(ids)
            for c in STR_COLS:
                rec.s[c].append_encoded(*enc[c])
            rec.num['src'].append(np.full(len(ids), src, np.int8))
            rec.num['country'].append(countries.encode(ctry))
            for k, v in nums.items():
                rec.num[k].append(v)
            rec.keys['n'].append(kn)
            rec.keys['a'].append(ka)

    for ids, names, addrs, ctry in iter_rows(path, cfg.read_chunk, cfg.nrows):
        if row_filter is not None:
            keep = [i for i, x in enumerate(ids) if x in row_filter]
            if not keep:
                continue
            ids, names, addrs, ctry = ([v[i] for i in keep] for v in (ids, names, addrs, ctry))
        fut = pool.submit(process_chunk, names, addrs) if pool else _Done(process_chunk(names, addrs))
        pending.append((ids, ctry, fut))
        drain(False)
    drain(True)
    return rec.finalize()


def make_pool(cfg):
    """Process pool for normalisation (None = run in this process)."""
    if cfg.workers <= 1:
        return None
    return ProcessPoolExecutor(max_workers=cfg.workers)


def default_workers():
    """Leave one core free; at most 6 workers."""
    return max(1, min(6, (os.cpu_count() or 2) - 1))
