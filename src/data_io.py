"""Reading the challenge TSV files and writing / checking the two submission files."""
import csv
import os
import subprocess
import sys

import pandas as pd

SRC_COLS = ['entity_id', 'business_name', 'business_address', 'country']


def count_lines(path):
    """Number of newline characters in a file (fast, constant memory)."""
    n = 0
    with open(path, 'rb') as f:
        for buf in iter(lambda: f.read(1 << 22), b''):
            n += buf.count(b'\n')
    return n


def _manual_parse(path, n_cols):
    """Fallback parser for rows with stray tabs: extra tabs are folded into the address field."""
    rows = []
    with open(path, encoding='utf-8', errors='replace') as f:
        header = f.readline().rstrip('\r\n').split('\t')
        for line in f:
            line = line.rstrip('\r\n')
            if not line:
                continue
            p = line.split('\t')
            if len(p) > n_cols:                       # id, name, <address with tabs>, country
                p = [p[0], p[1], ' '.join(p[2:-1]), p[-1]] if n_cols == 4 else p[:n_cols - 1] + [','.join(p[n_cols - 1:])]
            p += [''] * (n_cols - len(p))
            rows.append(p)
    return pd.DataFrame(rows, columns=[h.strip() for h in header][:n_cols])


def read_tsv(path, nrows=None):
    """Read a (small) challenge TSV as strings without quote processing (names can contain quote characters)."""
    try:
        df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                         encoding='utf-8', encoding_errors='replace', nrows=nrows)
    except pd.errors.ParserError:
        with open(path, encoding='utf-8', errors='replace') as f:
            n_cols = len(f.readline().split('\t'))
        df = _manual_parse(path, n_cols)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_ids(s):
    """'S2-1, S3-4' -> ['S2-1', 'S3-4']; empty -> []."""
    return [x.strip() for x in str(s).split(',') if x.strip()]


def write_outputs(s1_ids, cand_lists, match_lists, out_dir):
    """Write matching_results.tsv and candidate_pairs.tsv (one row per Source-1 id, unquoted, tab-separated)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for fname, col, lists in (('matching_results.tsv', 'matched_entity_ids', match_lists),
                              ('candidate_pairs.tsv', 'candidate_entity_ids', cand_lists)):
        p = os.path.join(out_dir, fname)
        with open(p, 'w', encoding='utf-8', newline='\n') as f:
            f.write(f'source1_entity_id\t{col}\n')
            for i, sid in enumerate(s1_ids):
                f.write(f"{sid}\t{','.join(lists.get(i, ()))}\n")
        paths[fname] = p
    return paths['matching_results.tsv'], paths['candidate_pairs.tsv']


def check_outputs(match_path, cand_path, s1_ids):
    """Format rules: header, every Source-1 id once and in order, only S2-/S3- ids, no duplicates, matches are a
    subset of candidates. (Existence of every id in the test files is checked by the official validator.)"""
    parsed = {}
    for path, col in ((match_path, 'matched_entity_ids'), (cand_path, 'candidate_entity_ids')):
        with open(path, encoding='utf-8') as f:
            header = f.readline().rstrip('\n')
            assert header == f'source1_entity_id\t{col}', f'bad header in {path}'
            d, n = {}, 0
            for line, sid in zip(f, s1_ids):
                a, b = line.rstrip('\n').split('\t')
                assert a == sid, f'{path}: row {n} is {a}, expected {sid}'
                v = [x for x in b.split(',') if x]
                assert len(v) == len(set(v)), f'{path}: duplicate ids for {a}'
                assert all(x.startswith(('S2-', 'S3-')) for x in v), f'{path}: non S2/S3 id for {a}'
                d[a] = v
                n += 1
            assert n == len(s1_ids) and f.readline() == '', f'{path}: wrong number of rows'
        parsed[col] = d
    for k, v in parsed['matched_entity_ids'].items():
        assert set(v) <= set(parsed['candidate_entity_ids'][k]), f'match not in candidates for {k}'
    return True


def run_official_validator(data_dir, match_path, cand_path):
    """Run utils/validate_submission.py from the challenge kit if it is present; returns its output text."""
    for root in (data_dir, os.path.dirname(data_dir)):
        v = os.path.join(root, 'utils', 'validate_submission.py')
        if os.path.exists(v):
            r = subprocess.run([sys.executable, v, '--matching', match_path, '--candidate', cand_path,
                                '--test-dir', os.path.join(data_dir, 'dataset', 'test')],
                               capture_output=True, text=True)
            return (r.stdout + r.stderr).strip()
    return 'official validator not found (utils/validate_submission.py) - skipped'
