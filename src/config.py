"""Default settings for the entity-resolution pipeline.

Every field can be overridden from the command line, e.g.
    python src/run_pipeline.py --data-dir ../student_resource --prune-recall 0.99
(underscores in field names become dashes on the command line).
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- paths / run control -------------------------------------------------------------
    data_dir: str = ''          # folder containing dataset/train and dataset/test (the unzipped student_resource)
    out_dir: str = ''           # where the two TSVs go (default: <package>/output)
    runs_dir: str = ''          # run history, one sub-folder per run (default: <code dir>/runs)
    models: str = ''            # path to runs/<id>/models.pkl -> skip training and reuse those models
    nrows: int = 0              # >0 = TRIAL run on the first N rows of every file (checks the pipeline works)
    seed: int = 42
    workers: int = 0            # normalisation processes (0 = CPU cores - 1, max 6)
    n_threads: int = 0          # LightGBM threads (0 = all cores)
    read_chunk: int = 100000    # rows per normalisation chunk

    # ---- stage 1: hashed, frequency-capped inverted index --------------------------------
    k_name: int = 20            # top-K per Source-1 record from the name-key view
    k_addr: int = 15            # top-K from the address-key view
    max_df_frac: float = 0.0001 # keys carried by more than this share of Source-2/3 records are not indexed ...
    max_df_min: int = 50        # ... but the cap is never below this many records
    max_df_max: int = 500       # ... and never above this many (bounds time and memory per query)
    chunk_rows: int = 2000      # Source-1 rows per sparse-product chunk

    # ---- stage 2: learned candidate filter (produces candidate_pairs.tsv) -----------------
    prune_recall: float = 0.998  # keep this share of the true pairs that stage 1 found
    prune_max_k: int = 15        # never more than this many candidates per Source-1 record
    stage2_folds: int = 3

    # ---- stage 3: matcher -----------------------------------------------------------------
    folds: int = 5
    max_train_s1: int = 100000   # Source-1 entities sampled for training (all Source 2/3 records are kept)
    test_batch_s1: int = 50000   # Source-1 rows per test batch (lower if RAM is short)
    one_to_one: int = 1          # 1 = each Source-2/3 record goes to at most one Source-1 entity (S1 is deduplicated)
