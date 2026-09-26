# Business Entity Resolution — reproduction guide

Pipeline: **data → normalisation → stage 1 blocking → stage 2 candidate filter → stage 3 matcher → output files**.
Only the provided training/test files are used; no external data, APIs or pretrained models.

## 1. Environment
Python 3.10+ on Windows, macOS or Linux. Everything runs on CPU. RAM: 8 GB works with `--workers 2 --test-batch-s1 20000`; 16 GB is comfortable.

```bash
cd code/business_entity_resolution
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Run end to end
Point `--data-dir` at the unzipped challenge folder (the one that contains `dataset/train` and `dataset/test`).
In Git Bash use forward slashes (`C:/Users/...`).

```bash
# 1) ~2 minute trial on the first 50k rows of every file - checks everything works (writes only to runs/)
python src/run_pipeline.py --data-dir /path/to/student_resource --nrows 50000

# 2) the real run (all ~24M records; roughly 1 hour, a few GB of RAM)
python src/run_pipeline.py --data-dir /path/to/student_resource
```

The real run writes, and checks against every format rule:
- `../../output/matching_results.tsv` (the file uploaded to the leaderboard)
- `../../output/candidate_pairs.tsv` (the exact candidate set the matcher scores)

It also runs `utils/validate_submission.py` from the challenge kit when that script is present.

Every run keeps a copy in `runs/<run_id>/`: the outputs, `log.txt`, `run_info.json` (settings, validation F0.5,
candidate statistics), `models.pkl`, `environment.txt` (exact library versions) and `methodology_draft.md`.
This is the version history of all submissions.

If the test phase fails after training finished, rerun without retraining:
```bash
python src/run_pipeline.py --data-dir /path/to/student_resource --models runs/<run_id>/models.pkl
```

## 3. Useful options
All settings are in `src/config.py` and can be changed on the command line:
| option | meaning |
|---|---|
| `--workers` (cores-1, max 6) | parallel normalisation processes; use 2-3 on an 8 GB machine |
| `--test-batch-s1` (50000) | test Source-1 rows per batch; lower (e.g. 20000) if RAM is short |
| `--max-train-s1` (60000) | Source-1 entities sampled for training (all Source 2/3 records are kept) |
| `--prune-recall` (0.995) | share of stage-1 true pairs the candidate filter must keep; lower = smaller candidate set |
| `--prune-max-k` (10) | hard cap on candidates per Source-1 entity |
| `--k-name / --k-addr` (15 / 10) | top-K neighbours per index view (raise if stage-1 recall is low) |
| `--max-df-max` (300) | keys shared by more records than this are not indexed (raise for recall, lower for speed) |
| `--one-to-one` (1) | each Source-2/3 record is assigned to at most one Source-1 entity |

## 4. Build the submission zip
1. Copy the challenge's `Documentation_template.md` to the package root (next to `output/` and `code/`) and fill it
   in. `runs/<run_id>/methodology_draft.md` has all sections written with that run's numbers.
2. Copy `runs/<run_id>/environment.txt` into `requirements.txt` to pin exact versions.
3. From `code/business_entity_resolution/`, run:
```bash
python src/package_submission.py --team-name <team_name>
```
This creates `<team_name>_submission.zip` next to the package folder, in the required structure.

## Source files
| file | purpose |
|---|---|
| `src/run_pipeline.py` | entry point; orchestrates all stages, logging, run history |
| `src/config.py` | all settings |
| `src/data_io.py` | TSV reading, output writing, format checks, official validator |
| `src/normalize.py` | name / address normalisation (abbreviations, legal suffixes, DBA, landmarks, postal codes, transliteration skeleton) |
| `src/records.py` | streaming reader, parallel normalisation, compact string storage |
| `src/blocking.py` | stage 1: hashed, frequency-capped inverted index, top-K per view |
| `src/features.py` | stage 2 cheap features and stage 3 full pairwise features |
| `src/model.py` | LightGBM with GroupKFold, candidate-filter threshold, macro F0.5, decision rule |
| `src/package_submission.py` | builds the submission zip |
