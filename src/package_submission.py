"""Build <team_name>_submission.zip in the structure the challenge asks for.

Usage (from code/business_entity_resolution/):
    python src/package_submission.py --team-name my_team

Zips the package root: output/ (both TSVs), code/business_entity_resolution/ (src, README, requirements) and
Documentation_template.md. The runs/ history folder is left out to keep the zip small.
"""
import argparse
import os
import zipfile

REQUIRED = ['output/matching_results.tsv', 'output/candidate_pairs.tsv',
            'code/business_entity_resolution/README.md', 'code/business_entity_resolution/requirements.txt',
            'Documentation_template.md']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--team-name', required=True)
    args = ap.parse_args()
    code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.dirname(os.path.dirname(code_dir))
    missing = [p for p in REQUIRED if not os.path.exists(os.path.join(root, p))]
    if missing:
        raise SystemExit(f'missing before zipping: {missing}')
    zip_path = os.path.join(os.path.dirname(root), f'{args.team_name}_submission.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for dp, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ('runs', '__pycache__', '.ipynb_checkpoints')]
            for fn in files:
                full = os.path.join(dp, fn)
                z.write(full, os.path.relpath(full, root))
    with zipfile.ZipFile(zip_path) as z:
        print('\n'.join(sorted(z.namelist())))
    print(f'\nwrote {zip_path}')


if __name__ == '__main__':
    main()
