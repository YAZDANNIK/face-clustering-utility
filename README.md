# Face Clustering and Organization

Organize a large photo set by clustering faces and copying/moving each image into per-person folders using `face_recognition` encodings and `DBSCAN` clustering.

## Features
- Generates 128‑D face embeddings per image
- Clusters with `DBSCAN` (auto-detects number of people; `-1` = noise)
- Copies or moves files into `clustered/Person_{id}` folders
- Progress display with `--percent`
- Robust detection: upsample, rotation checks, optional tile scanning
- Safe dry-run mode to preview actions

## Requirements
- Python 3.10 recommended
- Install dependencies:
  - `pip install -r requirements.txt`
- Windows notes:
  - If `dlib` fails to compile, use Python 3.10 and install a prebuilt Windows wheel for `dlib` (cp310, win_amd64). Alternatively use Anaconda and install `dlib` from conda‑forge.

## Quick Start
- Create and activate a venv (Windows PowerShell):
  - `py -3.10 -m venv .venv310`
  - `.\.venv310\Scripts\activate`
- Install packages:
  - `pip install -r requirements.txt`
- Run with copy mode (keeps originals):
  - `.\.venv310\Scripts\python cluster_faces.py --source all_photos_for_fr --recursive --output clustered --eps 0.6 --min-samples 2 --model hog --upsample 2 --tiles 2 --percent --copy`

## Parameters
- `--source`: input directory of images
- `--output`: destination root for clustered folders
- `--recursive`: scan nested directories
- `--eps`: DBSCAN distance tolerance (typical 0.5–0.7)
- `--min-samples`: DBSCAN minimum samples per cluster (2–3 useful)
- `--model`: `hog` (CPU, fast) or `cnn` (more accurate)
- `--upsample`: detection upsampling (1–2; use 2 for small faces)
- `--tiles`: subdivide image into tiles when faces are missed (>=1)
- `--percent`: print percentage progress instead of a progress bar
- `--extensions`: comma-separated extensions (default `jpg,jpeg,png`)
- `--exclude`: comma-separated directory names to skip
- `--copy`: copy files (default is move when omitted)
- `--dry-run`: preview moves/copies only
- `--limit`: process only the first N images (useful for testing)

## Examples
- Preview with 20 images:
  - `.\.venv310\Scripts\python cluster_faces.py --source all_photos_for_fr --recursive --output clustered_samples --eps 0.6 --min-samples 2 --model hog --upsample 2 --tiles 2 --percent --limit 20 --dry-run`
- Copy test with 20 images:
  - `.\.venv310\Scripts\python cluster_faces.py --source all_photos_for_fr --recursive --output clustered_samples --eps 0.6 --min-samples 2 --model hog --upsample 2 --tiles 2 --percent --limit 20 --copy`
- Full dataset copy:
  - `.\.venv310\Scripts\python cluster_faces.py --source all_photos_for_fr --recursive --output clustered --eps 0.6 --min-samples 2 --model hog --upsample 2 --tiles 2 --percent --copy`
- Full dataset move (relocate files):
  - `.\.venv310\Scripts\python cluster_faces.py --source all_photos_for_fr --recursive --output clustered --eps 0.6 --min-samples 2 --model hog --upsample 2 --tiles 2 --percent`

## Tuning Tips
- If one person is split across multiple folders: increase `--eps` (e.g., 0.65–0.70)
- If different people merge: decrease `--eps` (e.g., 0.50–0.55)
- For small/low-res faces: set `--upsample 2` or try `--model cnn`
- Use `--tiles 2` to help detect faces missed by standard detection

## Git Hygiene
- `.gitignore` excludes local images and outputs (`all_photos_for_fr/`, `clustered/`, `clustered_samples/`, common image extensions, venvs, caches).
- If images were committed previously, remove them from Git index:
  - `git rm -r --cached all_photos_for_fr clustered clustered_samples`
  - Commit the change.

## Notes
- Dry-run prints planned actions and no longer creates empty target subfolders.
- Copy mode (`--copy`) is recommended for the first full run to verify clusters before relocating files.