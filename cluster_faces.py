import argparse
import os
import glob
import shutil
import sys
from collections import Counter
import numpy as np
import face_recognition
from sklearn.cluster import DBSCAN
from PIL import Image, ImageOps

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def list_images(source, recursive, extensions, exclude_dirs):
    exts = tuple([f".{e.lower()}" for e in extensions])
    excludes = set([d.strip().lower() for d in exclude_dirs if d.strip()])
    if recursive:
        images = []
        for root, dirs, files in os.walk(source):
            dirs[:] = [d for d in dirs if d.lower() not in excludes and not d.startswith('.')]
            for f in files:
                if f.lower().endswith(exts):
                    images.append(os.path.join(root, f))
        return images
    pattern = os.path.join(source, "*")
    return [p for p in glob.glob(pattern) if os.path.isfile(p) and p.lower().endswith(exts)]

def _progress_iter(it, total, desc):
    if tqdm is not None:
        return tqdm(it, total=total, desc=desc)
    return it

def _pil_load(path):
    try:
        img = Image.open(path)
        return ImageOps.exif_transpose(img.convert("RGB"))
    except Exception:
        return None

def compute_encodings(paths, model, upsample, percent=False):
    encodings = []
    records = []
    no_face = []
    total = len(paths)
    last_pct = -1
    it = _progress_iter(paths, total, "faces") if not percent else paths
    for i, path in enumerate(it, 1):
        if percent and total:
            pct = int(i * 100 / total)
            if pct != last_pct:
                sys.stdout.write(f"\rProgress: {pct}%")
                sys.stdout.flush()
                last_pct = pct
        try:
            pil_img = _pil_load(path)
            if pil_img is None:
                no_face.append(path)
                continue
            image = np.array(pil_img)
            locations = face_recognition.face_locations(image, number_of_times_to_upsample=upsample, model=model)
            faces = face_recognition.face_encodings(image, known_face_locations=locations)
            if not faces:
                locations = face_recognition.face_locations(image, number_of_times_to_upsample=min(upsample + 1, 3), model=model)
                faces = face_recognition.face_encodings(image, known_face_locations=locations)
            if not faces:
                locations = face_recognition.face_locations(image, number_of_times_to_upsample=2, model="cnn")
                faces = face_recognition.face_encodings(image, known_face_locations=locations)
            if not faces:
                pil = pil_img
                if pil is not None:
                    w, h = pil.size
                    scale = 1
                    if max(w, h) < 500:
                        scale = 500 / max(w, h)
                        pil = pil.resize((int(w * scale), int(h * scale)))
                    for angle in (0, 90, 180, 270):
                        arr = np.array(pil.rotate(angle, expand=True))
                        loc = face_recognition.face_locations(arr, number_of_times_to_upsample=2, model=model)
                        faces = face_recognition.face_encodings(arr, known_face_locations=loc)
                        if faces:
                            break
                    if not faces and args.tiles > 1:
                        tw = pil.size[0] // args.tiles
                        th = pil.size[1] // args.tiles
                        for i in range(args.tiles):
                            for j in range(args.tiles):
                                box = (i * tw, j * th, (i + 1) * tw, (j + 1) * th)
                                tile = pil.crop(box)
                                arr = np.array(tile)
                                loc = face_recognition.face_locations(arr, number_of_times_to_upsample=2, model=model)
                                f2 = face_recognition.face_encodings(arr, known_face_locations=loc)
                                if f2:
                                    faces = f2
                                    break
                            if faces:
                                break
            if not faces:
                no_face.append(path)
                continue
            for idx, enc in enumerate(faces):
                encodings.append(enc)
                records.append((path, idx))
        except Exception:
            no_face.append(path)
    if percent:
        sys.stdout.write("\n")
        sys.stdout.flush()
    return encodings, records, no_face

def cluster_encodings(encodings, eps, min_samples):
    X = np.array(encodings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(X)
    return labels

def assign_labels_to_images(records, labels):
    by_image = {}
    for (path, _), lab in zip(records, labels):
        by_image.setdefault(path, []).append(lab)
    assignment = {}
    for path, labs in by_image.items():
        filtered = [l for l in labs if l != -1]
        if filtered:
            assignment[path] = Counter(filtered).most_common(1)[0][0]
        else:
            assignment[path] = -1
    return assignment

def ensure_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def organize_files(assignments, output_dir, move, dry_run, noise_folder, person_prefix):
    ensure_dir(output_dir)
    moved = 0
    per_cluster = {}
    for _, lab in assignments.items():
        per_cluster[lab] = per_cluster.get(lab, 0) + 1
    for path, lab in assignments.items():
        target_name = noise_folder if lab == -1 else f"{person_prefix}{lab}"
        target_dir = os.path.join(output_dir, target_name)
        dst = os.path.join(target_dir, os.path.basename(path))
        if dry_run:
            print(f"PLAN: {'MOVE' if move else 'COPY'} {path} -> {dst}")
        else:
            ensure_dir(target_dir)
            try:
                if move:
                    shutil.move(path, dst)
                else:
                    shutil.copy2(path, dst)
                moved += 1
            except Exception as e:
                print(f"SKIP {path}: {e}")
    print(f"Assigned images per cluster: {sorted(per_cluster.items(), key=lambda x: x[0])}")
    if not dry_run:
        print(f"Total {'moved' if move else 'copied'}: {moved}")

def compute_dhash_bits(path):
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=".")
    parser.add_argument("--output", default="clustered")
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--model", choices=["hog", "cnn"], default="hog")
    parser.add_argument("--upsample", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extensions", default="jpg,jpeg,png")
    parser.add_argument("--exclude", default=".venv,.venv310,.venv311,__pycache__,node_modules")
    parser.add_argument("--noise-folder", default="Noise")
    parser.add_argument("--person-prefix", default="Person_")
    parser.add_argument("--tiles", type=int, default=1)
    parser.add_argument("--percent", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    extensions = [e.strip().lower() for e in args.extensions.split(",") if e.strip()]
    exclude_dirs = [e.strip() for e in args.exclude.split(",") if e.strip()]
    images = list_images(args.source, args.recursive, extensions, exclude_dirs)
    if args.limit and args.limit > 0:
        images = images[:args.limit]
    print(f"Found {len(images)} images")
    if not images:
        print("No images found")
        return
    encodings, records, no_face = compute_encodings(images, args.model, args.upsample, args.percent)
    print(f"Images with faces: {len(set([r[0] for r in records]))}")
    print(f"Images without faces: {len(no_face)}")
    if not encodings:
        print("No face encodings computed")
        return
    labels = cluster_encodings(encodings, args.eps, args.min_samples)
    unique_people = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Found {unique_people} distinct people")
    assignments = assign_labels_to_images(records, labels)
    organize_files(assignments, args.output, move=not args.copy, dry_run=args.dry_run, noise_folder=args.noise_folder, person_prefix=args.person_prefix)

if __name__ == "__main__":
    main()