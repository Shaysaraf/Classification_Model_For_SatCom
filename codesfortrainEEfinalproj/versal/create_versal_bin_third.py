import os
import json
import shutil
import argparse
from collections import defaultdict
import random

def main():
    parser = argparse.ArgumentParser(description="Create equal bins for Versal DPU Testing")
    parser.add_argument("--src_dir", default=r"D:\versal_ready_bins", help="Source directory containing bins and manifest")
    parser.add_argument("--dest_dir", default=r"D:\versal_ready_equal_bins", help="Destination directory")
    parser.add_argument("--samples_per_class", type=int, default=6000, help="Number of samples to take per modulation class")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for selection")
    args = parser.parse_args()

    src_manifest_path = os.path.join(args.src_dir, "versal_manifest.json")
    dest_manifest_path = os.path.join(args.dest_dir, "versal_manifest.json")

    if not os.path.exists(src_manifest_path):
        print(f"[ERROR] Source manifest not found at {src_manifest_path}")
        return

    with open(src_manifest_path, "r") as f:
        manifest = json.load(f)

    # Group files by their label
    files_by_label = defaultdict(list)
    for filename, info in manifest.items():
        files_by_label[info["label"]].append(filename)

    os.makedirs(args.dest_dir, exist_ok=True)
    
    new_manifest = {}
    random.seed(args.random_seed)
    
    for label, filenames in files_by_label.items():
        # Shuffle to pick random samples, or just slice if you prefer the first N
        random.shuffle(filenames)
        selected_files = filenames[:args.samples_per_class]
        
        print(f"[INFO] Label {label}: selecting {len(selected_files)} out of {len(filenames)} total samples.")
        
        for filename in selected_files:
            src_file = os.path.join(args.src_dir, filename)
            dest_file = os.path.join(args.dest_dir, filename)
            
            if os.path.exists(src_file):
                shutil.copy2(src_file, dest_file)
                new_manifest[filename] = manifest[filename]
            else:
                print(f"[WARNING] File {src_file} not found, skipping.")

    with open(dest_manifest_path, "w") as f:
        json.dump(new_manifest, f, indent=4)
        
    print(f"[SUCCESS] Copied {len(new_manifest)} files to {args.dest_dir}.")
    print(f"[SUCCESS] Saved new manifest to {dest_manifest_path}.")

if __name__ == "__main__":
    main()
