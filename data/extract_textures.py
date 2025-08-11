#!/usr/bin/env python3

import zipfile
import sys
from pathlib import Path

def extract_png_from_archive(archive_path, dest_path):
    archive_path = Path(archive_path)
    dest_path = Path(dest_path)

    if not archive_path.exists():
        print(f"Error: {archive_path} does not exist.")
        return

    if not zipfile.is_zipfile(archive_path):
        print(f"Error: {archive_path} is not a valid zip or jar file.")
        return

    with zipfile.ZipFile(archive_path, 'r') as zf:
        for file in zf.namelist():
            if file.lower().endswith('.png') and file.startswith('assets/'):
                parts = Path(file).parts
                if len(parts) >= 4 and parts[2] == 'textures':
                    # Extract <name> from assets/<name>/textures/...
                    name = parts[1]
                    relative_path = Path(*parts[3:])  # skip 'assets', <name>, 'textures'
                    save_path = dest_path / name / relative_path
                    save_path.parent.mkdir(parents=True, exist_ok=True)

                    with zf.open(file) as source, open(save_path, 'wb') as target:
                        target.write(source.read())

    print(f"PNG files extracted successfully to {dest_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_png.py <archive_file.jar|archive_file.zip> <destination_folder>")
    else:
        extract_png_from_archive(sys.argv[1], sys.argv[2])
