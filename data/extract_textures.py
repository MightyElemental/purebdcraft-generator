#!/usr/bin/env python3
"""
This script automatically finds all .zip and .jar archive files within the raw/source
and raw/target directories (or a user-specified base directory). It extracts any PNG files
that are located in an assets/<name>/textures/... folder structure. The PNG files are saved
into a destination folder while preserving both the original raw subfolder ("source" or "target")
and the internal structure of the archive. For example, a PNG extracted from an archive found in
raw/source will be saved to: <dest_dir>/source/<name>/<relative_path>.

If an image already exists at the destination, it will be skipped. Once processing is complete,
the script prints how many new images were extracted.

The base directory and destination directory can be specified via command-line arguments using
the --base-dir and --dest-dir options. By default, base_dir is 'raw' and dest_dir is the current directory.
"""

import argparse
import zipfile
from pathlib import Path


def extract_png_from_archive(archive_path: Path, dest_path: Path) -> int:
    """
    Extracts new PNG files from the given archive if they are under an assets/<name>/textures/ directory.
    If the target PNG image already exists, it will not be overwritten.

    The PNG files are saved under the destination folder with the internal structure:
    dest_path/<name>/<relative_path>, where <relative_path> comes from the archive (skipping the
    first three folders - "assets", "<name>", "textures").

    Args:
        archive_path (Path): Path to the input .zip or .jar archive.
        dest_path (Path): Base directory where PNG files will be extracted.

    Returns:
        int: The number of new PNG images that were extracted.
    """
    if not archive_path.exists():
        print(f"Error: {archive_path} does not exist.")
        return 0

    if not zipfile.is_zipfile(archive_path):
        print(f"Error: {archive_path} is not a valid zip or jar file.")
        return 0

    new_images_count = 0
    with zipfile.ZipFile(archive_path, 'r') as zf:
        for file in zf.namelist():
            # Check that it's a PNG file within an assets folder.
            if file.lower().endswith('.png') and file.startswith('assets/'):
                parts = Path(file).parts
                # Expecting at least: assets, <name>, textures, ...
                if len(parts) >= 4 and parts[2] == 'textures':
                    # Extract <name> from assets/<name>/textures/...
                    name = parts[1]
                    # The relative path starts after "assets", "<name>", "textures"
                    relative_path = Path(*parts[3:])
                    save_path = dest_path / name / relative_path
                    # Create directory if it doesn't exist.
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    # If file already exists, skip extraction.
                    if save_path.exists():
                        continue
                    try:
                        with zf.open(file) as source, open(save_path, 'wb') as target:
                            target.write(source.read())
                        new_images_count += 1
                    except Exception as e:
                        print(f"Error extracting {file} from {archive_path}: {e}")

    print(f"PNG files from {archive_path.name} extracted successfully to {dest_path}")
    return new_images_count


def process_archives_in_folder(folder: Path, dest_path: Path) -> int:
    """
    Processes all .zip and .jar archives within the given folder.
    The destination for extraction will mirror the raw folder name (e.g., source or target)
    as a subfolder within the main destination directory.

    Args:
        folder (Path): Directory to search for archive files.
        dest_path (Path): Base destination directory for extraction.
                         The effective destination is dest_path/<folder.name>.

    Returns:
        int: The total number of new PNG images extracted from archives in this folder.
    """
    if not folder.exists() or not folder.is_dir():
        print(f"Directory {folder} does not exist or is not a directory.")
        return 0

    # Create a subfolder within the destination directory corresponding to the raw folder.
    effective_dest = dest_path / folder.name
    effective_dest.mkdir(parents=True, exist_ok=True)

    folder_new_images = 0
    for archive_file in folder.iterdir():
        if archive_file.suffix.lower() in ['.zip', '.jar']:
            print(f"Processing archive: {archive_file}")
            folder_new_images += extract_png_from_archive(archive_file, effective_dest)

    return folder_new_images


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for base directory and destination directory.

    Returns:
        argparse.Namespace: Parsed arguments with attributes base_dir and dest_dir.
    """
    parser = argparse.ArgumentParser(
        description="Extract PNG files from archives found in raw/source and raw/target."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="raw",
        help="Base directory containing 'source' and 'target' folders. Default is 'raw'."
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default=".",
        help="Destination directory where extracted PNG files will be saved. Default is current directory."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = Path(args.base_dir)
    source_dir = base_dir / "source"
    target_dir = base_dir / "target"
    dest_dir = Path(args.dest_dir)

    total_new_images = 0
    for archive_folder in (source_dir, target_dir):
        total_new_images += process_archives_in_folder(archive_folder, dest_dir)

    print(f"Total new images extracted: {total_new_images}")
