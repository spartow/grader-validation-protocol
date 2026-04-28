#!/usr/bin/env python3
"""Rename annotators: Omid → Omid, Ava → Ava everywhere in the project."""

import os
import csv
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Text replacements (order matters — do case-sensitive variants)
TEXT_REPLACEMENTS = [
    ("Omid", "Omid"),
    ("omid", "omid"),
    ("OMID", "OMID"),
    ("Ava", "Ava"),
    ("ava", "ava"),
    ("AVA", "AVA"),
]

# File renames: (old_name_part, new_name_part)
FILE_RENAMES = [
    ("omid", "omid"),
    ("ava", "ava"),
]

TEXT_EXTENSIONS = {".py", ".md", ".csv", ".json", ".txt", ".jsonl", ".gitignore"}


def rename_in_text(content):
    """Apply all text replacements."""
    for old, new in TEXT_REPLACEMENTS:
        content = content.replace(old, new)
    return content


def process_text_files(root):
    """Find and rename content in all text files."""
    changed_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip .git
        if ".git" in dirpath:
            continue
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if fpath.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                with open(fpath, encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, PermissionError):
                continue

            new_content = rename_in_text(content)
            if new_content != content:
                with open(fpath, "w", encoding="utf-8", newline="") as f:
                    f.write(new_content)
                changed_files.append(str(fpath.relative_to(root)))

    return changed_files


def rename_files(root):
    """Rename files and directories containing old names."""
    renamed = []
    # Collect all paths first, then rename bottom-up (deepest first)
    all_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        if ".git" in dirpath:
            continue
        for fname in filenames:
            all_paths.append(Path(dirpath) / fname)
        for dname in dirnames:
            if dname != ".git":
                all_paths.append(Path(dirpath) / dname)

    # Sort by depth (deepest first) so we rename children before parents
    all_paths.sort(key=lambda p: -len(p.parts))

    for fpath in all_paths:
        old_name = fpath.name
        new_name = old_name
        for old_part, new_part in FILE_RENAMES:
            new_name = new_name.replace(old_part, new_part)
        if new_name != old_name:
            new_path = fpath.parent / new_name
            if fpath.exists():
                fpath.rename(new_path)
                renamed.append((str(fpath.relative_to(root)), str(new_path.relative_to(root))))

    return renamed


def main():
    print("=" * 60)
    print("RENAMING: Omid → Omid, Ava → Ava")
    print("=" * 60)

    # Step 1: Rename content in text files
    print("\n--- Step 1: Renaming text content ---")
    changed = process_text_files(BASE)
    for f in changed:
        print(f"  Content updated: {f}")
    print(f"  Total files with content changes: {len(changed)}")

    # Step 2: Rename files/directories
    print("\n--- Step 2: Renaming files/directories ---")
    renamed = rename_files(BASE)
    for old, new in renamed:
        print(f"  {old} → {new}")
    print(f"  Total renames: {len(renamed)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
