"""Apply sticker tag and description edits from the labelling page to lilia's database.

    python scripts/apply_sticker_edits.py labels.json [--data-dir ~/.local/share/lilia] [--dry-run]

Reads the "edits" of a labels.json exported by scripts/sticker_label_page.py
({"<id>": {"tags", "description"}}) and rewrites those registered stickers,
marking them modified_by = 'USER' so machine tagging leaves them alone (an
edit already applied is only marked). The database is backed up to --backup-dir first. lilia reads the pool on every
search and embeds the new text then, so the running bot needs no restart.
Prints counts only.
"""

import argparse
import json
import os
import sqlite3
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels")
    parser.add_argument("--data-dir", default="~/.local/share/lilia")
    parser.add_argument("--backup-dir", default="~/backups/lilia")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with open(args.labels, encoding="utf-8") as file:
        edits = json.load(file).get("edits") or {}
    if not edits:
        print("no edits")
        return

    db = sqlite3.connect(os.path.join(os.path.expanduser(args.data_dir), "lilia.db"))
    db.execute("PRAGMA busy_timeout = 5000")
    columns = {row[1] for row in db.execute("PRAGMA table_info(stickers)")}
    if "modified_by" not in columns:
        raise SystemExit(
            "stickers.modified_by is missing: open the database with lilia (schema 13+) first"
        )
    current = {
        sticker_id: (tags, description, modified_by)
        for sticker_id, tags, description, modified_by in db.execute(
            "SELECT id, tags, description, modified_by FROM stickers WHERE status = 'registered'"
        )
    }
    changes, unchanged, missing, invalid = [], 0, 0, 0
    for key, edit in edits.items():
        sticker_id = int(key)
        tags = str(edit.get("tags", "")).strip()
        description = str(edit.get("description", "")).strip()
        if not description:
            invalid += 1
        elif sticker_id not in current:
            missing += 1
        elif current[sticker_id] == (tags, description, "USER"):
            unchanged += 1
        else:
            changes.append((sticker_id, tags, description))
    print(
        f"{len(changes)} to update, {unchanged} unchanged, {missing} not registered, {invalid} without a description"
    )
    if not changes or args.dry_run:
        return

    backups = os.path.expanduser(args.backup_dir)
    os.makedirs(backups, exist_ok=True)
    backup = os.path.join(backups, f"lilia.db-{time.strftime('%Y%m%d-%H%M%S')}")
    with sqlite3.connect(backup) as copy:
        db.backup(copy)
    os.chmod(backup, 0o600)
    print(f"backup: {backup}")

    with db:
        db.executemany(
            "UPDATE stickers SET tags = ?2, description = ?3, modified_by = 'USER' "
            "WHERE id = ?1 AND status = 'registered'",
            changes,
        )
    print(f"updated {len(changes)} stickers")


if __name__ == "__main__":
    main()
