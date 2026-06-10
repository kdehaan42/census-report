import csv
import re
from pathlib import Path
from typing import List, Optional

METADATA_FILE = Path("census_income_metadata.txt")
CSV_FILES = [Path("census_income_learn.csv"), Path("census_income_test.csv")]

HEADER_LINE_REGEX = re.compile(r"^\s*\|?\s*(.+?)\s{2,}([A-Z0-9][A-Z0-9]+)\s*$")


def parse_metadata_columns(metadata_path: Path, csv_cols: Optional[int] = None) -> List[str]:
    """Parse metadata lines and return ordered column names."""
    names: List[str] = []
    text = metadata_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        match = HEADER_LINE_REGEX.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        if name.startswith("|"):
            name = name.lstrip("|").strip()
        if name:
            names.append(name)

    if csv_cols is not None:
        names = normalize_metadata_columns(names, csv_cols)

    return names


def normalize_metadata_columns(names: list[str], csv_cols: int) -> list[str]:
    """Adjust the parsed metadata list to match the CSV structure."""
    normalized = list(names)
    if len(normalized) == csv_cols + 2:
        if "adjusted gross income" in normalized and "total person earnings" in normalized:
            normalized.remove("adjusted gross income")
            normalized.remove("total person earnings")
    elif len(normalized) == csv_cols + 1 and "total person earnings" in normalized:
        normalized.remove("total person earnings")
    return normalized


def build_headers(csv_cols: int, metadata_cols: list[str]) -> list[str]:
    """Map metadata names to the CSV column count with fallback names for extras."""
    if csv_cols == len(metadata_cols):
        return metadata_cols

    if csv_cols == len(metadata_cols) + 1:
        return metadata_cols + ["label"]

    if csv_cols > len(metadata_cols):
        extra_count = csv_cols - len(metadata_cols)
        extras = [f"extra_{i+1}" for i in range(extra_count)]
        return metadata_cols + extras

    return metadata_cols[:csv_cols]


def has_header(file_path: Path, header_row: list[str]) -> bool:
    with file_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        first_row = next(reader, [])
    return first_row == header_row


def prepend_header(file_path: Path, header_row: list[str]) -> None:
    temp_path = file_path.with_suffix(file_path.suffix + ".tmp")
    with file_path.open(newline="", encoding="utf-8", errors="replace") as src, temp_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        writer = csv.writer(dst)
        writer.writerow(header_row)
        for row in csv.reader(src):
            writer.writerow(row)
    temp_path.replace(file_path)


def main() -> None:
    metadata_names = parse_metadata_columns(METADATA_FILE)
    if not metadata_names:
        raise SystemExit(f"Could not parse any column names from {METADATA_FILE}")

    for csv_file in CSV_FILES:
        if not csv_file.exists():
            print(f"Skipping missing file: {csv_file}")
            continue

        with csv_file.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            first_row = next(reader, [])

        metadata_names = parse_metadata_columns(METADATA_FILE, csv_cols=len(first_row) - 1)
        header_row = build_headers(len(first_row), metadata_names)

        if has_header(csv_file, header_row):
            print(f"Header already present in {csv_file}")
            continue

        print(f"Adding header to {csv_file} ({len(header_row)} columns)")
        prepend_header(csv_file, header_row)

    print("Done.")


if __name__ == "__main__":
    main()
