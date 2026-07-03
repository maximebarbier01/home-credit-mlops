from __future__ import annotations

import json
from pathlib import Path

from openpyxl.drawing.image import Image as ExcelImage
import pandas as pd


TABLE_SUFFIXES = {".csv", ".parquet", ".json"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_SHEET_NAME_LENGTH = 31
MAX_IMAGE_WIDTH = 1200
MAX_IMAGE_HEIGHT = 800


def _normalize_json_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _read_json_as_frame(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return pd.DataFrame(
            {
                "key": list(payload.keys()),
                "value": [_normalize_json_value(value) for value in payload.values()],
            }
        )
    if isinstance(payload, list):
        if payload and all(isinstance(item, dict) for item in payload):
            return pd.DataFrame(payload)
        return pd.DataFrame({"value": [_normalize_json_value(item) for item in payload]})
    return pd.DataFrame({"value": [_normalize_json_value(payload)]})


def _read_supported_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        return _read_json_as_frame(path)
    raise ValueError(f"Unsupported table format: {path}")


def _sanitize_sheet_name(name: str, existing_names: set[str]) -> str:
    invalid_characters = set('[]:*?/\\')
    cleaned = "".join("_" if character in invalid_characters else character for character in name)
    cleaned = cleaned.strip().strip("'") or "sheet"
    cleaned = cleaned[:MAX_SHEET_NAME_LENGTH]

    candidate = cleaned
    counter = 1
    while candidate in existing_names:
        suffix = f"_{counter}"
        candidate = f"{cleaned[: MAX_SHEET_NAME_LENGTH - len(suffix)]}{suffix}"
        counter += 1

    existing_names.add(candidate)
    return candidate


def _resize_image(image: ExcelImage) -> None:
    width = getattr(image, "width", None)
    height = getattr(image, "height", None)
    if not width or not height:
        return

    ratio = min(MAX_IMAGE_WIDTH / width, MAX_IMAGE_HEIGHT / height, 1.0)
    image.width = int(width * ratio)
    image.height = int(height * ratio)


def build_workbook_from_directory(
    source_dir: str | Path,
    workbook_path: str | Path,
) -> Path | None:
    source = Path(source_dir)
    if not source.exists():
        return None

    supported_files = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in (TABLE_SUFFIXES | IMAGE_SUFFIXES)
    )
    if not supported_files:
        return None

    destination = Path(workbook_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_names: set[str] = set()

    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        manifest_rows: list[dict[str, str]] = []
        for path in supported_files:
            kind = "image" if path.suffix.lower() in IMAGE_SUFFIXES else "table"
            manifest_rows.append(
                {
                    "file_name": path.name,
                    "kind": kind,
                    "relative_path": path.relative_to(source).as_posix(),
                }
            )

        pd.DataFrame(manifest_rows).to_excel(
            writer,
            sheet_name=_sanitize_sheet_name("manifest", existing_names),
            index=False,
        )

        for path in supported_files:
            if path.suffix.lower() not in TABLE_SUFFIXES:
                continue
            frame = _read_supported_table(path)
            sheet_name = _sanitize_sheet_name(path.stem, existing_names)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

        for path in supported_files:
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            sheet_name = _sanitize_sheet_name(f"img_{path.stem}", existing_names)
            worksheet = writer.book.create_sheet(sheet_name)
            worksheet["A1"] = path.name
            worksheet["A2"] = path.as_posix()
            image = ExcelImage(path.as_posix())
            _resize_image(image)
            worksheet.add_image(image, "A4")

    return destination


def build_experiment_workbooks(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    workbooks: list[Path] = []

    root_workbook = build_workbook_from_directory(root, root / "summary.xlsx")
    if root_workbook is not None:
        workbooks.append(root_workbook)

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        workbook = build_workbook_from_directory(
            directory,
            directory / f"{directory.name}.xlsx",
        )
        if workbook is not None:
            workbooks.append(workbook)

    return workbooks
