from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from langsmith import Client

try:
    from dotenv import find_dotenv, load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def find_dotenv(*args, **kwargs) -> str:
        return ""

    def load_dotenv(*args, **kwargs) -> bool:
        return False


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _jsonable(model_dump(mode="json"))
        except TypeError:
            return _jsonable(model_dump())
        except Exception:
            pass

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return _jsonable(dict_method())
        except Exception:
            pass

    return str(value)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return slug.strip("-") or "project"


def _chunk_path(file_prefix: str, project_name: str, chunk_number: int) -> Path:
    prefix = Path(file_prefix)
    parent = prefix.parent if str(prefix.parent) != "." else Path(".")
    stem = prefix.name
    return parent / f"{stem}_{_slug(project_name)}_chunk_{chunk_number:04d}.jsonl"


def _write_chunk(path: Path, runs: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(_jsonable(run), ensure_ascii=False, default=str))
            handle.write("\n")


def _project_names_from_prefixes(client: Client, prefixes: Iterable[str]) -> list[str]:
    names: list[str] = []
    for prefix in prefixes:
        for project in client.list_projects(name_contains=prefix):
            name = str(_value(project, "name", "") or "")
            if name and name not in names:
                names.append(name)
    return names


def download_project_traces(
    *,
    client: Client,
    project_name: str,
    file_prefix: str,
    chunk_size: int = 1000,
    root_only: bool = True,
) -> list[Path]:
    saved_paths: list[Path] = []
    chunk: list[Any] = []
    chunk_number = 1
    total = 0

    run_iterator = client.list_runs(
        project_name=project_name,
        is_root=True if root_only else None,
    )
    for run in run_iterator:
        chunk.append(run)
        total += 1

        if len(chunk) >= chunk_size:
            path = _chunk_path(file_prefix, project_name, chunk_number)
            _write_chunk(path, chunk)
            saved_paths.append(path)
            print(f"Saved {len(chunk)} traces from {project_name} to {path}")
            chunk = []
            chunk_number += 1

    if chunk:
        path = _chunk_path(file_prefix, project_name, chunk_number)
        _write_chunk(path, chunk)
        saved_paths.append(path)
        print(f"Saved {len(chunk)} traces from {project_name} to {path}")

    if total == 0:
        print(f"No traces found for project: {project_name}")
    else:
        print(f"Downloaded {total} trace(s) from {project_name}.")

    return saved_paths


def main() -> None:
    load_dotenv(find_dotenv(usecwd=True), override=False)

    parser = argparse.ArgumentParser(
        description="Download LangSmith traces in JSONL chunks."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("LANGSMITH_API_KEY"),
        help="LangSmith API key. Defaults to LANGSMITH_API_KEY from the environment or .env.",
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("LANGSMITH_ENDPOINT"),
        help="Optional LangSmith API URL. Defaults to LANGSMITH_ENDPOINT if set.",
    )
    parser.add_argument(
        "--project-name",
        action="append",
        default=[],
        help="Exact LangSmith project name to download. Can be passed multiple times.",
    )
    parser.add_argument(
        "--project-prefix",
        action="append",
        default=[],
        help="Download all projects whose names contain this text. Can be passed multiple times.",
    )
    parser.add_argument(
        "--file-prefix",
        default="evaluators/traces",
        help="Prefix for saved JSONL files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of traces per saved file.",
    )
    parser.add_argument(
        "--include-child-runs",
        action="store_true",
        help="Download child runs too. By default only root traces are downloaded.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Set LANGSMITH_API_KEY or pass --api-key.")
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be greater than 0.")

    client_kwargs = {"api_key": args.api_key}
    if args.api_url:
        client_kwargs["api_url"] = args.api_url
    client = Client(**client_kwargs)

    project_names = list(dict.fromkeys(args.project_name))
    project_names.extend(_project_names_from_prefixes(client, args.project_prefix))
    project_names = list(dict.fromkeys(project_names or ["evaluators"]))

    saved_paths: list[Path] = []
    for project_name in project_names:
        saved_paths.extend(
            download_project_traces(
                client=client,
                project_name=project_name,
                file_prefix=args.file_prefix,
                chunk_size=args.chunk_size,
                root_only=not args.include_child_runs,
            )
        )

    print(f"Saved {len(saved_paths)} file(s).")


if __name__ == "__main__":
    main()
