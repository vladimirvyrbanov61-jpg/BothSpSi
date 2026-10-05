"""Experiment-manifest helpers for reproducible thesis runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".toml"}
_SOURCE_ROOTS = {"ciphers", "scripts", "Simon", "Speck", "tests", "thesis"}
_EXECUTION_SOURCE_SUFFIXES = {".py", ".yaml", ".yml"}
_EXECUTION_SOURCE_ROOTS = {"ciphers", "scripts", "Simon", "Speck", "thesis"}
_ROOT_SOURCE_FILES = {
    "README.md",
    "PROJECT_REVIEW_CURRENT.md",
    "AUDIT_IMPLEMENTATION_STATUS.md",
    "requirements-thesis.txt",
    "requirements-lock.txt",
    "pyproject.toml",
    ".github/workflows/tests.yml",
}
MANIFEST_SCHEMA_VERSION = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
    }


def source_tree_digest(repo_root: Path) -> str:
    """Hash maintained source/configuration files, including untracked files."""
    digest = hashlib.sha256()
    paths: list[Path] = []
    for root_name in sorted(_SOURCE_ROOTS):
        root = repo_root / root_name
        if not root.exists():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
        )
    for name in sorted(_ROOT_SOURCE_FILES):
        path = repo_root / name
        if path.exists():
            paths.append(path)
    for path in sorted(set(paths)):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_digest(path).encode("ascii"))
    return digest.hexdigest()


def execution_source_digest(repo_root: Path) -> str:
    """Hash runtime code and experiment configs, excluding tests and prose."""
    digest = hashlib.sha256()
    paths: list[Path] = []
    for root_name in sorted(_EXECUTION_SOURCE_ROOTS):
        root = repo_root / root_name
        if not root.exists():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in _EXECUTION_SOURCE_SUFFIXES
            and "__pycache__" not in path.parts
            and not path.name.startswith("test_")
        )
    for path in sorted(set(paths)):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_digest(path).encode("ascii"))
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str | None]:
    names = (
        "numpy",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "torch",
        "matplotlib",
        "seaborn",
        "tensorboard",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}

    cuda_available = bool(torch.cuda.is_available())
    environment: dict[str, Any] = {
        "available": True,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms_at_manifest_creation": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "requested_determinism_policy": "strict",
    }
    if cuda_available:
        environment["cuda_device_count"] = torch.cuda.device_count()
        environment["cuda_device_name"] = torch.cuda.get_device_name(0)
    return environment


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    repo_root: Path,
    run_type: str,
    config_path: Path,
    config: dict[str, Any],
    parameters: dict[str, Any],
    paths: dict[str, Path | str | None],
) -> dict[str, Any]:
    created_at = utc_now()
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_type": run_type,
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "completed_at": None,
        "training_completed_at": None,
        "postprocessing": [],
        "command": [sys.executable, *sys.argv],
        "working_directory": os.getcwd(),
        "git": _git_metadata(repo_root),
        "source": {
            "scope": (
                "maintained source, tests, configs, root reviews, CI workflow, "
                "README, requirements, and pyproject"
            ),
            "sha256": source_tree_digest(repo_root),
            "execution_scope": "runtime Python modules and experiment YAML configs",
            "execution_sha256": execution_source_digest(repo_root),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "dependencies": _dependency_versions(),
            "torch": _torch_environment(),
        },
        "config": {
            "path": str(config_path.resolve()),
            "sha256": config_digest(config),
            "resolved": config,
        },
        "parameters": parameters,
        "statistics": {
            "grouping": ["cipher", "rounds", "split"],
            "dispersion": "sample standard deviation (ddof=1; zero for one seed)",
            "error_bars": (
                "95% Student-t confidence interval across seeds; clipped to "
                "the mathematical range of bounded metrics"
            ),
            "null_significance": (
                "exact per-seed conditional random-label p-values retained in "
                "log10 form and combined with Fisher's method in log space"
            ),
        },
        "paths": {key: str(value) if value is not None else None for key, value in paths.items()},
        "progress": {"completed_seeds": [], "failed_seeds": []},
        "artifacts": [],
    }


def artifact_inventory(run_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "type": path.suffix.lstrip(".") or "file",
                    "sha256": file_digest(path),
                }
            )
    return artifacts


def artifact_record(path: Path, *, role: str, repo_root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "role": role,
        "bytes": resolved.stat().st_size,
        "type": resolved.suffix.lstrip(".") or "file",
        "sha256": file_digest(resolved),
    }


def write_artifact_index(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            existing = list(json.load(handle).get("artifacts", []))
    merged = {
        (
            record.get("path"),
            record.get("role"),
            record.get("cipher"),
            record.get("rounds"),
            record.get("seed"),
        ): record
        for record in [*existing, *records]
    }
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(
                {"schema_version": 1, "artifacts": list(merged.values())},
                handle,
                indent=2,
                sort_keys=True,
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_artifact_index(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return list(json.load(handle).get("artifacts", []))


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    statistics = manifest.setdefault("statistics", {})
    statistics.setdefault("grouping", ["cipher", "rounds", "split"])
    statistics.setdefault(
        "dispersion",
        "sample standard deviation (ddof=1; zero for one seed)",
    )
    statistics.setdefault(
        "error_bars",
        (
            "95% Student-t confidence interval across seeds; clipped to "
            "the mathematical range of bounded metrics"
        ),
    )
    statistics.setdefault(
        "null_significance",
        (
            "method is recorded per raw and aggregate CSV row; per-seed log10 "
            "p-values are combined with Fisher's method in log space"
        ),
    )
    manifest["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def record_postprocessing(
    manifest: dict[str, Any],
    stage: str,
    *,
    repo_root: Path | None = None,
) -> None:
    record: dict[str, Any] = {
        "stage": stage,
        "completed_at": utc_now(),
        "command": [sys.executable, *sys.argv],
    }
    if repo_root is not None:
        record["source_sha256"] = source_tree_digest(repo_root)
        record["execution_source_sha256"] = execution_source_digest(repo_root)
        record["git"] = _git_metadata(repo_root)
    manifest.setdefault("postprocessing", []).append(record)
