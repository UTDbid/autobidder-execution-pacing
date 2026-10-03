#!/usr/bin/env python3
"""Recover the eight immutable official Alimama archives.

The downloader has a closed allowlist with pinned Content-Length values.  It
resumes into ``<filename>.part``, validates size and ZIP integrity, and only
then atomically promotes the archive to its immutable final name.  Extraction
uses a sibling staging directory and never overwrites an existing original.
Every terminal manifest is written atomically.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


_REPO_ROOT = Path(__file__).resolve().parents[2]


class RecoveryGateError(RuntimeError):
    """A pinned recovery invariant failed."""


@dataclass(frozen=True)
class ArchiveSpec:
    url: str
    expected_bytes: int
    kind: str
    level: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        return Path(self.filename).stem


ARCHIVES: tuple[ArchiveSpec, ...] = (
    ArchiveSpec(
        "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
        "share/autoBidding_aigb_track_data_period_7-8.zip",
        1_218_197_248,
        "period",
        "high",
    ),
    ArchiveSpec(
        "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
        "share/final/autoBidding_aigb_track_final_data_period_7-8.zip",
        1_208_777_107,
        "period",
        "low",
    ),
    ArchiveSpec(
        "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
        "share/autoBidding_aigb_track_data_trajectory_data.zip",
        817_478_470,
        "trajectory",
        "high",
    ),
    ArchiveSpec(
        "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
        "share/autoBidding_aigb_track_data_trajectory_data_extended_1.zip",
        822_038_658,
        "trajectory",
        "high",
    ),
    ArchiveSpec(
        "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
        "share/autoBidding_aigb_track_data_trajectory_data_extended_2.zip",
        1_095_812_965,
        "trajectory",
        "high",
    ),
    ArchiveSpec(
        "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
        "share/final/autoBidding_aigb_track_final_data_trajectory_data_1.zip",
        815_948_175,
        "trajectory",
        "low",
    ),
    ArchiveSpec(
        "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
        "share/final/autoBidding_aigb_track_final_data_trajectory_data_2.zip",
        820_357_170,
        "trajectory",
        "low",
    ),
    ArchiveSpec(
        "https://alimama-bidding-competition.oss-cn-beijing.aliyuncs.com/"
        "share/final/autoBidding_aigb_track_final_data_trajectory_data_3.zip",
        1_093_292_843,
        "trajectory",
        "low",
    ),
)

_BY_FILENAME = {spec.filename: spec for spec in ARCHIVES}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path, *, progress_label: str | None = None) -> str:
    digest = hashlib.sha256()
    processed = 0
    next_heartbeat = 512 * 1024**2
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024**2)
            if not block:
                break
            digest.update(block)
            processed += len(block)
            if progress_label and processed >= next_heartbeat:
                print(
                    f"HEARTBEAT sha256 archive={progress_label} bytes={processed}",
                    flush=True,
                )
                next_heartbeat += 512 * 1024**2
    return digest.hexdigest()


def probe_content_length(url: str) -> int:
    """Return the final HTTP Content-Length or fail closed."""
    command = [
        "curl",
        "-fsSIL",
        "--retry",
        "3",
        "--connect-timeout",
        "30",
        "--max-time",
        "120",
        url,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=150,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RecoveryGateError(f"Content-Length probe failed for {url}: {exc}") from exc
    if completed.returncode != 0:
        raise RecoveryGateError(
            f"Content-Length probe failed for {url}: curl exit "
            f"{completed.returncode}: {completed.stderr[-500:]}"
        )
    lengths: list[int] = []
    statuses: list[int] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("http/"):
            fields = line.split()
            if len(fields) >= 2 and fields[1].isdigit():
                statuses.append(int(fields[1]))
        elif line.lower().startswith("content-length:"):
            value = line.split(":", 1)[1].strip()
            if value.isdigit():
                lengths.append(int(value))
    if not statuses or statuses[-1] != 200:
        raise RecoveryGateError(
            f"HTTP status gate failed for {url}: observed statuses={statuses}"
        )
    if not lengths:
        raise RecoveryGateError(f"Content-Length missing for {url}")
    return lengths[-1]


def verify_remote_contract(spec: ArchiveSpec) -> int:
    observed = probe_content_length(spec.url)
    if observed != spec.expected_bytes:
        raise RecoveryGateError(
            f"Content-Length changed for {spec.filename}: "
            f"expected={spec.expected_bytes} observed={observed}"
        )
    print(
        f"REMOTE_OK archive={spec.filename} http=200 content_length={observed}",
        flush=True,
    )
    return observed


def _safe_member_name(name: str) -> None:
    normalized = PurePosixPath(name)
    if normalized.is_absolute() or ".." in normalized.parts:
        raise RecoveryGateError(f"unsafe ZIP member path: {name!r}")


def inspect_zip(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise RecoveryGateError(f"archive missing: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            for info in infos:
                _safe_member_name(info.filename)
                if info.flag_bits & 0x1:
                    raise RecoveryGateError(
                        f"encrypted ZIP member is unsupported: {info.filename}"
                    )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RecoveryGateError(
                    f"ZIP CRC/integrity failure in {path.name}: {bad_member}"
                )
            members = [
                {
                    "name": info.filename,
                    "bytes": int(info.file_size),
                    "compressed_bytes": int(info.compress_size),
                    "crc32": f"{info.CRC:08x}",
                    "is_dir": bool(info.is_dir()),
                }
                for info in infos
            ]
    except (zipfile.BadZipFile, OSError) as exc:
        raise RecoveryGateError(f"ZIP integrity failure for {path}: {exc}") from exc
    return {
        "path": str(path.resolve()),
        "actual_bytes": path.stat().st_size,
        "sha256": sha256_file(path, progress_label=path.name),
        "zip_integrity": True,
        "members": members,
    }


def _crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024**2)
            if not block:
                break
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xFFFFFFFF


def _validate_extracted_tree(
    target: Path, infos: Iterable[zipfile.ZipInfo]
) -> list[dict[str, Any]]:
    expected_files = {info.filename: info for info in infos if not info.is_dir()}
    actual_files = {
        path.relative_to(target).as_posix(): path
        for path in target.rglob("*")
        if path.is_file()
    }
    if set(actual_files) != set(expected_files):
        missing = sorted(set(expected_files) - set(actual_files))
        extra = sorted(set(actual_files) - set(expected_files))
        raise RecoveryGateError(
            f"existing extraction inventory mismatch at {target}: "
            f"missing={missing} extra={extra}"
        )
    evidence: list[dict[str, Any]] = []
    for name, info in sorted(expected_files.items()):
        path = actual_files[name]
        size = path.stat().st_size
        crc = _crc32_file(path)
        if size != info.file_size or crc != info.CRC:
            raise RecoveryGateError(
                f"existing extraction content mismatch at {path}: "
                f"expected_size={info.file_size} actual_size={size} "
                f"expected_crc={info.CRC:08x} actual_crc={crc:08x}"
            )
        evidence.append(
            {
                "path": str(path.resolve()),
                "member": name,
                "bytes": size,
                "crc32": f"{crc:08x}",
                "sha256": sha256_file(path),
            }
        )
    return evidence


def extract_immutable(zip_path: Path, extracted_root: Path) -> dict[str, Any]:
    """Extract through staging, or validate an existing immutable original."""
    zip_path = Path(zip_path)
    extracted_root = Path(extracted_root)
    extracted_root.mkdir(parents=True, exist_ok=True)
    target = extracted_root / zip_path.stem
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        for info in infos:
            _safe_member_name(info.filename)
        if target.exists():
            if not target.is_dir():
                raise RecoveryGateError(
                    f"existing extraction target is not a directory: {target}"
                )
            files = _validate_extracted_tree(target, infos)
            return {
                "status": "validated_existing",
                "extracted_dir": str(target.resolve()),
                "files": files,
            }

        staging = extracted_root / (
            f".{zip_path.stem}.staging.{os.getpid()}.{uuid.uuid4().hex}"
        )
        staging.mkdir(parents=False, exist_ok=False)
        print(f"EXTRACT archive={zip_path} staging={staging}", flush=True)
        try:
            archive.extractall(staging)
            files = _validate_extracted_tree(staging, infos)
        except Exception:
            print(
                f"EXTRACT_FAILED retained_staging={staging}",
                file=sys.stderr,
                flush=True,
            )
            raise
    if target.exists():
        raise RecoveryGateError(
            f"extraction publication race: immutable target appeared: {target}"
        )
    os.rename(staging, target)
    for item in files:
        relative = Path(item["path"]).relative_to(staging.resolve())
        item["path"] = str((target / relative).resolve())
    return {
        "status": "published",
        "extracted_dir": str(target.resolve()),
        "files": files,
    }


def download_archive(
    spec: ArchiveSpec,
    downloads_root: Path,
    *,
    remote_already_verified: bool = False,
    no_proxy: bool = False,
) -> dict[str, Any]:
    downloads_root = Path(downloads_root)
    downloads_root.mkdir(parents=True, exist_ok=True)
    if not remote_already_verified:
        verify_remote_contract(spec)
    final = downloads_root / spec.filename
    partial = downloads_root / f"{spec.filename}.part"
    if final.exists():
        if final.stat().st_size != spec.expected_bytes:
            raise RecoveryGateError(
                f"immutable archive has wrong size: {final}: "
                f"expected={spec.expected_bytes} actual={final.stat().st_size}"
            )
        evidence = inspect_zip(final)
        evidence.update(
            {
                "status": "validated_existing",
                "url": spec.url,
                "expected_bytes": spec.expected_bytes,
            }
        )
        return evidence
    if partial.exists() and partial.stat().st_size > spec.expected_bytes:
        raise RecoveryGateError(
            f"partial archive is larger than pinned source: {partial}"
        )
    command = [
        "wget",
        "-c",
        "--tries=5",
        "--timeout=120",
        "--progress=dot:giga",
    ]
    if no_proxy:
        command.append("--no-proxy")
    command.extend(["-O", str(partial), spec.url])
    print(f"DOWNLOAD_COMMAND {' '.join(command)}", flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RecoveryGateError(
            f"download failed for {spec.filename}: wget exit={completed.returncode}; "
            f"partial retained at {partial}"
        )
    actual = partial.stat().st_size if partial.exists() else 0
    if actual != spec.expected_bytes:
        raise RecoveryGateError(
            f"download size gate failed for {spec.filename}: "
            f"expected={spec.expected_bytes} actual={actual}; "
            f"partial retained at {partial}"
        )
    evidence = inspect_zip(partial)
    if final.exists():
        raise RecoveryGateError(
            f"archive publication race: immutable target appeared: {final}"
        )
    os.rename(partial, final)
    evidence["path"] = str(final.resolve())
    evidence.update(
        {
            "status": "published",
            "url": spec.url,
            "expected_bytes": spec.expected_bytes,
        }
    )
    return evidence


def _select_specs(names: list[str]) -> list[ArchiveSpec]:
    if not names or names == ["all"]:
        return list(ARCHIVES)
    if "all" in names:
        raise RecoveryGateError("--archive all cannot be combined with filenames")
    unknown = sorted(set(names) - set(_BY_FILENAME))
    if unknown:
        raise RecoveryGateError(f"archive is not in the official allowlist: {unknown}")
    return [_BY_FILENAME[name] for name in names]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recover and validate the eight pinned official Alimama ZIPs."
    )
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="FILENAME",
        help="Pinned archive filename; repeat as needed. Default: all eight.",
    )
    parser.add_argument(
        "--downloads-root",
        type=Path,
        default=_REPO_ROOT / "data" / "recovery" / "raw_downloads",
    )
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=_REPO_ROOT / "data" / "recovery" / "extracted",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Terminal JSON manifest path.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download-only", action="store_true")
    mode.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        choices=range(1, 5),
        metavar="{1,2,3,4}",
        help="Bounded download concurrency. Extraction is capped at two.",
    )
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Bypass configured HTTP(S) proxies for archive GET requests.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    started_at = utc_now()
    t0 = time.monotonic()
    payload: dict[str, Any] = {
        "task": "official_alimama_recovery",
        "started_at": started_at,
        "ended_at": None,
        "success": False,
        "archives": [],
        "error": None,
        "allowlist": [asdict(spec) | {"filename": spec.filename} for spec in ARCHIVES],
        "downloads_root": str(args.downloads_root.resolve()),
        "extracted_root": str(args.extracted_root.resolve()),
    }
    exit_code = 1
    try:
        specs = _select_specs(args.archive)
        remote_sizes = {spec.filename: verify_remote_contract(spec) for spec in specs}
        download_evidence: dict[str, dict[str, Any]] = {}
        if not args.extract_only:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(args.max_workers, len(specs))
            ) as executor:
                futures = {
                    executor.submit(
                        download_archive,
                        spec,
                        args.downloads_root,
                        remote_already_verified=True,
                        no_proxy=args.no_proxy,
                    ): spec
                    for spec in specs
                }
                for future in concurrent.futures.as_completed(futures):
                    spec = futures[future]
                    download_evidence[spec.filename] = future.result()

        for spec in specs:
            archive_path = args.downloads_root / spec.filename
            archive = download_evidence.get(spec.filename)
            if archive is None:
                if not archive_path.exists():
                    raise RecoveryGateError(
                        f"extract-only archive is missing: {archive_path}"
                    )
                if archive_path.stat().st_size != spec.expected_bytes:
                    raise RecoveryGateError(
                        f"extract-only size gate failed for {archive_path}"
                    )
                archive = inspect_zip(archive_path)
                archive.update(
                    {
                        "status": "validated_existing",
                        "url": spec.url,
                        "expected_bytes": spec.expected_bytes,
                    }
                )
            record: dict[str, Any] = {
                "filename": spec.filename,
                "kind": spec.kind,
                "level": spec.level,
                "remote_content_length": remote_sizes[spec.filename],
                "archive": archive,
            }
            if not args.download_only:
                record["extraction"] = extract_immutable(
                    archive_path, args.extracted_root
                )
            payload["archives"].append(record)
        payload["success"] = True
        exit_code = 0
    except Exception as exc:  # terminal manifest must survive every gate failure
        payload["error"] = f"{type(exc).__name__}: {exc}"
        print(payload["error"], file=sys.stderr, flush=True)
    finally:
        payload["ended_at"] = utc_now()
        payload["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        payload["exit_code"] = exit_code
        write_json_atomic(args.manifest.resolve(), payload)
        print(f"MANIFEST {args.manifest.resolve()}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
