#!/usr/bin/env python3

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def _format_size(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}TB"


def _format_speed(num_bytes_per_sec: float) -> str:
    return f"{_format_size(num_bytes_per_sec)}/s"


def _download_with_aria2(url: str, output_path: str) -> bool:
    aria2c = shutil.which("aria2c")
    backend = os.environ.get("EPIC_DL_BACKEND", "aria2c").strip().lower()
    if backend == "urllib":
        return False
    if aria2c is None:
        return False

    connections = int(os.environ.get("EPIC_DL_CONNECTIONS", "16"))
    min_split_size = os.environ.get("EPIC_DL_MIN_SPLIT_SIZE", "10M")
    summary_interval = os.environ.get("EPIC_DL_SUMMARY_INTERVAL", "1")

    output_dir = os.path.dirname(output_path) or "."
    output_name = os.path.basename(output_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        aria2c,
        "--continue=true",
        "--allow-overwrite=true",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        "--console-log-level=warn",
        f"--summary-interval={summary_interval}",
        f"--max-connection-per-server={connections}",
        f"--split={connections}",
        f"--min-split-size={min_split_size}",
        f"--dir={output_dir}",
        f"--out={output_name}",
        url,
    ]

    print("Downloading with aria2c")
    print(f"from {url}")
    print(f"to {output_path}")
    print(
        "aria2 settings: "
        f"connections={connections}, min_split_size={min_split_size}, summary_interval={summary_interval}s"
    )
    subprocess.run(cmd, check=True)
    return True


def _download_with_progress(url: str, output_path: str, chunk_size: int = 1024 * 1024) -> None:
    if _download_with_aria2(url, output_path):
        return

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)

    print("Downloading with urllib")
    print(f"from {url}")
    print(f"to {output_path}")
    start = time.time()
    last_log = start
    downloaded = 0

    with urllib.request.urlopen(url) as response, open(output_path, "wb") as output_file:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total is not None else None

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_log >= 0.5:
                elapsed = max(now - start, 1e-6)
                speed = downloaded / elapsed
                if total_bytes:
                    pct = 100.0 * downloaded / total_bytes
                    msg = (
                        f"\r  {pct:6.2f}%  "
                        f"{_format_size(downloaded)} / {_format_size(total_bytes)}  "
                        f"{_format_speed(speed)}"
                    )
                else:
                    msg = f"\r  {_format_size(downloaded)}  {_format_speed(speed)}"
                print(msg, end="", flush=True)
                last_log = now

    elapsed = max(time.time() - start, 1e-6)
    avg_speed = downloaded / elapsed
    if downloaded > 0:
        if total_bytes:
            print(
                f"\r  100.00%  {_format_size(downloaded)} / {_format_size(total_bytes)}  "
                f"{_format_speed(avg_speed)}"
            )
        else:
            print(f"\r  {_format_size(downloaded)}  {_format_speed(avg_speed)}")
    else:
        print("  0B  0B/s")


def _load_epic_module(downloader_root: str):
    module_path = os.path.join(downloader_root, "epic_downloader.py")
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Cannot find epic_downloader.py under {downloader_root}")

    spec = importlib.util.spec_from_file_location("epic_downloader_local", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloader-root", required=True, help="Path to epic-kitchens-download-scripts")
    parser.add_argument("epic_args", nargs=argparse.REMAINDER, help="Arguments passed to epic_downloader.py")
    args = parser.parse_args()

    epic_args = list(args.epic_args)
    if epic_args and epic_args[0] == "--":
        epic_args = epic_args[1:]

    downloader_root = os.path.abspath(args.downloader_root)
    module = _load_epic_module(downloader_root)
    module.EpicDownloader.download_file = staticmethod(_download_with_progress)

    old_cwd = os.getcwd()
    old_argv = sys.argv[:]
    try:
        os.chdir(downloader_root)
        sys.argv = ["epic_downloader.py"] + epic_args
        parser = module.create_parser()
        parsed = module.parse_args(parser)
        module.print_header("*** Welcome to the EPIC Kitchens Downloader! ***")
        downloader = module.EpicDownloader(base_output=parsed.output_path, errata_only=parsed.errata)
        downloader.download(
            what=parsed.what,
            participants=parsed.participants,
            specific_videos=parsed.specific_videos,
            splits=parsed.splits,
            challenges=parsed.challenges,
            extension_only=parsed.extension_only,
            epic55_only=parsed.epic55_only,
        )
        module.print_header("*** All done, bye! ***")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
