import os
import shlex
import subprocess
import time
import fcntl


def get_visible_gpu_count(cuda_visible_devices=None):
    cuda_visible = cuda_visible_devices
    if cuda_visible is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_visible = str(cuda_visible).strip()
    if not cuda_visible:
        return 1
    gpu_ids = [gpu_id.strip() for gpu_id in cuda_visible.split(",") if gpu_id.strip() and gpu_id.strip() != "-1"]
    return max(len(gpu_ids), 1)


def print_gpu_memory_debug(cuda_visible_devices=None):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"[GPU_MEM_DEBUG] skip: cannot query nvidia-smi ({exc})")
        return

    cuda_visible = cuda_visible_devices
    if cuda_visible is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_visible = str(cuda_visible).strip()

    visible_indices = None
    if cuda_visible:
        visible_indices = {gpu_id.strip() for gpu_id in cuda_visible.split(",") if gpu_id.strip() and gpu_id.strip() != "-1"}

    print("[GPU_MEM_DEBUG] per-gpu memory usage:")
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        gpu_idx, used_str, total_str = parts
        if visible_indices is not None and gpu_idx not in visible_indices:
            continue
        try:
            used = float(used_str)
            total = float(total_str)
            ratio = 100.0 * used / total if total > 0 else 0.0
            print(f"  GPU {gpu_idx}: {used:.0f} MiB / {total:.0f} MiB ({ratio:.2f}%)")
        except ValueError:
            print(f"  GPU {gpu_idx}: used={used_str}, total={total_str}")


def resolve_ads_cli_bin(ads_cli_bin="", ads_cli_dir="", default="./ads-cli"):
    cli_bin = str(ads_cli_bin or "").strip()
    if cli_bin:
        return cli_bin

    cli_dir = str(ads_cli_dir or "").strip()
    if cli_dir:
        return cli_dir if cli_dir.endswith("/ads-cli") else os.path.join(cli_dir, "ads-cli")
    return default


def _is_remote_ckpt_path(path):
    return path.startswith("oss://") or path.startswith("s3://")


def resolve_model_weights(oss_ckpt_remote_dir="", ads_cli_bin="./ads-cli"):
    # Priority:
    # 1) GDRN_MODEL_WEIGHTS: direct local/remote path
    # 2) GDRN_WEIGHTS_OSS_PATH: direct oss/s3 path
    # 3) GDRN_WEIGHTS_NAME + GDRN_CKPT_OSS_DIR: resolve from checkpoint sync remote dir
    weights = os.environ.get("GDRN_MODEL_WEIGHTS", "").strip()
    if not weights:
        weights = os.environ.get("GDRN_WEIGHTS_OSS_PATH", "").strip()

    weights_name = os.environ.get("GDRN_WEIGHTS_NAME", "").strip()
    if not weights and weights_name and oss_ckpt_remote_dir:
        weights = "{}/{}".format(str(oss_ckpt_remote_dir).rstrip("/"), weights_name)

    if not weights:
        return ""

    if not _is_remote_ckpt_path(weights):
        return weights

    cache_dir = os.environ.get("GDRN_WEIGHTS_CACHE_DIR", "/tmp/gdrn_weights_cache").strip() or "/tmp/gdrn_weights_cache"
    os.makedirs(cache_dir, exist_ok=True)

    remote_name = os.path.basename(weights.rstrip("/"))
    local_path = os.path.join(cache_dir, remote_name)
    if os.path.exists(local_path):
        print(f"[WEIGHTS_OSS] Use cached checkpoint: {local_path}")
        return local_path

    lock_path = f"{local_path}.lock"

    timeout_sec = int(os.environ.get("GDRN_WEIGHTS_FETCH_TIMEOUT_SEC", "1800"))
    retry_times = int(os.environ.get("GDRN_WEIGHTS_FETCH_RETRY_TIMES", "3"))
    retry_interval_sec = int(os.environ.get("GDRN_WEIGHTS_FETCH_RETRY_INTERVAL_SEC", "20"))
    fetch_cmd_template = os.environ.get(
        "GDRN_CKPT_FETCH_CMD",
        f"{ads_cli_bin} --threads=32 cp {{remote_path_q}} {{local_path_q}}",
    )
    fetch_cmd = fetch_cmd_template.format(
        remote_path=weights,
        local_path=local_path,
        remote_path_q=shlex.quote(weights),
        local_path_q=shlex.quote(local_path),
    )

    # Use a local file lock to ensure only one rank/process fetches the file.
    # Other ranks wait and then reuse the same cached checkpoint.
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            if os.path.exists(local_path):
                print(f"[WEIGHTS_OSS] Use cached checkpoint after lock: {local_path}")
                return local_path

            for attempt in range(1, retry_times + 1):
                print(f"[WEIGHTS_OSS] Fetch checkpoint (attempt {attempt}/{retry_times}): {weights} -> {local_path}")
                result = subprocess.run(fetch_cmd, shell=True, capture_output=True, text=True, timeout=timeout_sec)
                if result.returncode == 0 and os.path.exists(local_path):
                    print(f"[WEIGHTS_OSS] Checkpoint fetched: {local_path}")
                    return local_path

                print(
                    "[WEIGHTS_OSS] Fetch failed, cmd: {}, stderr: {}".format(
                        fetch_cmd,
                        (result.stderr or "").strip()[-500:],
                    )
                )
                if attempt < retry_times:
                    time.sleep(retry_interval_sec)
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    raise RuntimeError(f"[WEIGHTS_OSS] Failed to fetch remote checkpoint: {weights}")
