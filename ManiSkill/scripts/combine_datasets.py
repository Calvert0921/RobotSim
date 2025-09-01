#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine two LeRobot v2.* datasets (SmolVLA-compatible) into one.

Default behavior:
- Merge meta/episodes_stats.jsonl directly from sources (preserves image stats keys).
  Only adjust episode_index for the second dataset.
- If sources lack stats, you can add --make_episode_stats to recompute numeric-only stats.

Also:
- Merge tasks.jsonl (dedupe by text), remap task_index per frame.
- Reindex episodes, rewrite episode_index + task_index in Parquet.
- Write meta/episodes.jsonl with both 'length' and 'num_frames'.
- Copy meta/info.json (optionally recompute totals).
- Optional copy & rename videos.
- Optional push to HF and create tag == info.json['codebase_version'].
"""

import argparse
import contextlib
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

# --------------------- file utils ---------------------

def load_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def read_tasks_jsonl(tasks_path: Path) -> Dict[int, str]:
    tasks = {}
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tasks[int(row["task_index"])] = row["task"]
    return tasks

def write_tasks_jsonl(idx2text: Dict[int, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for idx in sorted(idx2text.keys()):
            f.write(json.dumps({"task_index": int(idx), "task": idx2text[idx]}, ensure_ascii=False) + "\n")

def append_episode_jsonl(meta_dir: Path, episode_index: int, num_frames: int, tasks_in_episode: List[str]) -> None:
    ep_path = meta_dir / "episodes.jsonl"
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ep_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "episode_index": int(episode_index),
            "num_frames": int(num_frames),
            "length": int(num_frames),   # REQUIRED by current loader
            "tasks": sorted(list(set(tasks_in_episode))),
        }, ensure_ascii=False) + "\n")

def ensure_same_schema(info_a: dict, info_b: dict):
    keys = ["fps", "features", "cameras", "robot_type"]
    for k in keys:
        if info_a.get(k) != info_b.get(k):
            raise ValueError(f"[Schema mismatch] '{k}' differs.\nA: {info_a.get(k)}\nB: {info_b.get(k)}")

def list_episode_parquets(data_root: Path) -> List[Path]:
    return sorted(data_root.rglob("episode_*.parquet"))

def chunk_dir_for_episode(ep_idx: int) -> str:
    return f"chunk-{ep_idx // 1000:03d}"

def episode_index_from_path(p: Path) -> int:
    return int(p.stem.split("_")[1])

def copy_and_rename_videos_if_exist(info_meta: dict, src_root: Path, dst_root: Path, src_ep: int, dst_ep: int) -> int:
    cameras = info_meta.get("cameras", [])
    copied = 0
    for cam in cameras:
        src = src_root / "videos" / chunk_dir_for_episode(src_ep) / cam / f"episode_{src_ep:06d}.mp4"
        if not src.exists():
            continue
        dst_dir = dst_root / "videos" / chunk_dir_for_episode(dst_ep) / cam
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"episode_{dst_ep:06d}.mp4"
        shutil.copy2(src, dst)
        copied += 1
    return copied

# ----------------- numeric stats fallback -----------------

def is_numeric_like(value) -> bool:
    if isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
        v0 = np.asarray(value[0])
        return v0.ndim == 0
    return False

def detect_numeric_keys(info: dict, sample_row: dict) -> List[str]:
    numeric = []
    for k, ft in (info.get("features") or {}).items():
        if (ft or {}).get("dtype") in ["vector","sequence","array","float","float_vector","number"]:
            numeric.append(k)
    for k, v in sample_row.items():
        if k not in numeric and is_numeric_like(v):
            numeric.append(k)
    for k in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
        if k not in numeric and k in sample_row:
            numeric.append(k)
    seen, out = set(), []
    for k in numeric:
        if k not in seen:
            seen.add(k); out.append(k)
    return out

def to_2d_float_array(column_pylist: List) -> np.ndarray:
    rows, maxd = [], 0
    for v in column_pylist:
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=float).reshape(-1)
        else:
            arr = np.asarray([v], dtype=float)
        rows.append(arr)
        maxd = max(maxd, arr.shape[0])
    out = []
    for r in rows:
        if r.shape[0] < maxd:
            r = np.pad(r, (0, maxd - r.shape[0]), mode="constant")
        out.append(r)
    return np.stack(out, axis=0)  # [T, D]

def robust_stats_as_lists(arr: np.ndarray) -> dict:
    if arr.ndim == 1:
        arr = arr[:, None]
    mean = np.nanmean(arr, axis=0).tolist()
    std  = np.nanstd(arr,  axis=0).tolist()
    mn   = np.nanmin(arr,  axis=0).tolist()
    mx   = np.nanmax(arr,  axis=0).tolist()
    def clean(xs):
        out = []
        for v in xs:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                out.append(0.0)
            else:
                out.append(float(v))
        return out
    return {"min": clean(mn), "max": clean(mx), "mean": clean(mean), "std": clean(std), "count": [int(arr.shape[0])]}

# --------------- merge stats from sources ----------------

def fix_stats_shapes_inplace(stats_obj: Dict) -> None:
    """Ensure min/max/mean/std are lists; count is [N]."""
    if not isinstance(stats_obj, dict):
        return
    for k in ["min", "max", "mean", "std"]:
        if k in stats_obj:
            v = stats_obj[k]
            if isinstance(v, list):
                if len(v) == 0:
                    stats_obj[k] = [0.0]
            else:
                try: stats_obj[k] = [float(v)]
                except Exception: stats_obj[k] = [0.0]
    c = stats_obj.get("count", None)
    if isinstance(c, list):
        if len(c) == 0: stats_obj["count"] = [1]
    elif isinstance(c, (int, float)):
        stats_obj["count"] = [int(c)]
    else:
        stats_obj["count"] = [1]

# --------------------- main combine ---------------------

def combine_two(
    src_repo_a: str,
    src_repo_b: str,
    dest_repo: str,
    dest_root: Path,
    copy_videos: bool,
    recompute_info: bool,
    make_episode_stats: bool,  # fallback, only if sources lack stats
    push: bool,
    make_tag: bool,
):
    api = HfApi()

    # 1) download sources
    local_a = Path(snapshot_download(src_repo_a, repo_type="dataset"))
    local_b = Path(snapshot_download(src_repo_b, repo_type="dataset"))

    meta_a, data_a = local_a / "meta", local_a / "data"
    meta_b, data_b = local_b / "meta", local_b / "data"
    info_a, info_b = load_json(meta_a / "info.json"), load_json(meta_b / "info.json")
    ensure_same_schema(info_a, info_b)

    # 2) prepare dest layout
    dest_root = Path(dest_root).expanduser() / dest_repo
    meta_dst, data_dst, videos_dst = dest_root / "meta", dest_root / "data", dest_root / "videos"
    for p in [meta_dst, data_dst, videos_dst]:
        p.mkdir(parents=True, exist_ok=True)

    # 3) merge tasks
    tasks_a = read_tasks_jsonl(meta_a / "tasks.jsonl")
    tasks_b = read_tasks_jsonl(meta_b / "tasks.jsonl")
    all_texts: List[str] = []
    for d in [tasks_a, tasks_b]:
        for _, t in sorted(d.items()):
            if t not in all_texts:
                all_texts.append(t)
    text_to_new: Dict[str, int] = {t: i for i, t in enumerate(all_texts)}
    new_to_text: Dict[int, str] = {i: t for t, i in text_to_new.items()}
    write_tasks_jsonl(new_to_text, meta_dst / "tasks.jsonl")

    # 4) info copy
    info_out = info_a.copy()

    # 5) rewrite Parquets (task_index remap + episode_index offset), copy videos, write episodes.jsonl
    def map_task_index_column(table: pa.Table, old_idx_to_text: Dict[int, str]) -> pa.Table:
        old = table.column("task_index").to_pylist()
        new = [text_to_new[old_idx_to_text[int(i)]] for i in old]
        arr = pa.array(new, type=pa.int64())
        return table.set_column(table.schema.get_field_index("task_index"), "task_index", arr)

    def rewrite_int_column(table: pa.Table, colname: str, value: int) -> pa.Table:
        new_col = pa.array([int(value)] * table.num_rows, type=pa.int64())
        return table.set_column(table.schema.get_field_index(colname), colname, new_col)

    def process_one(local_dir: Path, info_meta: dict, old_tasks_map: Dict[int, str], ep_offset: int) -> Tuple[int,int,int,dict]:
        data_src = local_dir / "data"
        ep_paths = list_episode_parquets(data_src)
        frames_sum = 0
        videos_cnt = 0

        table0 = pq.read_table(ep_paths[0])
        sample0 = {k: table0[k][0].as_py() for k in table0.column_names}

        for p in ep_paths:
            src_ep = episode_index_from_path(p)
            dst_ep = src_ep + ep_offset
            table = pq.read_table(p)

            table = map_task_index_column(table, old_tasks_map)
            table = rewrite_int_column(table, "episode_index", dst_ep)

            dst_chunk = data_dst / chunk_dir_for_episode(dst_ep)
            dst_chunk.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dst_chunk / f"episode_{dst_ep:06d}.parquet")

            num_rows = table.num_rows
            frames_sum += num_rows

            task_idxs = sorted(set(table.column("task_index").to_pylist()))
            tasks_in_episode = [new_to_text[i] for i in task_idxs]
            append_episode_jsonl(meta_dst, dst_ep, num_rows, tasks_in_episode)

            if copy_videos:
                videos_cnt += copy_and_rename_videos_if_exist(info_meta, local_dir, dest_root, src_ep, dst_ep)

        return len(ep_paths), frames_sum, videos_cnt, sample0

    nA, framesA, vidsA, sampleA = process_one(local_a, info_a, tasks_a, ep_offset=0)
    nB, framesB, vidsB, sampleB = process_one(local_b, info_b, tasks_b, ep_offset=nA)

    total_eps = nA + nB
    total_frames = framesA + framesB
    total_videos = vidsA + vidsB

    # 6) episodes_stats.jsonl
    srcA_stats = meta_a / "episodes_stats.jsonl"
    srcB_stats = meta_b / "episodes_stats.jsonl"
    if srcA_stats.exists() and srcB_stats.exists():
        # default path: merge sources directly (preserves image keys)
        out_path = meta_dst / "episodes_stats.jsonl"
        with open(out_path, "w", encoding="utf-8") as fout:
            # A: as-is
            with open(srcA_stats, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    # normalize shapes: count->[N], stats lists
                    for feat, st in (obj.get("stats") or {}).items():
                        fix_stats_shapes_inplace(st)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            # B: add episode_index offset
            with open(srcB_stats, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    obj["episode_index"] = int(obj["episode_index"]) + nA
                    for feat, st in (obj.get("stats") or {}).items():
                        fix_stats_shapes_inplace(st)
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    elif make_episode_stats:
        # fallback: recompute numeric-only stats (no image keys)
        numeric_keys = detect_numeric_keys(info_out, sampleA if sampleA else sampleB)
        out_path = meta_dst / "episodes_stats.jsonl"
        with open(out_path, "w", encoding="utf-8") as fout:
            for ep_p in list_episode_parquets(data_dst):
                ep_idx = episode_index_from_path(ep_p)
                table = pq.read_table(ep_p)
                stats = {}
                for key in numeric_keys:
                    if key not in table.column_names:
                        continue
                    col_py = table[key].to_pylist()
                    try:
                        arr = to_2d_float_array(col_py)
                    except Exception:
                        continue
                    s = robust_stats_as_lists(arr)
                    for kk in ["min","max","mean","std"]:
                        if isinstance(s[kk], list) and len(s[kk]) == 0:
                            s[kk] = [0.0]
                    stats[key] = s
                fout.write(json.dumps({"episode_index": ep_idx, "stats": stats}, ensure_ascii=False) + "\n")
    else:
        raise FileNotFoundError(
            "Neither source has meta/episodes_stats.jsonl and --make_episode_stats was not set."
        )

    # 7) finalize info.json
    if recompute_info:
        info_out["total_episodes"] = int(total_eps)
        info_out["total_frames"] = int(total_frames)
        info_out["total_videos"] = int(total_videos)
    save_json(info_out, meta_dst / "info.json")

    print(f"[OK] Merged at: {dest_root}")
    print(f"     Episodes: {total_eps}, Frames: {total_frames}, Videos copied: {total_videos}")

    # 8) push & tag
    if push:
        api.create_repo(repo_id=dest_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(repo_id=dest_repo, repo_type="dataset", folder_path=str(dest_root))
        print(f"[OK] Pushed to https://huggingface.co/datasets/{dest_repo}")

        if make_tag:
            codebase_version = info_out.get("codebase_version")
            if not codebase_version:
                print("[WARN] info.json has no 'codebase_version'; skip tagging.")
            else:
                with contextlib.suppress(Exception):
                    api.delete_tag(repo_id=dest_repo, tag=codebase_version, repo_type="dataset")
                api.create_tag(repo_id=dest_repo, tag=codebase_version, repo_type="dataset")
                print(f"[OK] Created tag: {codebase_version}")

# ----------------------- CLI -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="Source dataset repo_id on HF. Provide exactly two --src entries.")
    ap.add_argument("--dest_repo", required=True,
                    help="Destination dataset repo_id (e.g. yourname/SmolVLA_BlueRed_2000)")
    ap.add_argument("--dest_root", default="~/.cache/huggingface/lerobot_merged",
                    help="Local root to build the merged dataset under.")
    ap.add_argument("--copy_videos", action="store_true", help="Copy & rename videos if present.")
    ap.add_argument("--recompute_info", action="store_true", help="Update totals in info.json.")
    ap.add_argument("--make_episode_stats", action="store_true",
                    help="Fallback: recompute stats from Parquets for numeric features (use only if sources lack stats).")
    ap.add_argument("--push", action="store_true", help="Push merged dataset to HF.")
    ap.add_argument("--tag", action="store_true",
                    help="After push, create tag == codebase_version in info.json.")
    args = ap.parse_args()

    if len(args.src) != 2:
        raise SystemExit("Please provide exactly two --src entries.")

    combine_two(
        src_repo_a=args.src[0],
        src_repo_b=args.src[1],
        dest_repo=args.dest_repo,
        dest_root=Path(args.dest_root),
        copy_videos=args.copy_videos,
        recompute_info=args.recompute_info,
        make_episode_stats=args.make_episode_stats,
        push=args.push,
        make_tag=args.tag,
    )

if __name__ == "__main__":
    main()
