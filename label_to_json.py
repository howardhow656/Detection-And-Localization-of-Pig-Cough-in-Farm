import numpy as np
import csv
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
import json


'''
最終目標:
{
    "label": 2,
    "start_times": [t1, t2, t3],
    'position': 1-2
}
'''
def load_data(path):
    file = pd.read_csv(path)
    df = file.copy()
    df["Start"] = df["Start"].astype(float)
    df["End"] = df["End"].astype(float)
    df["Label"] = df["Label"].astype(int)
    df["Position"] = df["Position"].astype(str)
    return df

def merge_events_from_all_mics(
    ann_tables: List[pd.DataFrame],
    mic_ids: List[int],
    valid_labels=(1, 2, 3, 4),
):
    merged = []

    for df, mic_id in zip(ann_tables, mic_ids):
        df = df[df["Label"].isin(valid_labels)].copy()

        for _, row in df.iterrows():
            item = {
                "mic_id": mic_id,
                "start": float(row["Start"]),
                "end": float(row["End"]),
                "label": int(row["Label"]),
            }
            if "Position" in row.index:
                item["position"] = row["Position"]

            merged.append(item)

    return merged

def cluster_events_across_mics(
    merged_events: List[Dict[str, Any]],
    n_mics: int = 3,
    max_time_diff: float = 0.35,
    min_mics_per_event: int = 2,
) -> List[Dict[str, Any]]:
    """
    把來自不同 mic 的事件依 label + start time 做 clustering。
    
    規則：
    - 只會把同 label 的事件聚在一起
    - 同一 cluster 裡同一支 mic 只能出現一次
    - cluster 內 start 差異不能超過 max_time_diff
    - 至少有 min_mics_per_event 支 mic 才保留
    """
    # 先依 label, start 排序
    merged_events = sorted(merged_events, key=lambda x: (x["label"], x["start"]))

    clusters = []

    for ev in merged_events:
        assigned = False

        for cluster in clusters:
            # label 不同不能放一起
            if cluster["label"] != ev["label"]:
                continue

            # 同一支 mic 不能重複
            if ev["mic_id"] in cluster["mic_ids"]:
                continue

            # 看時間是否接近
            cluster_starts = [e["start"] for e in cluster["events"]]
            test_starts = cluster_starts + [ev["start"]]

            if max(test_starts) - min(test_starts) <= max_time_diff:
                cluster["events"].append(ev)
                cluster["mic_ids"].add(ev["mic_id"])
                assigned = True
                break

        if not assigned:
            clusters.append({
                "label": ev["label"],
                "events": [ev],
                "mic_ids": {ev["mic_id"]},
            })

    # 只保留至少有 2 支 mic 的 cluster
    filtered = []
    for cluster in clusters:
        if len(cluster["mic_ids"]) < min_mics_per_event:
            continue

        start_times = [None] * n_mics
        end_times = [None] * n_mics

        for ev in cluster["events"]:
            start_times[ev["mic_id"]] = ev["start"]
            end_times[ev["mic_id"]] = ev["end"]

        item = {
            "label": cluster["label"],
            "start_times": start_times,
            "end_times": end_times,
        }

        if "position" in cluster["events"][0]:
            item["position"] = cluster["events"][0]["position"]

        filtered.append(item)

    # 依最早有效 start 排序
    filtered = sorted(
        filtered,
        key=lambda x: min([t for t in x["start_times"] if t is not None])
    )
    return filtered

def build_samples_from_clusters(
    clustered_events: List[Dict[str, Any]],
    audio_paths: List[str],
) -> List[Dict[str, Any]]:
    samples = []

    for ev in clustered_events:
        samples.append({
            "audio_paths": audio_paths,
            "label": ev["label"],
            "start_times": ev["start_times"],   # 可能含 None
        })

    return samples




path1 = Path(r'')
path2 = Path(r'')
path3 = Path(r'')
mic1 = load_data(path1)
mic2 = load_data(path2)
mic3 = load_data(path3)

df = merge_events_from_all_mics(ann_tables=[mic1, mic2, mic3], mic_ids=[1,2,3], valid_labels=(1, 2, 3, 4))
clustered = cluster_events_across_mics(
    merged_events=df,
    n_mics=3,
    max_time_diff=0.35,
    min_mics_per_event=2,   # 至少 2 支 mic 才保留
)

samples = build_samples_from_clusters(
    clustered_events=clustered,
    audio_paths=[path1, path2, path3]
)


with open("samples.json", "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=2, ensure_ascii=False)