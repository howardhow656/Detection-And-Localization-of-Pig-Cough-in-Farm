import tensorflow as tf
import webrtcvad
import os
import soundfile as sf
import librosa
import sound_detection as sd
import numpy as np
from datetime import timedelta
import audio_to_image as AtI
import json
from glob import glob
import csv 
from pathlib import Path
import time
import package 





webrtc_vad = webrtcvad.Vad()
webrtc_vad.set_mode(0)

#使用vad找出段落

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



frame_length = 10

def formattime(second:float)->str:
    return str(timedelta(seconds=second))



def vad_using(path):
    audio = path
    file_base_name = os.path.splitext(os.path.basename(audio))[0]
    y , sr = sf.read(audio)
    y_bandpass = package.bandpass_filter(y , sr, low_cut=0, high_cut=2500)
    if y_bandpass.ndim > 1:
        y_bandpass = np.mean(y_bandpass ,axis=1)
    target_rate = 48000
    y_resampled = np.clip(librosa.resample(y_bandpass, orig_sr=sr, target_sr=target_rate), -1, 1 )
    y_int16 = (y_resampled * 32767).astype(np.int16)

    base_dir = Path(r"C:\Users\howardhow\Desktop\sound detection\data")
    out_dir = base_dir / file_base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    speech_list = []
    for frame, start in package.split_to_frame(y_int16, target_rate ,frame_length=frame_length):
        if webrtc_vad.is_speech(frame, target_rate):
            speech_list.append([float(start/target_rate),float((start/target_rate+frame_length/1000))])

    merged = []
    raw_time = []
    if not speech_list:
        print("No speech detected")
        merged = []
    else:
        start, end = speech_list[0]
        for s ,e in speech_list[1:]:
            raw_time.append([s,e])
            if s - end < 0.5:
                end = e
            else:
                merged.append([start,end])
                start,end = s,e
        merged.append([start,end])

    min_duration = 0.2 #s
    merged = [[s,e] for [s,e] in merged if (e - s) >= min_duration]

    with open(out_dir / "merge.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Start",  "End"])
        for s, e in merged:
            w.writerow([f"{s:.6f}", f"{e:.6f}"])

    new_merge, compare_list = package.modified_time(merged ,y_resampled, target_rate)

    with open(out_dir / "label.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(['Index',"Start",  "End", 'Label', 'Probability'])
        for s, e, index in new_merge:
            w.writerow([index, f"{s:.6f}", f"{e:.6f}", "", ""])#取+-1，比較好聽出是三小


    package.modified_label(new_merge ,y ,sr ,out_dir ,file_base_name)
    return merged, y_resampled, target_rate, out_dir, file_base_name, compare_list

merged, y_resampled, target_rate, out_dir, file_base_name, compare_list = vad_using(path=r"C:\Users\howardhow\Desktop\sound detection\data\rec0817-020007.wav")




print("wav count =", len(list(out_dir.glob("*.wav"))), flush=True)



#使用model判斷是否為咳嗽聲
model = tf.keras.models.load_model(r"C:\Users\howardhow\Desktop\sound detection\cough_cnn_model.h5")


identify_list = []
with open(r'C:\Users\howardhow\Desktop\sound detection\model_config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

n_mels = config["n_mels"]
hop_length = config["hop_length"]
T_fix = config["T_fix"]
sr = config["sample_rate"]
use_audio_norm = config["normalize_audio"]
use_mfcc_norm = config["normalize_mfcc"]
bp_cfg = config["bandpass"]

def predict_segment(wav_path):#要改成完整掃過每個audio黨
    """
    wav_path: 音檔路徑
    start/end: 秒數
    """
    y, sr = sf.read(wav_path, dtype='float32')

    y = AtI.bandpass_filter(y , sr)
    if use_audio_norm:
        y = y / (np.max(np.abs(y)) + 1e-8)

    mfcc = AtI.log_mel(y, sr, n_mels=n_mels, hop_length=hop_length)
    if use_mfcc_norm:
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    if mfcc.shape[1] < T_fix:
        pad = T_fix - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :T_fix]

    X = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)


    t0 = time.time()
    prob = model.predict(X)[0][0]
    label = 1 if prob >= 0.6 else 0

    return label, prob


with open(out_dir /"label.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

row_dict = { int(r["Index"]): r for r in rows }

# ③ 對每一段 1.5 秒音檔做預測

for audio in (out_dir.glob("*.wav")):
    audio = str(audio)
    num = int(Path(audio).stem.split("_")[-1])
    label, prob = predict_segment(audio)

    if num in row_dict:
        row_dict[num]["Label"] = str(label)
        row_dict[num]["Probability"] = f"{prob:.4f}"
    else:
        print(f"[警告] {file_base_name}Index: {num} 在 CSV 找不到對應的列，因此跳過。")

    

fieldnames = rows[0].keys()   

with open(out_dir / "label.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("完成：成功更新 label / probability！")


#回推到咳嗽聲的
#假如為咳嗽聲，從label.csv回去找原本的準確時間，新開一個檔案來找填上正確的時間以及probability

indices = []

with open(out_dir / "label.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Label"] == "0":
            indices.append({
                "Index": row["Index"],
                "Probability": row["Probability"],
            })

target = { int(r["Index"]) for r in indices }

matched = [
    { 
        "orig_index": m["orig_index"],
        "orig_s": m["orig_s"],
        "orig_e": m["orig_e"],
        'new_s': m['new_s'],
        'new_e':m['new_e'],
        "mode": m["mode"]
    }
    for m in compare_list
    if m["orig_index"] in target
]

index_prob_map = { r["Index"]: r["Probability"] for r in indices }
combined = []

for m in matched:
    idx = str(m["orig_index"])   # 確保比對用字串

    if idx in index_prob_map:
        combined.append({
            "Index": idx,
            "Prob": index_prob_map[idx],
            "orig_s": m["orig_s"],
            "orig_e": m["orig_e"],
            "new_s": m["new_s"],
            "new_e": m["new_e"],
            "mode": m["mode"]
        })


with open(out_dir / "final_cough_detection.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Index", "True_Start", "True_End", "Probability", "Modified_Start", "Modified_End"])

    for row in combined:
        w.writerow([
            row["Index"],
            round(row["orig_s"], 6),
            round(row["orig_e"], 6),
            float(row["Prob"]),
            round(row["new_s"], 6),
            round(row["new_e"], 6),
        ])



