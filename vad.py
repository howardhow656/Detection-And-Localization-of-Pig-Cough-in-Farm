import sound_detection as sd
from pathlib import Path
import soundfile as sf
import csv
import librosa 
from datetime import timedelta
import numpy as np
import webrtcvad
from collections import defaultdict
import os

audio = r"C:\Users\howardhow\Desktop\sound detection\data\rec0812-220003.wav"
file_base_name = os.path.splitext(os.path.basename(audio))[0]

    
y , sr = sf.read(audio)
y_bandpass = sd.bandpass_filter(y , sr, low_cut=0, high_cut=2500)
if y_bandpass.ndim > 1:
    y_bandpass = np.mean(y_bandpass ,axis=1)
target_rate = 48000
y_resampled = np.clip(librosa.resample(y_bandpass, orig_sr=sr, target_sr=target_rate), -1, 1 )
y_int16 = (y_resampled * 32767).astype(np.int16)
vad = webrtcvad.Vad()
vad.set_mode(0)

def split_to_frame(audio_file , sample_rate , frame_length):
    n = int(sample_rate * frame_length/1000)
    start = 0
    while start + n <= len(audio_file):
        frame = audio_file[start:start+n]
        yield frame.tobytes(), start
        start += n

frame_length = 30
speech_list = []
out_dir = os.path.join(r"C:\Users\howardhow\Desktop\sound detection\data",f'{file_base_name}')

out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True) 
for frame, start in split_to_frame(y_int16, target_rate ,frame_length=frame_length):
    if vad.is_speech(frame, target_rate):
        speech_list.append([float(start/target_rate),float((start/target_rate+frame_length/1000))])

def formattime(second:float)->str:
    return str(timedelta(seconds=second))


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
        w.writerow(["'"+formattime(s),"'"+formattime(e)])#取+-1，比較好聽出是三小

print('finish!!')


#------------------------------------------------
'''
把聲音修剪成想要的格式(1s):
a. 太短:找前後來補
b. 太長:只要能滿1秒就切，一直切到不夠
'''

def modified_time(merged):
    new = []
    for s, e in merged:
        if e - s < 1.5:
            e = s + 1.2
            s = s - 0.3
            new.append([s,e])
        elif e - s == 1.5:
            new.append([s,e])
        while e - s > 1.5:
            new_e = s + 1.5
            new.append([s,new_e])
            s = new_e
            if 0.4 < e - s <= 1.5:
                e =  s + 1.5
                if e <= len(y_resampled)/target_rate:
                    new.append([s,e])
                continue
    return new

new_merge = modified_time(merged)

with open(out_dir / "label.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(['Number',"Start",  "End", 'Label'])
    i = 1
    for s, e in new_merge:
        w.writerow([i,"'"+formattime(s),"'"+formattime(e),""])#取+-1，比較好聽出是三小
        i += 1

#把要label的段落全部抓出來並儲存
def modified_label(new_merge):
    i = 1
    n_samples = len(y)
    for s , e in new_merge:
        start_idx = int(round(s * sr))
        end_idx   = int(round(e * sr))
        # 邊界保護
        start_idx = max(0, start_idx)
        end_idx   = min(n_samples, end_idx)
        if end_idx <= start_idx:
            continue
        audio_segment = y[start_idx : end_idx]
        output = os.path.join(out_dir,f'{file_base_name}_{i}.wav') 
        sf.write(output,audio_segment,sr)
        i += 1

modified_label(new_merge)
# --------------------------------------------------
# comparing with the label
# labels = r'C:\Users\howardhow\Desktop\sound detection\新增資料夾\train_label.tsv'
# train_label_map = defaultdict(list)


# with open(labels, newline='', encoding='utf-8') as f:
#     for row in csv.DictReader(f, delimiter='\t'):
#         key = Path(row['filename']).name.strip().lower()
#         try:
#             s = float(row['onset']); e=float(row['offset'])
#         except ValueError:
#             s = sd._parse_time_to_seconds(row['onset'])
#             e = sd._parse_time_to_seconds(row['offset'])
#         train_label_map[key].append([s,e])

# def match_onset_on_nearest_one(pred_on, gt_on, tol_s = 1):
#     pred = sorted((t, i) for i, t in enumerate(pred_on))
#     gt   = sorted((t, j) for j, t in enumerate(gt_on))

#     # 產生候選配對（只收 |err|<=tol_s）
#     pairs = []
#     gpos0 = 0
#     for tp, pi in pred:
#         while gpos0 < len(gt) and gt[gpos0][0] < tp - tol_s:
#             gpos0 += 1
#         k = gpos0
#         while k < len(gt) and gt[k][0] <= tp + tol_s:
#             err = tp - gt[k][0]
#             pairs.append((abs(err), pi, tp ,k, err))
#             k += 1

#     pairs.sort(key=lambda x: x[0])  # 先配最小誤差
#     used_p, used_g, matches = set(), set(), []
#     for _ , pi, tp, gpos, err in pairs:
#         gi = gt[gpos][1] #true start time
#         if pi in used_p or gi in used_g:
#             continue
#         used_p.add(pi); used_g.add(gi)
#         matches.append((pi, tp, gi, err))

#     un_p = set(range(len(pred_on))) - used_p
#     un_g = set(range(len(gt_on)))  - used_g
#     return matches, un_p, un_g

# def _to_onsets(evts):
#     """[(s,e), ...] -> sorted onsets list [s1, s2, ...]"""
#     return sorted(float(s) for s, _ in evts)

# fn_key = Path(audio).name.strip().lower()
# matches, un_p, un_g = match_onset_on_nearest_one(pred_on=[formattime(s) for s, _ in merged], gt_on=_to_onsets(train_label_map[fn_key]))

# tp, fp, fn = len(matches), len(un_p), len(un_g)
# prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
# rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
# f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
# mae  = float(np.mean([abs(e) for *_, e in matches])) if tp>0 else None
# print('tp:',tp)
# print('fp:',fp)
# print('fn:',fn)
# print('prec:',prec)
# print('rec:',rec)
# print('f1:',f1)
# print('MAE:',mae)



# gt_segments_sorted = sorted(train_label_map.get(fn_key, []), key=lambda x: x[0])
# gt_on  = [float(s) for s, _ in gt_segments_sorted]                  
# gt_end = [float(e) for _ , e in gt_segments_sorted]                                 
# pred_on     = [s for s, _ in merged]  
# pred_end = [e for _ ,e in merged]
# matches.sort(key=lambda x: x[0]) #(pi, tp, gi, err)


# with open(out_dir / "vadmatcheck_mode0_nofilter.csv", "w", newline="", encoding="utf-8") as f:
#     w = csv.writer(f)
#     w.writerow(['gt_start','gt_end','predict_start','predict_end'])
#     for pi, tp, gi , _ in matches:
#         if gi < 0 or gi > len(gt_on):
#             continue
#         w.writerow(["'"+formattime(gt_on[gi]),"'"+formattime(gt_end[gi]),"'"+formattime(tp),"'"+formattime(pred_end[pi])])
#     for pi in sorted(un_p):
#         if pi < 0 or pi > len(pred_on):
#             continue
#         w.writerow(['','', "'"+formattime(pred_on[pi]),"'"+ formattime(pred_end[pi])])
#     for gi in sorted(un_g):
#         if gi <0 or gi > len(gt_on):
#             continue
#         w.writerow(["'"+formattime(gt_on[gi]),"'"+formattime(gt_end[gi]),'',''])


