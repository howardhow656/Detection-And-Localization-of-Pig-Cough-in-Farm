import os
import soundfile as sf

import sound_detection as sd
import numpy as np
import signal
import audio_to_image as AtI

from glob import glob

from pathlib import Path



def split_to_frame(audio_file , sample_rate , frame_length):
    n = int(sample_rate * frame_length/1000)
    start = 0
    while start + n <= len(audio_file):
        frame = audio_file[start:start+n]
        yield frame.tobytes(), start
        start += n

def modified_label(new_merge ,y ,sr ,out_dir ,file_base_name):
    n_samples = len(y)
    for s , e , index in new_merge:
        start_idx = int(round(s * sr))
        end_idx   = int(round(e * sr))
        # 邊界保護
        start_idx = max(0, start_idx)
        end_idx   = min(n_samples, end_idx)
        if end_idx <= start_idx:
            continue
        audio_segment = y[start_idx : end_idx]
        output = os.path.join(out_dir,f'{file_base_name}_{index}.wav') 
        sf.write(output,audio_segment,sr)



def modified_time(merged ,y_resampled, target_rate):
    '''
    index:audio segments index
    orig_s、orig_e:original time
    new_s、new_e:modified time
    mode:which modified mode
    '''
    new = []
    mapping = []
    index = 1
    for s, e in merged:

        if e - s < 1.5:#往前-0.3後，e往後補1.5s
            e_new = s + 1.2
            s_new = max(0, s - 0.3)
            new.append([s_new,e_new,index])
            mapping.append({
                "orig_index": index,
                "orig_s": s,
                "orig_e": e,
                "new_s": s_new,
                "new_e": e_new,
                "mode": "pad_short"
            })
            index += 1
        elif e - s == 1.5:#完全不動
            new.append([s,e,index])
            mapping.append({
                "orig_index": index,
                "orig_s": s,
                "orig_e": e,
                "new_s": s,
                "new_e": e,
                "mode": "same"
            })
            index += 1
        while e - s > 1.5:#
            new_e = s + 1.5
            new.append([s,new_e,index])
            mapping.append({
                "orig_index": index,
                "orig_s": s,
                "orig_e": e,
                "new_s": s,
                "new_e": new_e,
                "mode": "split"
            })
            s = new_e
            index += 1
            if 0.4 < e - s <= 1.5:
                e2 =  s + 1.5
                if e2 <= len(y_resampled)/target_rate:
                    new.append([s,e2,index])
                    mapping.append({
                        "orig_index": index,
                        "orig_s": s,
                        "orig_e": e,
                        "new_s": s,
                        "new_e": e2,
                        "mode": "tail_pad"
                    })
                    index += 1
                break
    return new, mapping


def bandpass_filter(
    y,
    sr,
    low_cut=200.0,
    high_cut=5000.0,
    order=8,
    zero_phase=True,  
):
    """
    穩健的帶通濾波：
    - 使用 SOS（butter(..., output='sos')）
    - 若訊號太短無法 sosfiltfilt，就改用 sosfilt（不報錯）
    - 自動夾住截止頻率在 (0, Nyquist) 內
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = np.mean(y, axis=1)  

    nyq = 0.5 * float(sr)

    low = max(1.0, float(low_cut))
    high = min(float(high_cut), 0.99 * nyq)

    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return y

    wn = [low / nyq, high / nyq]

    # 設計濾波器
    sos = signal.butter(order, wn, btype="band", output="sos")


    if zero_phase:

        needed_pad = 3 * (2 * sos.shape[0])
        if y.size > needed_pad:
            return signal.sosfiltfilt(sos, y)
        else:
            return signal.sosfilt(sos, y)
    else:
        return signal.sosfilt(sos, y)


