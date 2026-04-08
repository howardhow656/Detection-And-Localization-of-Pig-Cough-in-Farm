from fastapi import FastAPI, Form, APIRouter, UploadFile, File, HTTPException
import tensorflow as tf
import io
import soundfile as sf
import numpy as np
import webrtcvad
import librosa
import package
from pathlib import Path
import csv
import os




app = FastAPI()

router = APIRouter()


model = tf.keras.models.load_model(r"C:\Users\ntuast\Desktop\Detection-And-Localization-of-Pig-Cough-in-Farm-main\cough_cnn_model.h5")
vad = webrtcvad()

@router.post('/upload_audio')
async def upload_audio(file: UploadFile = File(...)):
    try:
        file_base_name = os.path.splitext(os.path.basename(file.name))[0]
        audio_bytes = await file.read()

        audio_io = io.BytesIO(audio_bytes)
        y , sr = sf.read(audio_io)

        if y.ndim > 1:
            y = np.mean(y, axis=1)
        y_bandpass = package.bandpass_filter(y , sr, low_cut=0, high_cut=2500)
        

        duration = len(y) / sr
        target_rate = 48000
        frame_length = 10
        y_resampled = np.clip(librosa.resample(y_bandpass, orig_sr=sr, target_sr=target_rate), -1, 1 )
        y_int16 = (y_resampled * 32767).astype(np.int16)

        base_dir = Path(r"C:\Users\howardhow\Desktop\sound detection\data")
        out_dir = base_dir / file_base_name
        out_dir.mkdir(parents=True, exist_ok=True)

        speech_list = []
        for frame, start in package.split_to_frame(y_int16, target_rate ,frame_length=frame_length):
            if vad.is_speech(frame, target_rate):
                speech_list.append([float(start/target_rate),float((start/target_rate+frame_length/1000))])

        merged = []
        raw_time = []
        if not speech_list:
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音檔讀取失敗: {str(e)}")

    return {
        "filename": file.filename,
        "size_in_bytes": len(audio_bytes),
        'Duration:' : round(duration, 3),
        'Number of events:' : len(new_merge)
    }
