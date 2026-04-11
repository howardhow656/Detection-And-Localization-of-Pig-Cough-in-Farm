from fastapi import APIRouter, UploadFile, File, HTTPException, Form
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
import json
import pandas as pd
import threading





router = APIRouter()

pred_model = None
model_lock = threading.Lock()
def get_pred_model():
    global pred_model
    if pred_model is None:
        with model_lock:
            if pred_model is None:
                pred_model = tf.keras.models.load_model(
                    r"C:\Users\ntuast\Desktop\Detection-And-Localization-of-Pig-Cough-in-Farm-main\cough_cnn_model.h5"
                )
    return pred_model

vad = webrtcvad.Vad()


with open(r'model_config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

n_mels = config["n_mels"]
hop_length = config["hop_length"]
T_fix = config["T_fix"]
sr = config["sample_rate"]
use_audio_norm = config["normalize_audio"]
use_mfcc_norm = config["normalize_mfcc"]
bp_cfg = config["bandpass"]


@router.get("/testing")
def testing():
    return {"message": "Hello World from process app"}

@router.post('/upload_audio')
async def upload_audio (
    file: UploadFile = File(...),
    saved_path: str = Form('default')
 ):
    try:
        file_base_name = os.path.splitext(file.filename)[0]
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

        base_dir = Path(saved_path)
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
                if s - end < 0.3:
                    end = e
                else:
                    merged.append([start,end])
                    start,end = s,e
            merged.append([start,end])

        min_duration = 0.15 #s
        merged = [[s,e] for [s,e] in merged if (e - s) >= min_duration]

        with open(out_dir / "merge.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Start",  "End", 'Label', "Probabiltiy"])
            for s, e in merged:
                w.writerow([f"{s:.6f}", f"{e:.6f}", '', ''])


    except Exception as e:
        raise HTTPException(status_code=400, detail=f"音檔讀取失敗: {str(e)}")

    return {
        "filename": file_base_name,
        "size_in_bytes": len(audio_bytes),
        'Duration:' : round(duration, 3),
        'Number of events:' : len(merged)
    }



@router.post('/cough_detect')
async def cough_detect(
        file:UploadFile = File(...),
        location:UploadFile = File(...),
        saved_path: str = Form('default')
):
    '''
    file: 需要label的音檔(要事先把音檔時間分出來)
    location:聲音事件的檔案(先用vad找出來)
    '''

    try:
        output_name = "cough_detection.csv"
        paths = Path(saved_path) / output_name
        audio_bytes = await file.read()

        audio_io = io.BytesIO(audio_bytes)
        y, sr = sf.read(audio_io, dtype='float32')
        position = pd.read_csv(location.file)
        
        y = package.bandpass_filter(y , sr)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if use_audio_norm:
            y = y / (np.max(np.abs(y)) + 1e-8)
        
        prob_list = []
        label_list = []
        cough_count = 0
        for _, row in position.iterrows():
            s = float(row["Start"])
            e = float(row["End"])
            segment = y[int(s * sr) : int(e * sr)]
        

            mfcc = package.log_mel(segment, sr, n_mels=n_mels, hop_length=hop_length)
            if use_mfcc_norm:
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
            if mfcc.shape[1] < T_fix:
                pad = T_fix - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
            else:
                mfcc = mfcc[:, :T_fix]

            X = mfcc[np.newaxis, ..., np.newaxis].astype(np.float32)

            pred_model = get_pred_model()
            prob = pred_model.predict(X)[0][0]
            label = 1 if prob >= 0.6 else 0
            if label == 1 :
                cough_count += 1
                output = os.path.join(Path(saved_path),f'{cough_count}.wav') 
                sf.write(output,segment,sr)
            prob_list.append(prob)
            label_list.append(label)
        
        position["Probability"] = prob_list
        position["Label"] = label_list
        
        position.to_csv(paths, index=False, encoding="utf-8-sig")


    except Exception as e:
        raise HTTPException(status_code=400, detail=f"處理失敗: {str(e)}")


    return {
        'filename':file.filename,
        'Number of cough:':cough_count,
        'output_file:': output_name
    }



