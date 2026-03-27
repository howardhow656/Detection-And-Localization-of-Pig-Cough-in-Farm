from pathlib import Path
import csv
import sound_detection as sd
import soundfile as sf
import numpy as np
import audio_to_image as AtI
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, BatchNormalization,
ReLU, GlobalAveragePooling2D, Dense, Dropout)
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
import os
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import gc
import librosa
import json


train_folder = Path(r"C:\Users\howardhow\Desktop\sound detection\data\training")
try_data_path = Path(r"C:\Users\howardhow\Desktop\sound detection\data\training\rec0814-020007\try.wav")
vali_data_path = Path(r"C:\Users\howardhow\Desktop\sound detection\data\validation\rec0810-001351\rec0810-001351.wav")
test_data_path = Path(r"C:\Users\howardhow\Desktop\sound detection\data\testing\rec0811-232408\rec0811-232408.wav")
vali_label_path = Path(r"C:\Users\howardhow\Desktop\sound detection\data\validation\rec0810-001351\001351_label(csv).csv")
test_label_path = Path(r"C:\Users\howardhow\Desktop\sound detection\data\testing\rec0811-232408\232408_label(csv).csv")
train_label_map = defaultdict(list)
test_label_map = defaultdict(list)
vali_label_map = defaultdict(list)
try_label_map = defaultdict(list)
test_dataset = []
train_audio_files = []
n_mels = 40


for wav_path in train_folder.glob('*.wav'):
    base = wav_path.stem                      
    csv_path = wav_path.with_suffix(".csv")  
    if not csv_path.exists():
        print(f"[警告] 找不到 {base}.csv，跳過這筆")
        continue

    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                s = float(row['Start'])
                e = float(row['End'])
                t = int(row['Label']) - 1    # 你現在使用 0/1
            except:
                s = sd._parse_time_to_seconds(row['Start'])
                e = sd._parse_time_to_seconds(row['End'])
                t = int(row['Label']) - 1

            train_label_map[base].append([s, e, t])
    train_audio_files.append(wav_path)




test_base_name = os.path.splitext(os.path.basename(test_data_path))[0]
vali_base_name = os.path.splitext(os.path.basename(vali_data_path))[0]


with open(test_label_path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter=','):
        fn =  test_base_name
        try:
            s = float(row['Start']); e=float(row['End']); t=int(row['Label'])
        except ValueError:
            s = sd._parse_time_to_seconds(row['Start'])
            e = sd._parse_time_to_seconds(row['End'])
            t = int(row['Label']) - 1
        test_label_map[fn].append([s,e,t])     

with open(vali_label_path, newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter=','):
        fn = vali_base_name
        try:
            s = float(row['Start']); e=float(row['End']); t=int(row['Label'])
        except ValueError:
            s = sd._parse_time_to_seconds(row['Start'])
            e = sd._parse_time_to_seconds(row['End'])
            t = int(row['Label']) - 1 #一開始標記錯了XD
        vali_label_map[fn].append([s,e,t])   




def data_generator(audio_file_path, label_map, batch_size=8, hop_lengths=512):
    """
    Data generator for training and validation data.
    
    :param file_paths: List of paths to directories containing spectrograms.
    :param labels: Corresponding labels for each directory (1 for cough, 0 for non-cough).
    :param batch_size: Number of samples per batch.
    :yield: A tuple (batch_data, batch_labels).
    :max_clips:represent how long does one batch(second)
    1. 使用labels來找每一個的spectragram在哪以及要怎麼計算
    2. 每一組label就算一個batch
    """
    audios = []
    file_index = 0
    for path in audio_file_path:
        y, sr = sf.read(path, dtype='float32')
        y = AtI.bandpass_filter(y , sr)
        if y.ndim > 1: y = np.mean(y, axis=1)
        base = path.stem
        audios.append((base, y, sr))

    while True:
        base, audio, sr = audios[file_index]
        labels = label_map[base]

        file_index = (file_index + 1) % len(audios)
        for i in range(0, len(labels), batch_size):
            batch_labels = labels[i:i+batch_size]
            X_list, y_list = [], []
            for s, e, t in batch_labels:
                seg = audio[int(s*sr):int(e*sr)]
                seg = seg / (np.max(np.abs(seg)) + 1e-8)
                mfcc = AtI.log_mel(y=seg, sr=sr, n_mels=n_mels, hop_length=hop_lengths)
                mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
                mfcc = librosa.util.fix_length(mfcc, size=T_fix, axis=1)
                mfcc = mfcc[..., np.newaxis]

                X_list.append(mfcc)
                y_list.append(int(t))
            X = np.stack(X_list, axis=0).astype(np.float32)
            labels_out = np.array(y_list)
            yield X, labels_out
    

vali_labels  = vali_label_map[vali_base_name]
test_labels  = test_label_map[test_base_name]

fixed_sr = 16000
fixed_event_sec = 1.5 #s
hop_length_ms = 10
hop_length = int(fixed_sr * hop_length_ms/1000)


batch_size = 32
train_generator = data_generator(audio_file_path=train_audio_files, label_map=train_label_map, batch_size=batch_size, hop_lengths=hop_length)
test_generator = data_generator([test_data_path], label_map=test_label_map, batch_size=batch_size, hop_lengths=hop_length)
validation_generator = data_generator([vali_data_path], label_map=vali_label_map, batch_size=batch_size, hop_lengths=hop_length)




# 1. 隨便選第一個訓練檔來量 T

first_audio_path = train_audio_files[0]
base0           = first_audio_path.stem           # 檔名(不含副檔名)
y_tmp, sr_tmp   = sf.read(first_audio_path, dtype='float32')
if y_tmp.ndim > 1:
    y_tmp = np.mean(y_tmp, axis=1)

# 2. 拿這個檔案的第一筆標註來切一段
s0, e0, _ = train_label_map[base0][0]
seg_tmp   = y_tmp[int(s0 * sr_tmp) : int(e0 * sr_tmp)]

# ★ amplitude normalization：跟 generator 一樣
seg_tmp = seg_tmp / (np.max(np.abs(seg_tmp)) + 1e-8)

# 3. 用跟 training 完全一樣的方式算 MFCC
mfcc_tmp = AtI.log_mel(y=seg_tmp, sr=sr_tmp, n_mels=n_mels, hop_length=hop_length)

# ★ MFCC normalization：也跟 generator 一樣
mfcc_tmp = (mfcc_tmp - np.mean(mfcc_tmp)) / (np.std(mfcc_tmp) + 1e-8)

# 4. 量出時間維度 T_fix
T_fix = mfcc_tmp.shape[1]

del y_tmp, sr_tmp, seg_tmp, mfcc_tmp
gc.collect()

def build_2d_cnn(n_mels=64, T=T_fix, num_classes=2):
    inp = Input(shape=(n_mels, T, 1))


    def dw_block(x, c, s=(1,1)):
        x = DepthwiseConv2D((3,3), strides=s, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x); x = ReLU()(x)
        x = Conv2D(c, (1,1), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x); x = ReLU()(x)
        return x


    x = Conv2D(32, (3,3), padding='same', use_bias=False)(inp)
    x = BatchNormalization()(x); x = ReLU()(x)


    x = dw_block(x, 48, s=(2,2))
    x = dw_block(x, 64, s=(2,2))
    x = dw_block(x, 96, s=(2,2))


    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    

    if num_classes == 2:
        out = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    else:
        out = Dense(num_classes, activation='softmax')(x)
        loss = 'sparse_categorical_crossentropy'
        metrics = ['accuracy']


    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=metrics)
    return model

num_classes = 2

model = build_2d_cnn(n_mels=n_mels, T=T_fix, num_classes=num_classes if num_classes > 2 else 2)
model.summary()

def count_steps(n):
    return math.ceil(n / batch_size)

epochs = 40
patience = 6

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=patience,
    min_lr=1e-6      
)

es = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)



def count_total_events(label_map):
    total = 0
    for key in label_map:
        total += len(label_map[key])
    return total


history = model.fit(
train_generator,
steps_per_epoch=count_steps(count_total_events(train_label_map)),
validation_data=validation_generator,
validation_steps=count_steps(count_total_events(vali_label_map)),
epochs=epochs,
callbacks=[lr_scheduler , es],
class_weight={0: 2.0, 1: 1.0}
)   




plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='train')
plt.plot(history.history.get('val_loss', []), label='val')
plt.xlabel("Epoch")        # X 軸名稱
plt.ylabel("Loss")         # Y 軸名稱
plt.title('Loss')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()


plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='train')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='val')
plt.title('Accuracy')
plt.xlabel("Epoch")        # X 軸名稱
plt.ylabel("Accuracy")         # Y 軸名稱
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()




results = model.evaluate(test_generator, steps=count_steps(len(test_labels)))
model.save(r'C:\Users\howardhow\Desktop\sound detection\data\cough_cnn_model.h5')
print("Test Results:", dict(zip(model.metrics_names, results)))



config = {
    "n_mels": n_mels,
    "hop_length": hop_length,
    "T_fix": T_fix,
    "sample_rate": fixed_sr,
    "normalize_audio": True,        # 你 amplitude normalize
    "normalize_mfcc": True,         # 你 mfcc normalize
    "bandpass": {
        "low_cut": 200,
        "high_cut": 5000,
        "order": 8
    }
}


with open(r"C:\Users\howardhow\Desktop\sound detection\data\model_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)




