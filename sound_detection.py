import numpy as np
import librosa
from scipy.ndimage import label
import scipy.signal as signal
import optuna
import soundfile as sf
import csv ,os
from pathlib import Path
import random


def _robust_z(v):
    med = np.median(v)
    mad = np.median(np.abs(v - med)) * 1.4826 + 1e-12
    return (v - med) / mad

def log_RMS(y, sr, frame_length, hop_length):
    starts = np.arange(0, len(y) - frame_length + 1, hop_length)
    frames = np.lib.stride_tricks.as_strided(
        y, shape=(len(starts), frame_length),
        strides=(y.strides[0]*hop_length, y.strides[0])
    )
    rms = np.sqrt(np.mean(frames**2, axis=1)) + 1e-12
    return np.log(rms)

def _parse_time_to_seconds(ts: str) -> float:
    ts = ts.strip()
    if ';' in ts:
        ts = ts.replace(';',':')
    if ":" not in ts:
        return float(ts)  # 已經是秒
    parts = ts.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 2:        # mm:ss(.sss)
        m, s = parts
        return m*60 + s
    elif len(parts) == 3:      # hh:mm:ss(.sss)
        h, m, s = parts
        return h*3600 + m*60 + s
    else:
        raise ValueError(f"Bad timestamp format: {ts}")
    

def load_manifest(manifest_csv):
    files = []
    with open(manifest_csv, newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            files.append(row["file"])
    return files

def load_labels(labels_csv):
    # 回傳 dict[file] = [(s,e), ...]
    d = {}
    with open(labels_csv, newline='') as f:
        for row in csv.DictReader(f):
            fn = row["file"]; s = _parse_time_to_seconds(row['start']); e = _parse_time_to_seconds(row["end"])
            d.setdefault(fn, []).append((s, e))
    print("LABELS keys:", list(d.keys())[:3])
    return d

def build_dataset(manifest_csv, labels_csv):
    files = load_manifest(manifest_csv)
    labels = load_labels(labels_csv)
    dataset = []
    for fn in files:
        print("FN from manifest:", fn)
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        y, sr = sf.read(fn, dtype="float32", always_2d=False)
        # y, sr = ensure_mono_16k(y, sr, 16000)
        gt = labels.get(fn, [])
        dataset.append((fn, y, sr, gt))
    return dataset

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

def spectral_entropy(y, frame_length, hop_length):
    S = np.abs(librosa.stft(y, n_fft=frame_length,
                            hop_length=hop_length,
                            win_length=frame_length,
                            center=False))**2
    ps = S / (np.sum(S, axis=0, keepdims=True) + 1e-12)
    H = -np.sum(ps * np.log(ps + 1e-12), axis=0)
    H /= np.log(ps.shape[0] + 1e-12)
    return H  

def IOU_id(a,b):
    sa, ea = a; sb,eb =b
    inter = max(0,min(ea,eb)-max(sa,sb))
    union = max(0,max(ea,eb)-min(sa,sb))
    return inter/union if union >0 else 0

def greedy_match_by_iou(pred, gt, iou_th=0.5):
    if not pred or not gt:
        return [], set(range(len(pred))), set(range(len(gt)))
    pairs = []
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            pairs.append((i, j, IOU_id(p, g)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    used_p, used_g, matches = set(), set(), []
    for i, j, v in pairs:
        if v < iou_th: break
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); matches.append((i, j, v))
    return matches, set(range(len(pred))) - used_p, set(range(len(gt))) - used_g

def score_event_f1_and_boundary(pred, gt, iou_th=0.5, collar_ms=50.0):

    matches, un_p, un_g = greedy_match_by_iou(pred, gt, iou_th=iou_th)
    tp, fp, fn = len(matches), len(un_p), len(un_g)
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

    # 邊界誤差（取 matched 的平均 onset/offset MAE；各給予 ±collar 容差）
    collar_s = (collar_ms or 0.0) / 1000.0
    if tp == 0:
        be = 0.0  # 沒配到就不罰邊界（已在 F1 中懲罰），也可選擇給一個固定懲罰
    else:
        on_err, off_err = [], []
        for i, j, _ in matches:
            ps, pe = pred[i]; gs, ge = gt[j]
            on_err.append(max(0.0, abs(ps-gs)-collar_s))
            off_err.append(max(0.0, abs(pe-ge)-collar_s))
        be = (np.mean(on_err) + np.mean(off_err))/2.0
    return f1, be, rec 

def spectral_flux(y ,frame_length, hop_length):
    S = np.abs(librosa.stft(y , n_fft=frame_length, hop_length=hop_length, win_length=frame_length, center=False))
    S = S / (np.sum(S, axis=0, keepdims=True) + 1e-12)
    D = np.diff(S, axis=1)
    D = np.maximum(D, 0.0)                     # 只計正向變化
    flux = np.sqrt(np.sum(D * D, axis=0))      # L2
    flux = np.concatenate([[0.0], flux])       # 與其他特徵對齊
    return flux


def zero_crossing_rate_frames(y, frame_length, hop_length):
    """
    以與 log_RMS 相同的對齊（center=False）計算逐幀 ZCR。
    回傳長度 N。
    """
    # librosa 的 ZCR 可直接對齊 frame/hop，指定 center=False 與 RMS 對齊
    z = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length, center=False
    )[0]
    return z


def detect_sound(
    y,
    sr,
    mode,                      # 'and' 或 'score'
    min_event_length = 200,    # ms
    min_events_gap   = 300,    # ms
    frame_ms         = 100,
    hop_ms           = 20,
    Q_persentage     = 10,     # %
    min_std          = 1e-3,
    alpha            = 2.1, #low
    beta             = 2.2,   #high 
    score_weight     = 0.5,
    window_ms        = 5000,   # 自適應窗長（ms）
    update_ms        = 500,    # 門檻更新間隔（ms）
    hangover_ms      = 300,
):
    assert mode in ("and", "score")
    if beta <= alpha:
        beta = alpha + 1.0

    #filter the discarded frequency
    y = bandpass_filter(y,sr)

    frame_length = int(sr * frame_ms/1000)
    hop_length   = int(sr * hop_ms/1000)

    if len(y) < frame_length:
        return []
    
    # --- 特徵 ---
    z_rms = _robust_z(log_RMS(y, sr, frame_length, hop_length))
    H     = spectral_entropy(y, frame_length, hop_length)
    se    = 1.0 - H                   # 先反向成「有聲度」
    z_se  = _robust_z(se)             # 再做 robust z

    if mode == "score":
        feat = score_weight * z_se + (1.0 - score_weight) * z_rms
    else:
        feat = None  # and mode don't need feat

    N = len(z_rms)
    win_frames     = max(1, int(round(window_ms / hop_ms)))
    upd_stride     = max(1, int(round(update_ms  / hop_ms)))
    hangover_frames= max(1, int(round(hangover_ms / hop_ms)))

    # --- statement ---
    state = 0
    hold  = 0
    detection_result = np.zeros(N, dtype=np.uint8)

    # 初值：讓 i=0 時會強制更新門檻
    last_up = -upd_stride

    # 門檻暫存
    T_low = T_high = None
    T_low_r = T_high_r = None
    T_low_se = T_high_se = None

    for i in range(N):
        # if the threshold needs to update
        if (i - last_up) >= upd_stride or (T_low is None and T_low_r is None):
            s = max(0, i - win_frames + 1)

            if mode == "score":
                win = feat[s:i+1]
                q = Q_persentage / 100.0
                k = max(1, int(len(win) * q))
                noise_like = np.partition(win, k-1)[:k]
                mu0   = float(np.mean(noise_like))
                sigma = float(np.std(noise_like))
                if sigma < min_std: sigma = min_std
                T_low  = mu0 + alpha * sigma
                T_high = mu0 + beta  * sigma

            else:  # and
                # rms thershold
                win_r = z_rms[s:i+1]
                q = Q_persentage / 100.0
                k_r = max(1, int(len(win_r) * q))
                noise_r = np.partition(win_r, k_r-1)[:k_r]
                mu_r, sg_r = float(np.mean(noise_r)), float(np.std(noise_r))
                if sg_r < min_std: sg_r = min_std
                T_low_r  = mu_r + alpha * sg_r
                T_high_r = mu_r + beta  * sg_r
                # SE threshold
                win_s = z_se[s:i+1]
                k_s = max(1, int(len(win_s) * q))
                noise_s = np.partition(win_s, k_s-1)[:k_s]
                mu_s, sg_s = float(np.mean(noise_s)), float(np.std(noise_s))
                if sg_s < min_std: sg_s = min_std
                T_low_se  = mu_s + alpha * sg_s
                T_high_se = mu_s + beta  * sg_s

            last_up = i

        # rms + entropy
        if state == 0:
            if mode == "score":
                if feat[i] >= T_high:
                    state, hold = 1, 0
            else:
                if (z_rms[i] >= T_high_r) and (z_se[i] >= T_high_se):
                    state, hold = 1, 0
        else:
            if mode == "score":
                if feat[i] < T_low:
                    hold += 1
                    if hold >= hangover_frames:
                        state, hold = 0, 0
                else:
                    hold = 0
            else:
                if (z_rms[i] < T_low_r) or (z_se[i] < T_low_se):
                    hold += 1
                    if hold >= hangover_frames:
                        state, hold = 0, 0
                else:
                    hold = 0

        detection_result[i] = state

    # --- turn into times ---
    def idx_to_time(idx):
        start = idx * hop_length / sr
        end   = min(start + frame_length / sr, len(y) / sr)
        return start, end
    segments = []
    i = 0
    while i < N:
        if detection_result[i] == 1:
            j = i + 1
            while j < N and detection_result[j] == 1:
                j += 1
            s0, _ = idx_to_time(i)
            _, e0 = idx_to_time(j-1)   # 用 j-1 的幀
            segments.append([s0, e0])
            i = j
        else:
            i += 1

    # merger
    merged = []
    if segments:
        cur_s, cur_e = segments[0]
        for s, e in segments[1:]:
            if s - cur_e <= (min_events_gap / 1000.0):
                cur_e = e
            else:
                merged.append([cur_s, cur_e])
                cur_s, cur_e = s, e
        merged.append([cur_s, cur_e])

    # deletion
    final = []
    min_len_s = min_event_length / 1000.0
    for s, e in merged:
        if (e - s) >= min_len_s:
            final.append([round(s, 3), round(e, 3)])

    return final

def split_dataset(dataset):
    idxs = list(range(len(dataset)))
    random.Random(42).shuffle(idxs)
    cut = int(len(dataset) * (1 - 0.2))
    tr = [dataset[i] for i in idxs[:cut]]
    va = [dataset[i] for i in idxs[cut:]]
    return tr, va

def evaluate_dataset(dataset, params, iou_th=0.5, collar_ms=50.0):
    f1_list, be_list, rec_list = [], [], [] 
    for fn ,y, sr, gt in dataset:
        pred = detect_sound(
            y, sr,
            mode=params.get("mode", "score"),
            min_event_length=params.get("min_event_length", 200),
            min_events_gap=params.get("min_events_gap", 300),
            frame_ms=params.get("frame_ms", 100),
            hop_ms=params.get("hop_ms", 20),
            Q_persentage=params.get("Q_persentage", 10),
            min_std=params.get("min_std", 1e-3),
            alpha=params.get("alpha", 2.1),
            beta=params.get("beta", 2.2),
            score_weight=params.get("score_weight", 0.5),
            window_ms=params.get("window_ms", 5000),
            update_ms=params.get("update_ms", 500),
            hangover_ms=params.get("hangover_ms", 300)
        )
        f1, be, recall = score_event_f1_and_boundary(pred, gt, iou_th=iou_th, collar_ms=collar_ms)
        f1_list.append(f1)
        be_list.append(be)
        rec_list.append(recall)

    F1 = float(np.mean(f1_list)) if f1_list else 0.0
    BE = float(np.mean(be_list)) if be_list else 0.0

    RECALL = float(np.mean(rec_list)) if rec_list else 0.0
    return  F1, BE, RECALL                

def make_objective(dataset, iou_th=0.5, collar_ms=50.0, lambda_be=0.2, fixed_mode=None):
    """
    Score = F1 - lambda_be * BE  (我們要最大化，Optuna 用 minimize，所以回傳 -Score)
    fixed_mode: 若想固定 'and' 或 'score'，就丟字串；不指定就讓 Optuna 自己選。
    """
    def objective(trial: optuna.Trial):
        # 搜尋空間（你可按需求增減）
        params = {
            "mode": trial.suggest_categorical("mode", ["score", "and"]) if fixed_mode is None else fixed_mode,

            # 顆粒度
            "frame_ms": trial.suggest_int("frame_ms", 50, 130, step=10),
            "hop_ms":   trial.suggest_int("hop_ms",   20,  80, step=5),

            # 自適應窗
            "window_ms": trial.suggest_int("window_ms", 2000, 8000, step=200),
            "update_ms": trial.suggest_int("update_ms", 200, 1000, step=50),

            # 分位數法與雙門檻
            "Q_persentage": trial.suggest_int("Q_persentage", 5, 15, step=5),
            "alpha": trial.suggest_float("alpha", 2, 16.0),
            "beta":  trial.suggest_float("beta",  2, 24.0),
            "score_weight": trial.suggest_float("score_weight", 0.2, 0.8),

            # 輸出穩定
            "hangover_ms": trial.suggest_int("hangover_ms", 100, 800, step=50),

            # 事件後處理
            "min_event_length": 50,#trial.suggest_int("min_event_length", 200, 1200, step=100),
            "min_events_gap":   100,#trial.suggest_int("min_events_gap",   200, 1200, step=100),

            # 固定其他
            "min_std": 1e-3,
        }

        F1, BE,recall = evaluate_dataset(dataset, params, iou_th=iou_th, collar_ms=collar_ms)
        score = F1 - lambda_be * BE   # 想最大化
        # 你也可以回傳多指標到 trial.user_attrs
        trial.set_user_attr("F1", F1)
        trial.set_user_attr("BE", BE)

        return -score
    return objective



def _overlap(a, b):
    sa, ea = a; sb, eb = b
    return max(0.0, min(ea, eb) - max(sa, sb))




def dump_predictions_and_gt(dataset, params, out_dir,
                            iou_th=0.3, collar_ms=150):
    """
    逐檔偵測，輸出：
      1) summary.csv: file, gt_cnt, pred_cnt, precision, F1
      2) segments_all.csv: type(TP/FP/FN), file, pred_s, pred_e, gt_s, gt_e
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []          # 每檔摘要
    segments_all_rows = []     # 全資料集逐段清單

    for fn, y, sr, gt in dataset:
        # ---- 1) 偵測 ----
        pred = detect_sound(
            y, sr,
            mode=params.get("mode", "score"),
            min_event_length=params.get("min_event_length", 200),
            min_events_gap=params.get("min_events_gap", 300),
            frame_ms=params.get("frame_ms", 100),
            hop_ms=params.get("hop_ms", 20),
            Q_persentage=params.get("Q_persentage", 10),
            min_std=params.get("min_std", 1e-3),
            alpha=params.get("alpha", 2.1),
            beta=params.get("beta", 2.2),
            score_weight=params.get("score_weight", 0.5),
            window_ms=params.get("window_ms", 5000),
            update_ms=params.get("update_ms", 500),
            hangover_ms=params.get("hangover_ms", 300)
        )

        # ---- 2) 事件級 F1（只取 F1；忽略 BE / recall）----
        F1, _, _ = score_event_f1_and_boundary(pred, gt, iou_th=iou_th, collar_ms=collar_ms)

        # ---- 3) IoU 配對拿 TP/FP/FN（為了 precision 與 segments_all）----
        matches, un_p, un_g = greedy_match_by_iou(pred, gt, iou_th=iou_th)
        tp_idx = [i for i, _, _ in matches]
        fp_idx = sorted(list(un_p))
        fn_idx = sorted(list(un_g))

        TP = [pred[i] for i in tp_idx]
        FP = [pred[i] for i in fp_idx]
        FN = [gt[j]   for j in fn_idx]

        precision = (len(TP) / (len(TP) + len(FP))) if (len(TP)+len(FP))>0 else 0.0

        # ---- 4) 推進 segments_all（TP/FP/FN 全部攤平寫一次）----
        # TP：同時有 pred 與 gt
        for i, j, _ in matches:
            ps, pe = pred[i]; gs, ge = gt[j]
            segments_all_rows.append(["TP", fn, ps, pe, gs, ge])
        # FP：只有 pred
        for i in fp_idx:
            ps, pe = pred[i]
            segments_all_rows.append(["FP", fn, ps, pe, "", ""])
        # FN：只有 gt
        for j in fn_idx:
            gs, ge = gt[j]
            segments_all_rows.append(["FN", fn, "", "", gs, ge])

        # ---- 5) 單檔摘要（只留你要的欄位）----
        summary_rows.append([fn, len(gt), len(pred), precision, F1])

    # ---- 6) 存 summary.csv ----
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "gt_cnt", "pred_cnt", "precision", "F1"])
        for row in summary_rows:
            w.writerow(row)
        # 追加 <ALL>（micro）：用總 TP/FP/FN 算 precision 與 F1
        if summary_rows:
            # 先重算一次總 TP/FP/FN（需要用 segments_all_rows）
            TPm = sum(1 for t, *_ in segments_all_rows if t == "TP")
            FPm = sum(1 for t, *_ in segments_all_rows if t == "FP")
            FNm = sum(1 for t, *_ in segments_all_rows if t == "FN")
            prec_all = TPm/(TPm+FPm) if (TPm+FPm)>0 else 0.0
            rec_all  = TPm/(TPm+FNm) if (TPm+FNm)>0 else 0.0
            F1_all   = 2*prec_all*rec_all/(prec_all+rec_all) if (prec_all+rec_all)>0 else 0.0

            gt_total   = sum(r[1] for r in summary_rows)
            pred_total = sum(r[2] for r in summary_rows)
            w.writerow(["<ALL>", gt_total, pred_total, prec_all, F1_all])

    # ---- 7) 存 segments_all.csv ----
    with open(out_dir / "segments_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type", "file", "pred_start", "pred_end", "gt_start", "gt_end"])
        for row in segments_all_rows:
            w.writerow(row)
