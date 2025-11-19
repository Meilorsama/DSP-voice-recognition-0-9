import os
import numpy as np
import librosa
import librosa.display
import matplotlib

matplotlib.use('TkAgg')  # 解决后端兼容问题
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
from scipy.io import wavfile
import matplotlib
# 切换为稳定的后端（TkAgg兼容性好，无需额外依赖）
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 设置支持中文的字体（根据系统选择，Windows推荐SimHei，Mac推荐Heiti TC）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# -------------------------- 核心配置参数 --------------------------
# 语音参数（帧长设为2的整数次幂，适配FFT蝶形算法）
SAMPLE_RATE = 22050  # 采样率
DURATION = 1.5  # 单条语音时长（秒）
N_FFT = 512  # FFT点数（2^9，符合2的整数次幂要求）
HOP_LENGTH = int(SAMPLE_RATE * 0.01)  # 帧移（10ms）
FRAME_LENGTH = N_FFT  # 帧长=FFT点数，确保FFT计算效率
PRE_EMPHASIS = 0.97  # 预加重系数
HAMMING_WINDOW = np.hamming(FRAME_LENGTH)  # 汉明窗

# 路径配置
DATA_DIR = "voice_data"  # 语音数据目录（同实验1）
MODEL_DIR = "models"  # 模型保存目录
RESULT_DIR = "experiment2_results"  # 实验结果保存目录
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# -------------------------- 1. 预加重滤波实现 --------------------------
def pre_emphasis_filter(audio):
    """语音预加重：增强高频分量，补偿高频衰减"""
    # 一阶FIR预加重滤波器：y(n) = x(n) - α*x(n-1)
    return np.append(audio[0], audio[1:] - PRE_EMPHASIS * audio[:-1])


# -------------------------- 2. 频谱与Mel频谱计算 --------------------------
def extract_freq_features(audio, sr=SAMPLE_RATE):
    """提取频域特征：FFT频谱 → Mel频谱 → MFCC（13维）"""
    # 1. 分帧加窗（确保帧长为2的整数次幂）
    frames = librosa.util.frame(
        audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    ).T.copy()  # 可写副本
    frames *= HAMMING_WINDOW  # 加窗减少频谱泄漏

    # 2. FFT计算频谱（幅度谱，仅保留前半部分：N_FFT//2 + 1 个点）
    fft_spectrum = np.fft.fft(frames, n=N_FFT, axis=1)  # 按帧FFT
    # 关键修改：只取前半部分（对称部分）
    magnitude_spectrum = np.abs(fft_spectrum[:, :N_FFT // 2 + 1])  # 形状：(帧数, 257)

    # 3. Mel频谱转换（基于人类听觉特性）
    mel_filterbank = librosa.filters.mel(
        sr=sr, n_fft=N_FFT, n_mels=40  # 40个Mel滤波器
    )
    # 现在维度对齐：(帧数, 257) × (257, 40) → (帧数, 40)
    mel_spectrum = np.dot(magnitude_spectrum, mel_filterbank.T)

    # 后续步骤保持不变...

    # 4. 对数变换（增强区分度）
    log_mel_spectrum = np.log1p(mel_spectrum)  # log(1+x)避免数值溢出

    # 5. MFCC提取（Mel频率倒谱系数，取前13维）
    mfcc = librosa.feature.mfcc(
        S=log_mel_spectrum, n_mfcc=30
    ).T  # 形状：(帧数, 13)

    return {
        "mfcc": mfcc,  # 核心特征（用于DTW匹配）
        "fft_spectrum": magnitude_spectrum,  # FFT幅度谱（用于分析）
        "mel_spectrum": mel_spectrum  # Mel频谱（用于分析）
    }


# -------------------------- 3. DTW技术实现（不等长特征匹配） --------------------------
def dtw_distance(feat1, feat2):
    """计算两个MFCC特征序列的DTW距离（动态时间规整）"""
    # 基于 librosa 实现，subseq=True适配子序列匹配
    return librosa.sequence.dtw(feat1.T, feat2.T, subseq=True, metric='euclidean')[1][-1, -1]


# -------------------------- 4. 数据加载模块 --------------------------
def load_freq_dataset():
    """加载数据并提取频域特征（MFCC）"""
    X_mfcc = []  # MFCC特征列表（不等长）
    X_fft = []  # FFT频谱（用于分析）
    y = []  # 标签（数字0-9）
    file_paths = []  # 文件路径（用于追溯）

    for label in os.listdir(DATA_DIR):
        label_path = os.path.join(DATA_DIR, str(label))
        if not os.path.isdir(label_path):
            continue
        for file in os.listdir(label_path):
            if not file.endswith(".wav"):
                continue
            # 加载语音
            sr, audio = wavfile.read(os.path.join(label_path, file))
            audio = audio.astype(np.float32) / 32767.0  # 转为float32（-1.0~1.0）
            # 预加重处理
            audio_pre = pre_emphasis_filter(audio)
            # 提取频域特征
            features = extract_freq_features(audio_pre, sr=sr)
            # 存储
            X_mfcc.append(features["mfcc"])
            X_fft.append(features["fft_spectrum"])
            y.append(str(label))
            file_paths.append(os.path.join(label_path, file))

    return X_mfcc, X_fft, y, file_paths


# -------------------------- 5. 实验核心逻辑（训练+识别+对比） --------------------------
def experiment2():
    print("\n===== 实验2：基于频域分析技术的语音识别 =====")
    # 步骤1：加载数据
    X_mfcc, X_fft, y, file_paths = load_freq_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X_mfcc, y, test_size=0.3, random_state=45, stratify=y
    )
    print(f"数据加载完成：训练集{len(X_train)}条，测试集{len(X_test)}条")

    # 步骤2：DTW匹配识别（不等长特征匹配）
    print("\n开始DTW匹配识别...")
    y_pred = []
    dist_matrix = []  # 距离矩阵（用于分析）
    for test_feat in X_test:
        min_dist = float("inf")
        pred_label = None
        test_dists = []
        for train_feat, train_label in zip(X_train, y_train):
            # 计算DTW距离
            dist = dtw_distance(test_feat, train_feat)
            test_dists.append(dist)
            if dist < min_dist:
                min_dist = dist
                pred_label = train_label
        y_pred.append(pred_label)
        dist_matrix.append(test_dists)
    dist_matrix = np.array(dist_matrix)

    # 步骤3：量化评估（正确率、误纳率）
    accuracy = accuracy_score(y_test, y_pred)
    # 计算误纳率（错误识别为目标类别的非目标样本数/总非目标样本数）
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
    total_samples = len(y_test)
    correct_samples = np.trace(cm)
    incorrect_samples = total_samples - correct_samples
    # 误纳率=错误识别样本数/(总样本数-正确识别数)（简化计算，聚焦类别间误判）
    false_accept_rate = incorrect_samples / (total_samples - correct_samples) if (
                                                                                             total_samples - correct_samples) > 0 else 0.0

    print(f"\n=== 实验2量化结果 ===")
    print(f"正确率：{accuracy:.4f}")
    print(f"误纳率：{false_accept_rate:.4f}")

    # 步骤4：结果可视化（图+表）
    # 4.1 混淆矩阵
    plt.figure(figsize=(10, 8))
    labels = np.unique(y)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        cmap=plt.cm.Blues, xticks_rotation=0
    )
    plt.title("实验2 频域特征+DTW混淆矩阵", fontsize=14)
    plt.ylabel("真实标签", fontsize=12)
    plt.xlabel("预测标签", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=300)
    plt.show()

    # 4.2 FFT与Mel频谱对比（以第一条测试样本为例）
    test_audio_idx = 0
    test_audio_path = file_paths[test_audio_idx]
    sr, test_audio = wavfile.read(test_audio_path)
    test_audio = test_audio.astype(np.float32) / 32767.0
    test_audio_pre = pre_emphasis_filter(test_audio)
    features = extract_freq_features(test_audio_pre, sr=sr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    # FFT频谱图
    fft_freq = np.fft.fftfreq(N_FFT, 1 / sr)[:N_FFT // 2]
    ax1.plot(fft_freq, np.mean(features["fft_spectrum"], axis=0)[:N_FFT // 2])
    ax1.set_title("FFT幅度频谱", fontsize=12)
    ax1.set_xlabel("频率(Hz)", fontsize=10)
    ax1.set_ylabel("幅度", fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Mel频谱图
    mel_freq = librosa.mel_frequencies(n_mels=40, fmin=0, fmax=sr // 2)
    ax2.plot(mel_freq, np.mean(features["mel_spectrum"], axis=0))
    ax2.set_title("Mel频谱", fontsize=12)
    ax2.set_xlabel("Mel频率", fontsize=10)
    ax2.set_ylabel("幅度", fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "fft_mel_comparison.png"), dpi=300)
    plt.show()

    # 步骤5：保存实验结果
    result_dict = {
        "accuracy": accuracy,
        "false_accept_rate": false_accept_rate,
        "confusion_matrix": cm,
        "labels": labels,
        "dist_matrix": dist_matrix
    }
    with open(os.path.join(RESULT_DIR, "experiment2_results.pkl"), "wb") as f:
        pickle.dump(result_dict, f)
    print(f"\n实验结果已保存至：{RESULT_DIR}")

    # 步骤6：与实验1（时域方法）对比（需加载实验1结果）
    try:
        with open(os.path.join(MODEL_DIR, "experiment1_results.pkl"), "rb") as f:
            exp1_result = pickle.load(f)
        print(f"\n=== 时域vs频域方法对比 ===")
        print(f"实验1（时域）正确率：{exp1_result['accuracy']:.4f}")
        print(f"实验2（频域）正确率：{accuracy:.4f}")
        print(f"正确率提升：{accuracy - exp1_result['accuracy']:.4f}")
    except FileNotFoundError:
        print("\n提示：未找到实验1结果文件，无法进行对比分析")


# -------------------------- 6. 实验1结果保存适配（用于对比） --------------------------
def save_experiment1_result(accuracy, cm, labels):
    """适配实验1，保存结果用于与实验2对比"""
    result_dict = {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "labels": labels
    }
    with open(os.path.join(MODEL_DIR, "experiment1_results.pkl"), "wb") as f:
        pickle.dump(result_dict, f)


# -------------------------- 主函数（执行实验2） --------------------------
if __name__ == "__main__":
    # 执行实验2
    experiment2()