import os
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
from scipy.signal import lfilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 新增特征标准化
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # Fisher线性判别
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pickle
import librosa
# 切换为稳定的后端（TkAgg兼容性好，无需额外依赖）
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# 设置支持中文的字体（根据系统选择，Windows推荐SimHei，Mac推荐Heiti TC）
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class ExperimentConfig:
    """实验参数配置类，统一管理所有可配置参数"""

    def __init__(self):
        # 语音采集与处理参数
        self.SAMPLE_RATE = 22050  # 采样率（Hz）
        self.DURATION = 1.5  # 单条录音时长（秒）
        self.N_FFT = 512  # FFT点数
        self.HOP_LENGTH = int(self.SAMPLE_RATE * 0.01)  # 帧移（10ms）
        self.FRAME_LENGTH = int(self.SAMPLE_RATE * 0.02)  # 帧长（20ms）
        self.HAMMING_WINDOW = np.hamming(self.FRAME_LENGTH)  # 汉明窗
        self.PRE_EMPHASIS = 0.97  # 预加重系数

        # 路径配置
        self.DATA_DIR = "voice_data"  # 语音数据保存目录
        self.MODEL_DIR = "models"  # 模型与结果保存目录

        # 实验参数
        self.NUM_SAMPLES_PER_LABEL = 20  # 每个类别录制样本数
        self.TEST_SIZE = 0.3  # 测试集占比
        self.RANDOM_STATE = 48  # 随机种子（保证实验可复现）


# 初始化配置
config = ExperimentConfig()


def validate_wav_file(file_path):
    """校验WAV文件格式是否符合实验要求"""
    try:
        sr, data = wavfile.read(file_path)
        if sr != config.SAMPLE_RATE:
            raise ValueError(f"采样率错误（需{config.SAMPLE_RATE}Hz，实际{sr}Hz）")
        if data.dtype != np.int16:
            raise ValueError(f"数据类型错误（需int16，实际{data.dtype}）")
        return True
    except Exception as e:
        print(f"文件校验失败：{file_path} - {str(e)}")
        return False


def preprocess_audio(audio):
    """语音预处理：预加重→分帧→端点检测"""
    # 转换为float32并归一化
    audio_float = audio.astype(np.float32) / 32767.0

    # 预加重（增强高频分量）
    pre_emphasis = config.PRE_EMPHASIS
    audio_pre = np.append(audio_float[0], audio_float[1:] - pre_emphasis * audio_float[:-1])

    # 分帧
    frames = librosa.util.frame(
        audio_pre,
        frame_length=config.FRAME_LENGTH,
        hop_length=config.HOP_LENGTH
    ).T.copy()
    frames *= config.HAMMING_WINDOW  # 加窗

    # 计算短时能量和过零率（用于端点检测）
    energy = np.sum(frames ** 2, axis=1)
    zcr = []
    for frame in frames:
        frame_dc = frame - np.mean(frame)  # 去直流分量
        sgn = np.sign(frame_dc)
        zcr_frame = 0.5 * np.sum(np.abs(sgn[1:] - sgn[:-1]))  # 计算过零率
        zcr.append(zcr_frame / config.FRAME_LENGTH)  # 归一化
    zcr = np.array(zcr)

    # 自适应阈值计算（避免异常值影响）
    energy_thresh = max(np.mean(energy) / 5, np.median(energy) / 3)
    zcr_thresh = min(np.mean(zcr) * 2, np.median(zcr) * 3)
    valid_frames = (energy > energy_thresh) & (zcr < zcr_thresh)

    # 提取有效语音段
    if np.sum(valid_frames) == 0:
        return audio_pre  # 无有效帧时返回原始预处理信号
    start_idx = max(0, np.argmax(valid_frames) - 2)  # 向前扩展2帧
    end_idx = min(len(valid_frames) - 1, len(valid_frames) - np.argmax(valid_frames[::-1]) + 1)  # 向后扩展1帧
    start_sample = start_idx * config.HOP_LENGTH
    end_sample = (end_idx + 1) * config.HOP_LENGTH
    return audio_pre[start_sample:end_sample]


def extract_time_features(audio):
    """提取时域特征（短时能量、过零率及其统计量）"""
    # 分帧
    frames = librosa.util.frame(
        audio,
        frame_length=config.FRAME_LENGTH,
        hop_length=config.HOP_LENGTH
    ).T.copy()
    frames *= config.HAMMING_WINDOW

    # 短时能量统计量
    energy = np.sum(frames ** 2, axis=1)
    energy_mean = np.mean(energy)
    energy_max = np.max(energy)
    energy_std = np.std(energy)

    # 过零率统计量
    zcr = []
    for frame in frames:
        frame_dc = frame - np.mean(frame)
        sgn = np.sign(frame_dc)
        zcr_frame = 0.5 * np.sum(np.abs(sgn[1:] - sgn[:-1]))
        zcr.append(zcr_frame / config.FRAME_LENGTH)
    zcr_mean = np.mean(zcr)

    return np.array([energy_mean, energy_max, zcr_mean, energy_std])


def extract_freq_features(audio):
    """提取频域特征（Mel频谱→MFCC），确保输出固定13维"""
    try:
        # 分帧
        frames = librosa.util.frame(
            audio,
            frame_length=config.FRAME_LENGTH,
            hop_length=config.HOP_LENGTH
        ).T.copy()
        if len(frames) == 0:  # 无有效帧时返回默认特征
            return np.zeros(13)

        frames *= config.HAMMING_WINDOW
        fft_result = np.fft.fft(frames, n=config.N_FFT)
        fft_spec = np.abs(fft_result)[:, :config.N_FFT // 2 + 1]  # 仅保留正频率

        # Mel频谱转换
        mel_filter = librosa.filters.mel(
            sr=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            n_mels=40
        )
        mel_spec = np.dot(fft_spec, mel_filter.T)
        mel_spec = 20 * np.log10(mel_spec + 1e-6)

        # 提取MFCC并确保维度为13
        mfcc = librosa.feature.mfcc(S=mel_spec, n_mfcc=13)
        if mfcc.shape[1] == 0:  # 无MFCC帧时返回默认特征
            return np.zeros(13)

        return np.mean(mfcc, axis=1)  # 固定13维

    except Exception as e:
        print(f"特征提取失败：{str(e)}，返回默认特征")
        return np.zeros(13)  # 异常时返回0向量


def extract_speaker_features(audio):
    """提取说话人特征（LPC+MFCC均值+基音周期）"""
    # LPC系数（12阶，反映声道特性）
    lpc = librosa.lpc(audio, order=12)

    # MFCC均值（13维，全局频谱特征）
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mfcc=13
    )
    mfcc_mean = np.mean(mfcc, axis=1)

    # 基音周期（反映声门特性）
    f0, _, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),  # 最低基音（男声）
        fmax=librosa.note_to_hz('C7')  # 最高基音（女声/童声）
    )
    f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0
    f0_std = np.nanstd(f0) if np.any(~np.isnan(f0)) else 0.0

    return np.concatenate([lpc, mfcc_mean, [f0_mean, f0_std]])


def record_voice(label):
    """录制指定类别的语音样本（在已有数据基础上追加）"""
    label_dir = os.path.join(config.DATA_DIR, str(label))
    os.makedirs(label_dir, exist_ok=True)

    # 查找已有样本最大索引
    existing_files = [f for f in os.listdir(label_dir) if f.endswith(".wav")]
    pattern = re.compile(f"^{label}_(\d+)\.wav$")
    indices = []
    for file in existing_files:
        match = pattern.match(file)
        if match:
            indices.append(int(match.group(1)))

    start_idx = max(indices) + 1 if indices else 0  # 确定新样本起始索引
    end_idx = start_idx + config.NUM_SAMPLES_PER_LABEL  # 计算结束索引

    print(f"\n===== 开始录制类别 [{label}] 的语音 =====")
    print(f"提示：每次录制时长{config.DURATION}秒，本次将录制{config.NUM_SAMPLES_PER_LABEL}条")
    print(f"样本编号将从 {start_idx} 开始（在已有{len(indices)}条基础上追加）")
    print(f"录制内容：清晰朗读数字[{label}]，背景尽量安静\n")

    recorded_count = 0  # 实际录制成功计数
    current_idx = start_idx

    while recorded_count < config.NUM_SAMPLES_PER_LABEL:
        # 等待用户准备
        input(f"按Enter键开始录制第{recorded_count + 1}/{config.NUM_SAMPLES_PER_LABEL}条（编号：{current_idx}）...")

        # 录音
        print("正在录制...")
        audio = sd.rec(
            int(config.DURATION * config.SAMPLE_RATE),
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        audio = np.squeeze(audio)

        # 静音检测（过滤无效录音）
        energy = np.sum(audio ** 2)
        if energy < 0.01:
            print("警告：录音能量过低（可能为静音），请重新录制！")
            continue  # 跳过当前计数，重新录制

        # 转换为int16格式并保存
        audio_int16 = (audio * 32767).astype(np.int16)
        save_path = os.path.join(label_dir, f"{label}_{current_idx}.wav")
        wavfile.write(save_path, config.SAMPLE_RATE, audio_int16)
        print(f"已保存：{save_path}\n")

        recorded_count += 1
        current_idx += 1

    print(f"类别 [{label}] 录制完成！当前总计 {len(indices) + config.NUM_SAMPLES_PER_LABEL} 条样本")


def load_dataset(feature_type="time"):
    """加载数据集，过滤维度异常的特征"""
    X, y = [], []
    feature_func = {
        "time": extract_time_features,
        "freq": extract_freq_features,
        "speaker": extract_speaker_features
    }[feature_type]

    # 先获取预期特征长度（从第一个有效样本提取）
    expected_dim = None

    for label in os.listdir(config.DATA_DIR):
        label_path = os.path.join(config.DATA_DIR, label)
        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if not file.endswith(".wav"):
                continue
            file_path = os.path.join(label_path, file)

            if not validate_wav_file(file_path):
                continue

            # 提取特征
            sr, audio = wavfile.read(file_path)
            audio_pre = preprocess_audio(audio)
            feature = feature_func(audio_pre)

            # 校验特征维度
            if expected_dim is None:
                expected_dim = len(feature)  # 初始化预期维度
                print(f"预期特征维度：{expected_dim}")

            if len(feature) != expected_dim:
                print(f"警告：{file_path} 特征维度异常（{len(feature)}≠{expected_dim}），已过滤")
                continue

            X.append(feature)
            y.append(label)

    if not X:
        raise ValueError("未加载到有效数据，请先录制语音样本！")
    return np.array(X), np.array(y)


def calculate_metrics(y_true, y_pred, labels):
    """计算评估指标（正确率、误纳率、误拒率）"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    correct = np.diag(cm).sum()
    acc = correct / total if total != 0 else 0.0  # 正确率

    # 误纳率（非目标被误判为目标）
    false_accept = (cm.sum(axis=0) - np.diag(cm)).sum()
    false_accept_rate = false_accept / (total - correct) if (total - correct) != 0 else 0.0

    # 误拒率（目标被误判为非目标）
    false_reject = (cm.sum(axis=1) - np.diag(cm)).sum()
    false_reject_rate = false_reject / total if total != 0 else 0.0

    return {
        "正确率": round(acc, 4),
        "误纳率": round(false_accept_rate, 4),
        "误拒率": round(false_reject_rate, 4)
    }, cm


def save_experiment_results(exp_name, best_clf_name, metrics, cm, labels, scaler=None):
    """保存实验结果（指标+混淆矩阵图+标准化器）"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    # 保存指标到文本文件
    metrics_path = os.path.join(config.MODEL_DIR, f"{exp_name}_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"===== {exp_name} 实验结果 =====\n")
        f.write(f"最佳分类器：{best_clf_name}\n")
        for k, v in metrics.items():
            f.write(f"{k}：{v}\n")
    print(f"指标已保存至：{metrics_path}")

    # 保存混淆矩阵图
    cm_path = os.path.join(config.MODEL_DIR, f"{exp_name}_confusion_matrix.png")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"{exp_name}（{best_clf_name}）混淆矩阵")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"混淆矩阵已保存至：{cm_path}")

    # 保存标准化器（如果有）
    if scaler:
        scaler_path = os.path.join(config.MODEL_DIR, f"{exp_name}_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        print(f"特征标准化器已保存至：{scaler_path}")


def run_experiment(exp_name, feature_type):
    """通用实验函数：测试多种分类器并选择最优者"""
    print(f"\n===== 运行{exp_name} =====")
    try:
        # 加载数据
        X, y = load_dataset(feature_type=feature_type)
        labels = np.unique(y)
        print(f"加载完成：{len(X)}个样本，{len(labels)}个类别，特征维度{X.shape[1]}")

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE
        )

        # 特征标准化（对尺度敏感的分类器需要）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 定义待测试的分类器集合
        classifiers = {
            "KNN(3近邻)": KNeighborsClassifier(n_neighbors=3),
            "朴素贝叶斯": GaussianNB(),
            "Fisher线性判别": LinearDiscriminantAnalysis(),
            "决策树(深度5)": DecisionTreeClassifier(max_depth=5, random_state=config.RANDOM_STATE),
            "SVM(RBF核)": SVC(kernel="rbf", C=10, gamma=0.01, random_state=config.RANDOM_STATE)
        }

        # 存储各分类器性能
        results = {}

        # 测试所有分类器
        print("\n开始测试各分类器性能：")
        for clf_name, clf in classifiers.items():
            # 对需要标准化特征的分类器使用标准化后的数据
            if clf_name in ["Fisher线性判别", "SVM(RBF核)"]:
                clf.fit(X_train_scaled, y_train)
                y_pred = clf.predict(X_test_scaled)
            else:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

            metrics, cm = calculate_metrics(y_test, y_pred, labels)
            results[clf_name] = {
                "model": clf,
                "metrics": metrics,
                "cm": cm,
                "y_pred": y_pred
            }
            print(f"{clf_name}：正确率={metrics['正确率']}，误纳率={metrics['误纳率']}，误拒率={metrics['误拒率']}")

        # 选择正确率最高的分类器作为最佳模型
        best_clf_name = max(results.items(), key=lambda x: x[1]["metrics"]["正确率"])[0]
        best_results = results[best_clf_name]
        print(f"\n最佳分类器：{best_clf_name}（正确率：{best_results['metrics']['正确率']}）")

        # 保存最佳模型和结果
        save_experiment_results(
            exp_name,
            best_clf_name,
            best_results["metrics"],
            best_results["cm"],
            labels,
            scaler  # 保存标准化器
        )

        # 保存最佳模型
        model_path = os.path.join(config.MODEL_DIR, f"{exp_name}_best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_results["model"], f)
        print(f"最佳模型已保存至：{model_path}")

    except Exception as e:
        print(f"{exp_name}运行失败：{str(e)}")


def experiment1():
    """实验1：基于时域特征的语音识别"""
    run_experiment("experiment1", "time")


def experiment2():
    """实验2：基于频域特征的语音识别"""
    run_experiment("experiment2", "freq")


def experiment3():
    """实验3：基于说话人特征的识别"""
    run_experiment("experiment3", "speaker")


def main():
    """主函数：提供交互界面，引导用户完成实验流程"""
    print("===== 数字信号处理实验：语音信号分析与识别 =====")
    while True:
        print("\n请选择操作：")
        print("1. 录制语音样本")
        print("2. 运行实验1（时域特征识别）")
        print("3. 运行实验2（频域特征识别）")
        print("4. 运行实验3（说话特征+频域特征识别）")
        print("5. 退出")

        choice = input("输入操作编号（1-5）：")
        if choice == "1":
            label = input("请输入录制的数字标签（如0-9）：")
            record_voice(label)
        elif choice == "2":
            experiment1()
        elif choice == "3":
            experiment2()
        elif choice == "4":
            experiment3()
        elif choice == "5":
            print("实验结束，再见！")
            break
        else:
            print("无效输入，请重新选择！")


if __name__ == "__main__":
    main()