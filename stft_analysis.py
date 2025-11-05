import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib import font_manager


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def perform_stft(audio_data, sample_rate, n_fft=8192, hop_length=128):
    """
    对音频数据执行STFT变换
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        
    返回:
        stft_result (np.ndarray): STFT复数结果
        frequencies (np.ndarray): 频率数组
        times (np.ndarray): 时间数组
    """
    # 执行STFT
    stft_result = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    
    # 计算频率和时间轴
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=sample_rate, hop_length=hop_length)
    
    return stft_result, frequencies, times


def plot_spectrogram(stft_result, sample_rate, hop_length=128, title="STFT频谱图", save_path=None, cmap='jet'):
    """
    绘制频谱图
    
    参数:
        stft_result (np.ndarray): STFT结果
        sample_rate (int): 采样率
        hop_length (int): 帧移大小
        title (str): 图表标题
        save_path (str): 保存路径，如果为None则不保存
        cmap (str): 颜色映射方案，默认'jet'（蓝紫->红）
    """
    plt.figure(figsize=(20, 16), dpi=300)
    
    # 转换为dB刻度
    magnitude_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
    # 绘制频谱图热力图
    img = librosa.display.specshow(
        magnitude_db, 
        sr=sample_rate, 
        hop_length=hop_length,
        x_axis='time', 
        y_axis='hz',
        cmap=cmap  # 使用jet配色：蓝紫色(低)到红色(高)
    )

    plt.colorbar(img, format='%+2.0f dB', label='dB')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 5000)  # 限制显示频率范围为 0-5000Hz
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def plot_mel_spectrogram(audio_data, sample_rate, n_fft=8192, hop_length=128, n_mels=1024, title="Mel频谱图", save_path=None):
    """
    绘制Mel频谱图（更符合人耳感知，热力图形式）

    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        n_mels (int): Mel滤波器数量
        title (str): 图表标题
        save_path (str): 保存路径
    """
    plt.figure(figsize=(20, 16), dpi=300)

    # 计算Mel频谱图
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # 转换为dB刻度
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 绘制Mel频谱图热力图
    img = librosa.display.specshow(
        mel_spectrogram_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='jet'  # 使用jet配色：蓝紫色(低)到红色(高)
    )

    plt.colorbar(img, format='%+2.0f dB', label='dB')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylim(0, 5000)  # 限制显示频率范围为 0-5000Hz
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_audio_with_stft(audio_data, sample_rate, n_fft=8192, hop_length=128, save_path=None):
    """
    对音频进行完整的STFT分析并可视化
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        save_path (str): 图像保存路径
    """
    print("Starting analysis...")
    
    # 执行STFT
    stft_result, frequencies, times = perform_stft(audio_data, sample_rate, n_fft, hop_length)
    
    # 绘制标准频谱图
    plot_spectrogram(stft_result, sample_rate, hop_length, save_path=save_path)
    
    # 绘制Mel频谱图
    mel_save_path = save_path.replace('.png', '_mel.png') if save_path else None
    plot_mel_spectrogram(audio_data, sample_rate, n_fft, hop_length, save_path=mel_save_path)

    print("Done.")