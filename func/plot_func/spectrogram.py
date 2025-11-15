import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_spectrogram(stft_result, sample_rate, hop_length, win_length, window, n_fft,
                     max_len, save_path=None, cmap='jet', vmin=-80):
    """
    绘制频谱图

    参数:
        stft_result (np.ndarray): STFT结果
        sample_rate (int): 采样率
        hop_length (int): 帧移大小
        win_length (int): 窗口长度
        window (str): 窗口函数类型
        n_fft (int): FFT窗口大小
        max_len (int): 最大显示频率
        save_path (str): 保存路径，如果为None则不保存
        cmap (str): 颜色映射方案
        vmin (float): 颜色映射的最小值（dB），默认-80
    """
    plt.figure(figsize=(22, 18), dpi=400)

    # 转换为dB刻度
    magnitude_db = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)

    # 绘制频谱图热力图
    img = librosa.display.specshow(
        magnitude_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='hz',
        cmap=cmap,  # 使用jet配色：蓝紫色(低)到红色(高)
        vmin=vmin,  # 颜色映射的最小值
        vmax=0  # 颜色映射的最大值
    )

    # 调整colorbar(图例栏)
    cbar = plt.colorbar(img, format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=24)  # colorbar刻度数字大小

    plt.xlabel('Time(s)', fontsize=36, labelpad=16)
    plt.ylabel('Frequency(Hz)', fontsize=36, labelpad=16)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # 简化标题
    plt.title("频谱图", fontsize=40, pad=20)
    plt.ylim(0, max_len)  # 限制显示频率范围
    
    # 在图形底部添加参数说明（白色背景，无边框）
    param_text = f'Sample Rate = {sample_rate} Hz  |  FFT Size = {n_fft}  |  Hop Length = {hop_length}  |  Window Length = {win_length}  |  Window = {window}'
    plt.figtext(0.5, 0.015, param_text, 
                ha='center', fontsize=28,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为底部文字留出更多空间

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_mel_spectrogram(audio_data, sample_rate, n_fft, hop_length, win_length, window, n_mels,
                         max_len, save_path=None, vmin=-80):
    """
    绘制Mel频谱图（更符合人耳感知，热力图形式）

    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        win_length (int): 窗口长度
        window (str): 窗口函数类型
        n_mels (int): Mel滤波器数量
        max_len (int): 最大显示频率
        save_path (str): 保存路径
        vmin (float): 颜色映射的最小值（dB），默认-80
    """
    plt.figure(figsize=(22, 18), dpi=300)

    # 使用librosa原生函数计算Mel频谱图
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
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
        cmap='jet',  # 使用jet配色：蓝紫色(低)到红色(高)
        vmin=vmin,  # 颜色映射的最小值
        vmax=0  # 颜色映射的最大值
    )

    # 调整colorbar(图例栏)
    cbar = plt.colorbar(img, format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=24)  # colorbar刻度数字大小

    plt.xlabel('Time(s)', fontsize=36, labelpad=16)
    plt.ylabel('Frequency(Hz)', fontsize=36, labelpad=16)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # 简化标题
    plt.title("Mel频谱图", fontsize=40, pad=20)
    plt.ylim(0, max_len)  # 限制显示频率范围
    
    # 在图形底部添加参数说明（Mel频谱图包含n_mels参数，白色背景，无边框）
    param_text = f'Sample Rate = {sample_rate} Hz  |  FFT Size = {n_fft}  |  Hop Length = {hop_length}  |  Window Length = {win_length}  |  Window = {window}  |  Mel Bands = {n_mels}'
    plt.figtext(0.5, 0.015, param_text, 
                ha='center', fontsize=28,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为底部文字留出更多空间

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()