import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm
from func.analysis_func.pitch_analysis import analyze_fundamental_frequency


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def perform_stft(audio_data, sample_rate, n_fft, hop_length):
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
    # 执行STFT，显示进度条
    print("执行STFT变换...")
    
    # 计算总帧数
    n_frames = 1 + (len(audio_data) - n_fft) // hop_length
    
    # 手动实现STFT以显示真实进度
    # 初始化结果数组
    stft_result = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    
    # 创建汉宁窗
    window = np.hanning(n_fft)
    
    # 使用进度条遍历每一帧
    with tqdm(total=n_frames, desc="STFT进度", unit="帧", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for frame_idx in range(n_frames):
            # 计算当前帧的起始位置
            start = frame_idx * hop_length
            end = start + n_fft
            
            # 提取当前帧的音频数据
            if end <= len(audio_data):
                frame = audio_data[start:end] * window
            else:
                # 处理边界情况，用零填充
                frame = np.zeros(n_fft)
                available = len(audio_data) - start
                if available > 0:
                    frame[:available] = audio_data[start:] * window[:available]
            
            # 执行FFT
            stft_result[:, frame_idx] = np.fft.rfft(frame)
            
            # 更新进度条
            pbar.update(1)
    
    # 计算频率和时间轴
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=sample_rate, hop_length=hop_length)
    
    return stft_result, frequencies, times


def plot_spectrogram(stft_result, sample_rate, hop_length, max_len, save_path=None, cmap='jet'):
    """
    绘制频谱图
    
    参数:
        stft_result (np.ndarray): STFT结果
        sample_rate (int): 采样率
        hop_length (int): 帧移大小
        max_len (int): 最大显示频率
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

    # 调整colorbar(图例栏)
    cbar = plt.colorbar(img, format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=24)  # colorbar刻度数字大小

    plt.xlabel('Time(s)', fontsize=36, labelpad=16)
    plt.ylabel('Frequency(Hz)', fontsize=36, labelpad=16)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.title(f"频谱图-SampleRate{sample_rate}-HopLength{hop_length}", fontsize=40, pad=20)
    plt.ylim(0, max_len)  # 限制显示频率范围为 0-5000Hz
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_mel_spectrogram(audio_data, sample_rate, n_fft, hop_length, n_mels, max_len, save_path=None):
    """
    绘制Mel频谱图（更符合人耳感知，热力图形式）

    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        n_mels (int): Mel滤波器数量
        max_len (int): 最大显示频率
        title (str): 图表标题
        save_path (str): 保存路径
    """
    plt.figure(figsize=(20, 16), dpi=300)

    # 计算总帧数
    n_frames = 1 + (len(audio_data) - n_fft) // hop_length
    
    # 手动计算STFT用于Mel频谱图
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    window = np.hanning(n_fft)
    
    # 第一步：计算STFT
    with tqdm(total=n_frames, desc="Mel STFT计算", unit="帧", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for frame_idx in range(n_frames):
            start = frame_idx * hop_length
            end = start + n_fft
            
            if end <= len(audio_data):
                frame = audio_data[start:end] * window
            else:
                frame = np.zeros(n_fft)
                available = len(audio_data) - start
                if available > 0:
                    frame[:available] = audio_data[start:] * window[:available]
            
            stft_matrix[:, frame_idx] = np.fft.rfft(frame)
            pbar.update(1)
    
    # 计算功率谱
    power_spec = np.abs(stft_matrix) ** 2
    
    # 第二步：应用Mel滤波器组
    mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    
    # 逐帧应用Mel滤波器
    mel_spectrogram = np.zeros((n_mels, n_frames))
    with tqdm(total=n_frames, desc="Mel滤波器应用", unit="帧", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for frame_idx in range(n_frames):
            mel_spectrogram[:, frame_idx] = np.dot(mel_basis, power_spec[:, frame_idx])
            pbar.update(1)

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

    # 调整colorbar(图例栏)
    cbar = plt.colorbar(img, format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=24)  # colorbar刻度数字大小

    plt.xlabel('Time(s)', fontsize=36, labelpad=16)
    plt.ylabel('Frequency(Hz)', fontsize=36, labelpad=16)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    plt.title(f"Mel频谱图-SampleRate{sample_rate}-Nfft{n_fft}-HopLength{hop_length}-Nmels{n_mels}", fontsize=40, pad=20)
    plt.ylim(0, max_len)  # 限制显示频率范围为 0-5000Hz
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()


def analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, n_mels, max_len, save_path=None):
    """
    对音频进行完整的STFT分析并可视化
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        save_path (str): 图像保存路径
    """
    
    print("\n=== 开始音频分析 ===")
    print(f"音频长度: {len(audio_data)} 采样点")
    print(f"采样率: {sample_rate} Hz")
    print(f"预计处理时长: {len(audio_data)/sample_rate:.2f} 秒\n")
    
    # 分析基频范围
    # print("Starting analysis...")
    # analyze_fundamental_frequency(audio_data, sample_rate)
    
    # 执行STFT
    stft_result, frequencies, times = perform_stft(audio_data, sample_rate, n_fft, hop_length)
    
    # 绘制标准频谱图
    print("\n绘制标准频谱图...")
    with tqdm(total=100, desc="频谱图生成", unit="%", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        plot_spectrogram(stft_result, sample_rate, hop_length, max_len, save_path=save_path)
        pbar.update(100)
    
    # 绘制Mel频谱图
    print("\n计算Mel频谱图...")
    mel_save_path = save_path.replace('.png', '_mel.png') if save_path else None
    plot_mel_spectrogram(audio_data, sample_rate, n_fft, hop_length, n_mels, max_len, save_path=mel_save_path)

    print("\n✓ 分析完成！")
    if save_path:
        print(f"✓ 文件已保存至: {save_path}")
        print(f"✓ Mel频谱图已保存至: {mel_save_path}")