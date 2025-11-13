import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib import font_manager
from func.plot_func.spectrogram import plot_spectrogram, plot_mel_spectrogram
from func.analysis_func.filter import lowpass_filter


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def perform_stft(audio_data, sample_rate, n_fft, hop_length, win_length, window='hann'):
    """
    对音频数据执行STFT变换
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        win_length (int): 窗口长度
        window (str): 窗口函数类型，默认'hann'
        
    返回:
        stft_result (np.ndarray): STFT复数结果
        frequencies (np.ndarray): 频率数组
        times (np.ndarray): 时间数组
    """
    # 使用librosa原生STFT函数
    stft_result = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    
    # 计算频率和时间轴
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=sample_rate, hop_length=hop_length)
    
    return stft_result, frequencies, times


def analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, win_length, n_mels, max_len,
                            window='hann', save_path=None, vmin=-80, filter_cutoff_freq=None, filter_order=5):
    """
    对音频进行完整的STFT分析并可视化
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        win_length (int): 窗口长度
        n_mels (int): Mel频带数量
        max_len (int): 最大显示频率
        window (str): 窗口函数类型，默认'hann'
        save_path (str): 图像保存路径
        vmin (float): 颜色映射的最小值（dB），默认-80
        filter_cutoff_freq (float): 低通滤波器截止频率 (Hz)，默认None表示不使用滤波
        filter_order (int): 低通滤波器阶数，默认5
    """
    
    # 在STFT之前应用低通滤波
    if filter_cutoff_freq is not None:
        print(f"\nApplying lowpass filter before STFT (cutoff: {filter_cutoff_freq} Hz)...")
        audio_data = lowpass_filter(audio_data, sample_rate, filter_cutoff_freq, order=filter_order)
    
    # 执行STFT
    print("\nPerforming STFT transformation...")
    stft_result, frequencies, times = perform_stft(audio_data, sample_rate, n_fft, hop_length, win_length, window)
    
    # 绘制标准频谱图
    print("\nPlotting standard spectrogram...")
    plot_spectrogram(stft_result, sample_rate, hop_length, win_length, window, n_fft, max_len,
                     save_path=save_path, vmin=vmin)
    print("Standard spectrogram plotted.")
    
    # 绘制Mel频谱图
    #print("\nComputing Mel spectrogram...")
    #mel_save_path = save_path.replace('.png', '_mel.png') if save_path else None
    #plot_mel_spectrogram(audio_data, sample_rate, n_fft, hop_length, win_length, window, n_mels, max_len, save_path=mel_save_path, vmin=vmin)

    print("\nDone. Spectrograms generated successfully.")