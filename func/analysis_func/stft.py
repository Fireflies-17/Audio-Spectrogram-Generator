import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from matplotlib import font_manager
from func.plot_func.spectrogram import plot_spectrogram, plot_mel_spectrogram


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
    # 使用librosa原生STFT函数
    stft_result = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
    
    # 计算频率和时间轴
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(stft_result.shape[1]), sr=sample_rate, hop_length=hop_length)
    
    return stft_result, frequencies, times


def analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, n_mels, max_len,
                            save_path=None):
    """
    对音频进行完整的STFT分析并可视化
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        n_mels (int): Mel频带数量
        max_len (int): 最大显示频率
        save_path (str): 图像保存路径
        is_demodulated (bool): 是否为解调信号
        title_suffix (str): 标题后缀
    """
    
    # 执行STFT
    print("\nPerforming STFT transformation...")
    stft_result, frequencies, times = perform_stft(audio_data, sample_rate, n_fft, hop_length)
    
    # 绘制标准频谱图
    print("\nPlotting standard spectrogram...")
    plot_spectrogram(stft_result, sample_rate, hop_length, n_fft, max_len,
                     save_path=save_path)
    print("Standard spectrogram plotted.")
    
    # 绘制Mel频谱图
    #print("\nComputing Mel spectrogram...")
    #mel_save_path = save_path.replace('.png', '_mel.png') if save_path else None
    #plot_mel_spectrogram(audio_data, sample_rate, n_fft, hop_length, n_mels, max_len, save_path=mel_save_path)

    print("\nDone. Spectrograms generated successfully.")