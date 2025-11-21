import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import get_window
from matplotlib import font_manager
from func.plot_func.stft_spectrogram import stft_plot_spectrogram
from func.analysis_func.filter import lowpass_filter


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def perform_stft_scipy(audio_data, sample_rate, n_fft, hop_length, win_length, window='hann'):
    """
    使用scipy.signal.ShortTimeFFT对音频数据执行STFT变换
    
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
    # 生成窗函数
    win = get_window(window, win_length, fftbins=True)
    
    # 创建ShortTimeFFT对象
    # hop_length对应mfft (每帧之间的跳跃大小)
    # n_fft对应FFT大小
    stft_obj = ShortTimeFFT(win, hop=hop_length, fs=sample_rate, 
                            fft_mode='onesided', mfft=n_fft, 
                            scale_to='magnitude')
    
    # 执行STFT变换
    stft_result = stft_obj.stft(audio_data)
    
    # 获取频率和时间轴
    frequencies = stft_obj.f  # 频率数组
    times = stft_obj.t(len(audio_data))  # 时间数组
    
    return stft_result, frequencies, times


def analyze_audio_with_stft_scipy(audio_data, sample_rate, n_fft, hop_length, win_length, max_len,
                            window='hann', save_path=None, vmin=-80, filter_cutoff_freq=None, filter_order=5):
    """
    使用scipy对音频进行完整的STFT分析并可视化
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 帧移大小
        win_length (int): 窗口长度
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
    print("\nPerforming STFT transformation using scipy.signal.ShortTimeFFT...")
    stft_result, frequencies, times = perform_stft_scipy(audio_data, sample_rate, n_fft, hop_length, win_length, window)
    
    # 绘制标准频谱图
    print("\nPlotting standard spectrogram...")
    stft_plot_spectrogram(stft_result, sample_rate, hop_length, win_length, window, n_fft, max_len,
                          save_path=save_path, vmin=vmin)
    print("Standard spectrogram plotted.")
    
    print("\nDone. Spectrogram generated successfully using scipy.")
