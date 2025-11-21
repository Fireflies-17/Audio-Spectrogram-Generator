import numpy as np
import pywt
from func.plot_func.cwt_spectrogram import cwt_plot_scalogram
from func.analysis_func.filter import lowpass_filter


def perform_cwt_pywt(audio_data, sample_rate, scales, wavelet='morl'):
    """
    使用PyWavelets对音频数据执行连续小波变换(CWT)
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        scales (np.ndarray): 尺度数组，控制频率分辨率
        wavelet (str): 小波基函数，默认'morl'(Morlet小波)
                      其他选项: 'mexh'(墨西哥帽), 'gaus1'-'gaus8'(高斯), 'cgau1'-'cgau8'(复高斯), 'cmor'(复Morlet)
        
    返回:
        coefficients (np.ndarray): CWT系数矩阵，shape为(len(scales), len(audio_data))
        frequencies (np.ndarray): 对应的频率数组
    """
    # 执行连续小波变换
    coefficients, frequencies = pywt.cwt(
        audio_data,
        scales,
        wavelet,
        sampling_period=1.0/sample_rate
    )
    
    return coefficients, frequencies


def analyze_audio_with_cwt_pywt(audio_data, sample_rate, scales=None, wavelet='morl',
                                 max_len=5000, save_path=None, vmin=-80,
                                 filter_cutoff_freq=None, filter_order=5,
                                 scale_min=1, scale_max=128, scale_count=256):
    """
    对音频进行完整的CWT分析并可视化
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        scales (np.ndarray): 尺度数组，如果为None则自动生成
        wavelet (str): 小波基函数，默认'morl'
        max_len (int): 最大显示频率
        save_path (str): 图像保存路径
        vmin (float): 颜色映射的最小值（dB），默认-80
        filter_cutoff_freq (float): 低通滤波器截止频率 (Hz)，默认None表示不使用滤波
        filter_order (int): 低通滤波器阶数，默认5
        scale_min (int): 最小尺度值，默认1
        scale_max (int): 最大尺度值，默认128
        scale_count (int): 尺度数量，默认256
    """
    
    # 在CWT之前应用低通滤波
    if filter_cutoff_freq is not None:
        print(f"\nApplying lowpass filter before CWT (cutoff: {filter_cutoff_freq} Hz)...")
        audio_data = lowpass_filter(audio_data, sample_rate, filter_cutoff_freq, order=filter_order)
    
    # 如果未提供尺度数组，则自动生成
    if scales is None:
        scales = np.arange(scale_min, scale_max, (scale_max - scale_min) / scale_count)
        print(f"\nGenerating scales: {scale_count} scales from {scale_min} to {scale_max}")
    
    # 执行CWT
    print(f"\nPerforming CWT transformation with wavelet '{wavelet}'...")
    coefficients, frequencies = perform_cwt_pywt(audio_data, sample_rate, scales, wavelet)
    
    # 绘制CWT频谱图（scalogram）
    print("\nPlotting CWT scalogram...")
    cwt_plot_scalogram(
        coefficients, frequencies, audio_data, sample_rate,
        wavelet, scales, max_len,
        save_path=save_path, vmin=vmin,
        scale_min=scale_min, scale_max=scale_max, scale_count=scale_count,
        filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order
    )
    
    print("\nDone. CWT scalogram generated successfully.")
