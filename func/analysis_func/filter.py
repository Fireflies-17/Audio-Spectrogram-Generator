import numpy as np
from scipy import signal


def lowpass_filter(audio_data, sample_rate, cutoff_freq, order=5, filter_type='butter'):
    """
    对音频数据应用低通滤波器
    
    参数:
        audio_data (np.ndarray): 输入音频数据
        sample_rate (int): 采样率
        cutoff_freq (float): 截止频率 (Hz)
        order (int): 滤波器阶数，默认为5
        filter_type (str): 滤波器类型，'butter' (Butterworth) 或 'cheby1' (Chebyshev I)
        
    返回:
        filtered_data (np.ndarray): 滤波后的音频数据
    """
    if cutoff_freq is None or cutoff_freq <= 0:
        print("No lowpass filter applied (cutoff frequency not set)")
        return audio_data
    
    # 检查截止频率是否有效
    nyquist_freq = sample_rate / 2
    if cutoff_freq >= nyquist_freq:
        print(f"Warning: Cutoff frequency ({cutoff_freq} Hz) exceeds Nyquist frequency ({nyquist_freq} Hz)")
        print("No lowpass filter applied")
        return audio_data
    
    # 归一化截止频率
    normalized_cutoff = cutoff_freq / nyquist_freq
    
    # 设计滤波器
    if filter_type == 'butter':
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
    elif filter_type == 'cheby1':
        b, a = signal.cheby1(order, 0.5, normalized_cutoff, btype='low', analog=False)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}. Use 'butter' or 'cheby1'")
    
    # 应用滤波器
    filtered_data = signal.filtfilt(b, a, audio_data)
    
    print(f"Applied {filter_type} lowpass filter: cutoff={cutoff_freq} Hz, order={order}")
    
    return filtered_data
