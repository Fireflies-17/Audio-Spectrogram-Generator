"""
信号解调模块
提供多种解调方法用于从调制信号中提取原始信号
"""

import numpy as np
from scipy.signal import hilbert


def demodulate_hilbert(signal):
    """
    使用希尔伯特变换法进行信号解调
    
    原理：
    - 使用希尔伯特变换计算信号的解析信号
    - 提取包络作为解调结果
    - 移除直流分量
    
    参数:
        signal (np.ndarray): 输入的调制信号
        
    返回:
        demodulated_signal (np.ndarray): 解调后的信号
    """
    print("Demodulating signal using Hilbert transform...")
    
    # 使用希尔伯特变换计算解析信号
    analytic_signal = hilbert(signal)
    
    # 提取包络
    envelope = np.abs(analytic_signal)
    
    # 移除直流分量
    demodulated_signal = envelope - np.mean(envelope)
    
    print("Demodulation complete.")
    
    return demodulated_signal
