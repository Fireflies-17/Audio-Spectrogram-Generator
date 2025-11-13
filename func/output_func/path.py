import os
from datetime import datetime
import numpy as np
from scipy.io import wavfile


def generate_output_path(prefix="spectrogram", extension="png"):
    """
    生成带时间戳的输出文件路径

    参数:
        prefix (str): 文件名前缀
        extension (str): 文件扩展名

    返回:
        str: 完整的输出路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.{extension}"
    output_dir = "data/output_data"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return os.path.join(output_dir, filename)


def export_to_wav(audio_data, sample_rate, prefix="demodulated"):
    """
    将音频数据导出为WAV文件
    
    参数:
        audio_data (np.ndarray): 音频数据
        sample_rate (int): 采样率
        prefix (str): 文件名前缀
        
    返回:
        str: 保存的文件路径
    """
    # 生成输出路径
    output_path = generate_output_path(prefix=prefix, extension="wav")
    
    # 归一化音频数据到 [-1, 1] 范围
    audio_normalized = audio_data / np.max(np.abs(audio_data))
    
    # 转换为 16-bit PCM 格式
    audio_int16 = np.int16(audio_normalized * 32767)
    
    # 保存为 WAV 文件
    wavfile.write(output_path, sample_rate, audio_int16)
    
    print(f"Demodulated audio saved to: {output_path}")
    
    return output_path
