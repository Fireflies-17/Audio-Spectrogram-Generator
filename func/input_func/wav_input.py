import librosa
import numpy as np


def load_audio_from_file(file_path, sr=44100):
    """
    从文件加载音频数据
    
    参数:
        file_path (str): 音频文件路径
        sr (int): 采样率，默认44100Hz
        
    返回:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
    """
    try:
        # 使用librosa加载音频文件
        audio_data, sample_rate = librosa.load(file_path, sr=sr)
        print(f"Loading: {file_path}, {sample_rate} Hz, {len(audio_data)/sample_rate:.2f} seconds")
        return audio_data, sample_rate

    except Exception as e:
        print(f"Error: {e}")
        return None, None
