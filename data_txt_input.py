import numpy as np


def load_data_from_txt(file_path, sample_rate=100000):
    """
    从txt文件加载采样数据
    
    参数:
        file_path (str): txt文件路径
        sample_rate (int): 采样率，默认100000Hz (100kHz)
        
    返回:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
    """
    try:
        # 从txt文件读取数据，每行一个采样点
        data = np.loadtxt(file_path)
        
        duration = len(data) / sample_rate
        print(f"Loading: {file_path}, {sample_rate} Hz, {duration:.2f} seconds, {len(data)} samples")
        
        return data, sample_rate

    except Exception as e:
        print(f"Error: {e}")
        return None, None
