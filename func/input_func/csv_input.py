import numpy as np
import pandas as pd
import re


def load_data_from_csv(file_path, sample_rate=None, channel='CH2V'):
    """
    从CSV文件加载单通道采样数据
    
    参数:
        file_path (str): CSV文件路径
        sample_rate (int): 采样率，如果为None则从文件头读取，单位Hz
        channel (str): 要读取的通道，'CH1V'或'CH2V'，默认'CH2V'
        
    返回:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
    """
    try:
        # 读取文件头部信息
        with open(file_path, 'r') as f:
            header_line = f.readline()
        
        # 使用正则表达式提取 tInc
        match_tinc = re.search(r'tInc\s*=\s*([-\d.e+]+)', header_line, re.IGNORECASE)
        
        if not match_tinc:
            raise ValueError("Cannot find tInc in the CSV header.")
        
        tInc = float(match_tinc.group(1))
        
        # 计算采样率
        if sample_rate is None:
            sample_rate = int(1 / tInc)
        
        # 读取数据部分（跳过第一行表头）
        df = pd.read_csv(
            file_path,
            skiprows=1,
            usecols=[0, 1],
            header=None,
            names=['CH1V', 'CH2V']
        )
        
        # 移除NaN值
        df = df.dropna()
        
        # 提取指定通道的数据
        if channel not in df.columns:
            raise ValueError(f"Channel {channel} does not exist in CSV file")
        
        data = df[channel].values
        n_points = len(data)
        duration = n_points / sample_rate
        
        print(f"Loading: {file_path}")
        print(f"  Sample rate: {sample_rate / 1e6:.2f} MSa/s")
        print(f"  Duration: {duration:.2f} seconds, {n_points} samples")
        print(f"  Channel: {channel}")
        
        return data, sample_rate

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None
