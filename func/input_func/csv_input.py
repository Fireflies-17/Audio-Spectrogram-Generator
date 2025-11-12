import numpy as np
import pandas as pd


def load_data_from_csv(file_path, sample_rate=None, channel='CH2V'):
    """
    从CSV文件加载采样数据
    
    参数:
        file_path (str): CSV文件路径
        sample_rate (int): 采样率，如果为None则从文件头读取，单位Hz
        channel (str): 要读取的通道，'CH1V'或'CH2V'，默认'CH2V'
        
    返回:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 从文件头读取采样率信息（tInc = 时间增量）
        if sample_rate is None:
            # 从第一行提取时间增量
            header_line = df.columns[-1]  # 最后一列包含t0和tInc信息
            if 'tInc' in header_line:
                # 提取tInc的值
                tInc_str = header_line.split('tInc =')[1].strip().rstrip(',')
                tInc = float(tInc_str)
                sample_rate = int(1 / tInc)
            else:
                raise ValueError("无法从CSV文件头读取采样率信息")
        
        # 提取指定通道的数据
        if channel not in df.columns:
            raise ValueError(f"通道 {channel} 不存在于CSV文件中")
        
        data = df[channel].values
        
        # 移除NaN值
        data = data[~np.isnan(data)]
        
        duration = len(data) / sample_rate
        print(f"Loading: {file_path}, {sample_rate} Hz, {duration:.2f} seconds, {len(data)} samples, channel: {channel}")
        
        return data, sample_rate

    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None, None
