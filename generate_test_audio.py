import numpy as np
import wave


def generate_test_audio(filename="test_audio.wav", duration=3, sample_rate=48000):
    """
    生成包含多个频率的测试音频
    
    参数:
        filename (str): 输出文件名
        duration (float): 音频时长（秒）
        sample_rate (int): 采样率
    """
    # 生成时间轴
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 创建包含多个频率成分的信号
    # 440Hz (A4音符)
    signal1 = 0.3 * np.sin(2 * np.pi * 440 * t)
    
    # 880Hz (A5音符)
    signal2 = 0.2 * np.sin(2 * np.pi * 880 * t)
    
    # 1320Hz
    signal3 = 0.1 * np.sin(2 * np.pi * 1320 * t)
    
    # 添加一些调制效果（让频谱图更有趣）
    # 频率随时间变化的信号（chirp）
    chirp = 0.2 * np.sin(2 * np.pi * (200 + 300 * t / duration) * t)
    
    # 组合所有信号
    audio = signal1 + signal2 + signal3 + chirp
    
    # 归一化
    audio = audio / np.max(np.abs(audio))
    
    # 转换为int16格式
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # 保存为WAV文件
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 2字节 (int16)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"测试音频已生成: {filename}")
    print(f"时长: {duration}秒")
    print(f"采样率: {sample_rate} Hz")
    print(f"包含频率: 440Hz, 880Hz, 1320Hz, 以及从200Hz到500Hz的扫频")


if __name__ == "__main__":
    generate_test_audio()
