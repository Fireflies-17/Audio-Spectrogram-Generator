import pyaudio
import numpy as np
import wave


def record_audio_from_mic(duration=5, sample_rate=44100, channels=1, chunk_size=1024):
    """
    从麦克风录制音频
    
    参数:
        duration (float): 录制时长（秒）
        sample_rate (int): 采样率，默认44100Hz
        channels (int): 声道数，默认1（单声道）
        chunk_size (int): 每次读取的音频帧数
        
    返回:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
    """
    try:
        print(f"Start recording...")
        
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        
        # 打开音频流
        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        frames = []
        
        # 计算需要读取的帧数
        num_chunks = int(sample_rate / chunk_size * duration)
        
        # 录制音频
        for i in range(num_chunks):
            data = stream.read(chunk_size)
            frames.append(data)
        
        print("Done.")
        
        # 停止并关闭流
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # 将录制的数据转换为numpy数组
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        
        # 归一化到[-1, 1]区间
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def save_recorded_audio(audio_data, sample_rate, output_path="recorded_audio.wav"):
    """
    保存录制的音频到文件
    
    参数:
        audio_data (np.ndarray): 音频数据
        sample_rate (int): 采样率
        output_path (str): 输出文件路径
    """
    try:
        # 将归一化的数据转换回int16格式
        audio_int16 = (audio_data * 32768.0).astype(np.int16)
        
        # 保存为WAV文件
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        print(f"Saved.")

    except Exception as e:
        print(f"Error: {e}")
