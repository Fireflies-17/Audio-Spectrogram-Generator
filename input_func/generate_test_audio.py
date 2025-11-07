import numpy as np
import wave


def generate_test_audio(filename="test_audio.wav", duration=10, sample_rate=48000):
    """
    生成包含丰富频率成分的测试音频（10秒）
    
    参数:
        filename (str): 输出文件名
        duration (float): 音频时长（秒）
        sample_rate (int): 采样率
    """
    # 生成时间轴
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # ===== 基础音符频率 (音乐和声) =====
    # 440Hz (A4音符 - 主旋律)
    signal1 = 0.25 * np.sin(2 * np.pi * 440 * t)
    
    # 554.37Hz (C#5 - 大三度和声)
    signal2 = 0.15 * np.sin(2 * np.pi * 554.37 * t)
    
    # 659.25Hz (E5 - 五度和声)
    signal3 = 0.12 * np.sin(2 * np.pi * 659.25 * t)
    
    # 880Hz (A5 - 八度音)
    signal4 = 0.18 * np.sin(2 * np.pi * 880 * t)
    
    # 1320Hz (E6 - 高频泛音)
    signal5 = 0.08 * np.sin(2 * np.pi * 1320 * t)
    
    # ===== 低频节拍和律动 =====
    # 60Hz - 低频脉动 (类似心跳)
    bass_pulse = 0.2 * np.sin(2 * np.pi * 60 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    
    # 110Hz - 低音节奏 (A2音符)
    bass_rhythm = 0.15 * np.sin(2 * np.pi * 110 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))
    
    # ===== 频率扫描 (Chirp Signals) =====
    # 从200Hz到1000Hz的线性扫频 (上升)
    chirp_up = 0.15 * np.sin(2 * np.pi * (200 + 800 * t / duration) * t)
    
    # 从1500Hz到500Hz的下降扫频
    chirp_down = 0.12 * np.sin(2 * np.pi * (1500 - 1000 * t / duration) * t)
    
    # 正弦波形的频率调制 (FM合成)
    fm_signal = 0.1 * np.sin(2 * np.pi * 300 * t + 5 * np.sin(2 * np.pi * 5 * t))
    
    # ===== 节奏脉冲 =====
    # 创建4Hz的脉冲包络
    pulse_envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    pulse_envelope = np.power(pulse_envelope, 3)  # 使脉冲更尖锐
    
    # 应用脉冲包络到1760Hz载波
    pulse_signal = 0.12 * np.sin(2 * np.pi * 1760 * t) * pulse_envelope
    
    # ===== 颤音效果 (Tremolo) =====
    # 在350Hz上加入6Hz的振幅调制
    tremolo = 0.1 * np.sin(2 * np.pi * 350 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 6 * t))
    
    # ===== 高频亮色 =====
    # 2000Hz - 高频泛音
    high_freq1 = 0.08 * np.sin(2 * np.pi * 2000 * t) * (0.5 + 0.5 * np.cos(2 * np.pi * 0.5 * t))
    
    # 3000Hz - 更高频率 (逐渐衰减)
    high_freq2 = 0.06 * np.sin(2 * np.pi * 3000 * t) * np.exp(-0.1 * t)
    
    # ===== 组合所有信号 =====
    audio = (signal1 + signal2 + signal3 + signal4 + signal5 + 
             bass_pulse + bass_rhythm + 
             chirp_up + chirp_down + fm_signal + 
             pulse_signal + tremolo + 
             high_freq1 + high_freq2)
    
    # 添加渐入渐出效果，让音频更自然
    fade_in = np.linspace(0, 1, int(sample_rate * 0.5))  # 0.5秒渐入
    fade_out = np.linspace(1, 0, int(sample_rate * 0.5))  # 0.5秒渐出
    audio[:len(fade_in)] *= fade_in
    audio[-len(fade_out):] *= fade_out
    
    # 归一化到合适的音量
    audio = audio / np.max(np.abs(audio)) * 0.95  # 留一点余量防止削波
    
    # 转换为int16格式
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # 保存为WAV文件
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 2字节 (int16)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"✨ 测试音频已生成: {filename}")
    print(f"⏱️  时长: {duration}秒")
    print(f"🎵 采样率: {sample_rate} Hz")
    print(f"\n📊 包含的频率成分:")
    print(f"  🎹 和声音符: 440Hz(A4), 554Hz(C#5), 659Hz(E5), 880Hz(A5), 1320Hz(E6)")
    print(f"  🥁 低频节奏: 60Hz(脉动), 110Hz(低音)")
    print(f"  🌊 扫频信号: 200→1000Hz(上升), 1500→500Hz(下降)")
    print(f"  ✨ 特殊效果: FM调制(300Hz), 脉冲(1760Hz), 颤音(350Hz)")
    print(f"  💎 高频泛音: 2000Hz, 3000Hz")
    print(f"\n🎨 这将生成一个富有层次的频谱图！")


if __name__ == "__main__":
    generate_test_audio()
