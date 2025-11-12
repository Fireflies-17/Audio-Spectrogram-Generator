
import os
from datetime import datetime

from func.analysis_func.demodulate import demodulate_hilbert
from func.analysis_func.stft import analyze_audio_with_stft
from func.input_func.csv_input import load_data_from_csv
from func.input_func.wav_input import load_audio_from_file


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


def process_csv_file(sample_rate, n_fft, hop_length, n_mels, max_height, 
                     channel='CH1V', demodulated=False):
    """
    处理CSV格式的数据文件

    参数:
        sample_rate (int): 采样率，如果为None则从文件头读取
        n_fft (int): FFT窗口大小
        hop_length (int): 跳跃长度
        n_mels (int): Mel频带数量
        max_height (int): 最大频率高度
        channel (str): 要处理的通道，'CH1V'或'CH2V'
        demodulated (bool): 是否对指定通道执行解调操作，默认False
    """
    file_path = 'data/input_data/5_Fs2e6_tswp1s.csv' #input("Path: ")

    # 加载指定通道数据
    audio_data, sample_rate = load_data_from_csv(file_path, sample_rate, channel)
    
    if audio_data is not None:
        if demodulated:
            print(f"Demodulating {channel} signal...")
            audio_data = demodulate_hilbert(audio_data)
            
            # 对解调后的信号执行STFT分析
            print("\nPerforming STFT analysis on demodulated signal...")
            save_path = generate_output_path(
                prefix=f"demodulated_{channel}_stft", extension="png"
            )
            analyze_audio_with_stft(
                audio_data, sample_rate, n_fft, hop_length,
                n_mels, max_height, save_path=save_path
            )
            
            print("\nDemodulation analysis complete!")

        else:
            save_path = generate_output_path(prefix=f"csv_{channel}_stft", extension="png")
            analyze_audio_with_stft(
                audio_data, sample_rate, n_fft, hop_length,
                n_mels, max_height, save_path
            )


def process_wav_file(sample_rate, n_fft, hop_length, n_mels, max_height):
    """
    处理WAV格式的音频文件

    参数:
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 跳跃长度
        n_mels (int): Mel频带数量
        max_height (int): 最大频率高度
    """
    file_path = ''
    audio_data, sample_rate = load_audio_from_file(file_path, sample_rate)

    if audio_data is not None:
        # 自动生成输出路径
        save_path = generate_output_path(prefix="wav_stft", extension="png")
        analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, n_mels, max_height, save_path)