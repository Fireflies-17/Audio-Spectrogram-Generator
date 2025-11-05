import sys
import os
from datetime import datetime
from audio_file_input import load_audio_from_file
from audio_mic_input import record_audio_from_mic, save_recorded_audio
from stft_analysis import analyze_audio_with_stft


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
    return os.path.join("output", filename)


def process_file_with_stft():
    file_path = input("Path: ").strip('"').strip("'")
    audio_data, sample_rate = load_audio_from_file(file_path)
    
    if audio_data is not None:
        # 自动生成输出路径
        save_path = generate_output_path(prefix="stft", extension="png")
        analyze_audio_with_stft(audio_data, sample_rate, save_path=save_path)


def process_mic_with_stft():
    try:
        duration = float(input("Record time(5): ").strip() or "5")
    except ValueError:
        duration = 5
        
    audio_data, sample_rate = record_audio_from_mic(duration=duration)
    
    if audio_data is not None:
        # 自动保存录音到output文件夹
        audio_path = generate_output_path(prefix="recorded", extension="wav")
        save_recorded_audio(audio_data, sample_rate, audio_path)
        
        # 自动生成频谱图输出路径
        save_path = generate_output_path(prefix="mic_stft", extension="png")

        analyze_audio_with_stft(audio_data, sample_rate, save_path=save_path)


def main():
    process_file_with_stft()
    #process_mic_with_stft()
    exit()


if __name__ == "__main__":
    main()
