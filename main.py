import sys
import os
from datetime import datetime
from audio_file_input import load_audio_from_file
from audio_mic_input import record_audio_from_mic, save_recorded_audio
from data_txt_input import load_data_from_txt
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


def process_file_with_stft(sample_rate, n_fft, hop_length, n_mels, max_len):
    file_path = input("Path: ").strip('"').strip("'")
    audio_data, sample_rate = load_audio_from_file(file_path, sr=sample_rate)
    
    if audio_data is not None:
        # 自动生成输出路径
        save_path = generate_output_path(prefix="stft", extension="png")
        analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, n_mels, max_len, save_path=save_path)


def process_mic_with_stft(sample_rate, n_fft, hop_length, n_mels, max_len):
    try:
        duration = float(input("Record time: ").strip() or "5")
    except ValueError:
        duration = 5
        
    audio_data, sample_rate = record_audio_from_mic(duration=duration, sample_rate=sample_rate)
    
    if audio_data is not None:
        # 自动保存录音到output文件夹
        audio_path = generate_output_path(prefix="recorded", extension="wav")
        save_recorded_audio(audio_data, sample_rate, audio_path)
        
        # 自动生成频谱图输出路径
        save_path = generate_output_path(prefix="mic_stft", extension="png")

        analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, n_mels, max_len, save_path=save_path)


def process_txt_data_with_stft(sample_rate, n_fft, hop_length, n_mels, max_len):
    file_path = 'data.txt' #input("TXT file path: ").strip('"').strip("'")

    audio_data, sample_rate = load_data_from_txt(file_path, sample_rate=sample_rate)
    
    if audio_data is not None:
        # 自动生成输出路径
        save_path = generate_output_path(prefix="txt_stft", extension="png")
        analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, n_mels, max_len, save_path=save_path)


def main():
    sample_rate = 100000 #int(input("Sample rate: "))
    n_fft = 8192 #int(input("FFT size (n_fft): "))
    hop_length = 128 #int(input("Hop length: "))
    n_mels = 256 #int(input("Number of Mel bands: "))
    max_len = 2500 #int(input("Max length: "))

    #process_file_with_stft(sample_rate, n_fft, hop_length, n_mels, max_len)
    #process_mic_with_stft(sample_rate, n_fft, hop_length, n_mels, max_len)
    process_txt_data_with_stft(sample_rate, n_fft, hop_length, n_mels, max_len)


if __name__ == "__main__":
    main()
