import os
from datetime import datetime
from func.input_func.wav_input import load_audio_from_file
from func.input_func.csv_input import load_data_from_csv
from func.analysis_func.stft_analysis import analyze_audio_with_stft


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
    return os.path.join("dat/output_dat", filename)


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
    file_path = 'dat/input_dat/test_audio.wav' #input("Path: ").strip('"').strip("'")
    audio_data, sample_rate = load_audio_from_file(file_path, sample_rate)
    
    if audio_data is not None:
        # 自动生成输出路径
        save_path = generate_output_path(prefix="wav_stft", extension="png")
        analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, n_mels, max_height, save_path)


def process_csv_file(sample_rate, n_fft, hop_length, n_mels, max_height, channel='CH2V'):
    """
    处理CSV格式的数据文件
    
    参数:
        sample_rate (int): 采样率，如果为None则从文件头读取
        n_fft (int): FFT窗口大小
        hop_length (int): 跳跃长度
        n_mels (int): Mel频带数量
        max_height (int): 最大频率高度
        channel (str): 要处理的通道，'CH1V'或'CH2V'，默认'CH2V'
    """
    file_path = 'dat/input_dat/3_Fs2e6_tswp200ms.csv'  #input("CSV file path: ").strip('"').strip("'")

    audio_data, sample_rate = load_data_from_csv(file_path, sample_rate, channel)
    
    if audio_data is not None:
        # 自动生成输出路径
        save_path = generate_output_path(prefix="csv_stft", extension="png")
        analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, n_mels, max_height, save_path)


def main():
    sample_rate = int(2e6) #int(input("Sample rate: "))
    n_fft = 8192 #int(input("FFT size (n_fft): "))
    hop_length = 128 #int(input("Hop length: "))
    n_mels = 256 #int(input("Number of Mel bands: "))
    max_height = 28000  #int(input("Max frequency height: "))

    process_csv_file(sample_rate, n_fft, hop_length, n_mels, max_height, channel='CH1V')

    #process_wav_file(sample_rate, n_fft, hop_length, n_mels, max_height)


if __name__ == "__main__":
    main()
