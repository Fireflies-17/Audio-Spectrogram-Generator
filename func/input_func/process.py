from func.analysis_func.demodulate import demodulate_hilbert
from func.analysis_func.stft import analyze_audio_with_stft
from func.input_func.csv_input import load_data_from_csv, load_data_from_csv_simple
from func.input_func.wav_input import load_audio_from_file
from func.output_func.path import generate_output_path, export_to_wav


def process_csv_file(sample_rate, n_fft, hop_length, win_length, window, n_mels, max_height, 
                     channel='CH1V', demodulated=False, vmin=-80, filter_cutoff_freq=None, filter_order=5):
    """
    处理CSV格式的数据文件

    参数:
        sample_rate (int): 采样率，如果为None则从文件头读取
        n_fft (int): FFT窗口大小
        hop_length (int): 跳跃长度
        win_length (int): 窗口长度
        window (str): 窗口函数类型
        n_mels (int): Mel频带数量
        max_height (int): 最大频率高度
        channel (str): 要处理的通道，'CH1V'或'CH2V'
        demodulated (bool): 是否对指定通道执行解调操作，默认False
        vmin (float): 颜色映射的最小值（dB），默认-80
        filter_cutoff_freq (float): 低通滤波器截止频率 (Hz)，默认None表示不使用滤波
        filter_order (int): 低通滤波器阶数，默认5
    """
    file_path = 'data/input_data/fs5e6_tswp500ms_t2s_demo.csv' #input("Path: ")

    # 加载指定通道数据
    audio_data, sample_rate = load_data_from_csv_simple(file_path, sample_rate)
    
    if audio_data is not None:
        if demodulated:
            print(f"Demodulating signal...")
            audio_data = demodulate_hilbert(audio_data)
            
            # 导出解调后的音频为 WAV 文件
            #print("\nExporting demodulated signal to WAV...")
            #export_to_wav(audio_data, sample_rate, prefix=f"demodulated_{channel}")

            save_path = generate_output_path(
                prefix=f"demodulated_{channel}_stft", extension="png"
            )

        else:
            save_path = generate_output_path(prefix=f"csv_{channel}_stft", extension="png")

        analyze_audio_with_stft(
            audio_data, sample_rate, n_fft, hop_length, win_length,
            n_mels, max_height, window=window, save_path=save_path, vmin=vmin,
            filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order
        )


def process_wav_file(sample_rate, n_fft, hop_length, win_length, window, n_mels, max_height, vmin=-80, 
                     filter_cutoff_freq=None, filter_order=5):
    """
    处理WAV格式的音频文件

    参数:
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小
        hop_length (int): 跳跃长度
        win_length (int): 窗口长度
        window (str): 窗口函数类型
        n_mels (int): Mel频带数量
        max_height (int): 最大频率高度
        vmin (float): 颜色映射的最小值（dB），默认-80
        filter_cutoff_freq (float): 低通滤波器截止频率 (Hz)，默认None表示不使用滤波
        filter_order (int): 低通滤波器阶数，默认5
    """
    file_path = ''
    audio_data, sample_rate = load_audio_from_file(file_path, sample_rate)

    if audio_data is not None:
        # 自动生成输出路径
        save_path = generate_output_path(prefix="wav_stft", extension="png")
        analyze_audio_with_stft(audio_data, sample_rate, n_fft, hop_length, win_length, n_mels, max_height, 
                               window=window, save_path=save_path, vmin=vmin,
                               filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order)