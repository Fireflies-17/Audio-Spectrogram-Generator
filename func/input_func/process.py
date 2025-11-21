from func.analysis_func.demodulate import demodulate_hilbert
from func.analysis_func.stft_librosa import analyze_audio_with_stft_librosa
from func.analysis_func.stft_scipy import analyze_audio_with_stft_scipy
from func.analysis_func.cwt_pywavelets import analyze_audio_with_cwt_pywt
from func.input_func.csv_input import load_data_from_csv, load_data_from_csv_simple
from func.input_func.wav_input import load_audio_from_file
from func.output_func.path import generate_output_path, export_to_wav


def process_csv_file(sample_rate, n_fft, hop_length, win_length, window, n_mels,
                     max_height, channel='CH1V', demodulated=False, vmin=-80,
                     filter_cutoff_freq=None, filter_order=5,
                     library='librosa', transform_method='stft',
                     wavelet='morl', scale_min=1, scale_max=128, scale_count=256):
    """
    处理CSV格式的数据文件

    参数:
        sample_rate (int): 采样率，如果为None则从文件头读取
        n_fft (int): FFT窗口大小（仅用于STFT）
        hop_length (int): 跳跃长度（仅用于STFT）
        win_length (int): 窗口长度（仅用于STFT）
        window (str): 窗口函数类型（仅用于STFT）
        n_mels (int): Mel频带数量（仅用于STFT）
        max_height (int): 最大频率高度

        channel (str): 要处理的通道，'CH1V'或'CH2V'
        demodulated (bool): 是否对指定通道执行解调操作，默认False
        vmin (float): 颜色映射的最小值（dB），默认-80
        filter_cutoff_freq (float): 低通滤波器截止频率 (Hz)，默认None表示不使用滤波
        filter_order (int): 低通滤波器阶数，默认5

        library (str): STFT实现库选择，'librosa'或'scipy'，默认'librosa'（仅用于STFT）
        transform_method (str): 变换方法选择，'stft'或'cwt'，默认'stft'

        wavelet (str): 小波基函数（仅用于CWT），默认'morl'
        scale_min (int): 最小尺度值（仅用于CWT），默认1
        scale_max (int): 最大尺度值（仅用于CWT），默认128
        scale_count (int): 尺度数量（仅用于CWT），默认256
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

            if transform_method == 'cwt':
                save_path = generate_output_path(
                    prefix=f"demodulated_{channel}_cwt", extension="png"
                )
            else:
                save_path = generate_output_path(
                    prefix=f"demodulated_{channel}_stft", extension="png"
                )

        else:
            if transform_method == 'cwt':
                save_path = generate_output_path(prefix=f"csv_{channel}_cwt", extension="png")
            else:
                save_path = generate_output_path(prefix=f"csv_{channel}_stft", extension="png")

        # 根据transform_method选择变换方法
        if transform_method == 'cwt':
            print(f"\nUsing PyWavelets for CWT analysis...")
            analyze_audio_with_cwt_pywt(
                audio_data, sample_rate, scales=None, wavelet=wavelet,
                max_len=max_height, save_path=save_path, vmin=vmin,
                filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order,
                scale_min=scale_min, scale_max=scale_max, scale_count=scale_count
            )
        elif transform_method == 'stft':
            # 根据library选择对应的STFT实现
            if library == 'scipy':
                print(f"\nUsing scipy.signal.ShortTimeFFT for STFT analysis...")
                analyze_audio_with_stft_scipy(
                    audio_data, sample_rate, n_fft, hop_length, win_length,
                    max_height, window=window, save_path=save_path, vmin=vmin,
                    filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order
                )
            elif library == 'librosa':
                print(f"\nUsing librosa for STFT analysis...")
                analyze_audio_with_stft_librosa(
                    audio_data, sample_rate, n_fft, hop_length, win_length,
                    n_mels, max_height, window=window, save_path=save_path, vmin=vmin,
                    filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order
                )


def process_wav_file(sample_rate, n_fft, hop_length, win_length, window, n_mels,
                     max_height, vmin=-80,
                     filter_cutoff_freq=None, filter_order=5,
                     library='librosa', transform_method='stft',
                     wavelet='morl', scale_min=1, scale_max=128, scale_count=256):
    """
    处理WAV格式的音频文件

    参数:
        sample_rate (int): 采样率
        n_fft (int): FFT窗口大小（仅用于STFT）
        hop_length (int): 跳跃长度（仅用于STFT）
        win_length (int): 窗口长度（仅用于STFT）
        window (str): 窗口函数类型（仅用于STFT）
        n_mels (int): Mel频带数量（仅用于STFT）
        max_height (int): 最大频率高度

        vmin (float): 颜色映射的最小值（dB），默认-80
        filter_cutoff_freq (float): 低通滤波器截止频率 (Hz)，默认None表示不使用滤波
        filter_order (int): 低通滤波器阶数，默认5

        library (str): STFT实现库选择，'librosa'或'scipy'，默认'librosa'（仅用于STFT）
        transform_method (str): 变换方法选择，'stft'或'cwt'，默认'stft'

        wavelet (str): 小波基函数（仅用于CWT），默认'morl'
        scale_min (int): 最小尺度值（仅用于CWT），默认1
        scale_max (int): 最大尺度值（仅用于CWT），默认128
        scale_count (int): 尺度数量（仅用于CWT），默认256
    """
    file_path = ''
    audio_data, sample_rate = load_audio_from_file(file_path, sample_rate)

    if audio_data is not None:
        # 自动生成输出路径
        if transform_method == 'cwt':
            save_path = generate_output_path(prefix="wav_cwt", extension="png")
        else:
            save_path = generate_output_path(prefix="wav_stft", extension="png")
        
        # 根据transform_method选择变换方法
        if transform_method == 'cwt':
            print(f"\nUsing PyWavelets for CWT analysis...")
            analyze_audio_with_cwt_pywt(
                audio_data, sample_rate, scales=None, wavelet=wavelet,
                max_len=max_height, save_path=save_path, vmin=vmin,
                filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order,
                scale_min=scale_min, scale_max=scale_max, scale_count=scale_count
            )
        elif transform_method == 'stft':
            # 根据library选择对应的STFT实现
            if library == 'scipy':
                print(f"\nUsing scipy.signal.ShortTimeFFT for STFT analysis...")
                analyze_audio_with_stft_scipy(
                    audio_data, sample_rate, n_fft, hop_length, win_length, max_height, 
                    window=window, save_path=save_path, vmin=vmin,
                    filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order
                )
            elif library == 'librosa':
                print(f"\nUsing librosa for STFT analysis...")
                analyze_audio_with_stft_librosa(
                    audio_data, sample_rate, n_fft, hop_length, win_length, n_mels, max_height, 
                    window=window, save_path=save_path, vmin=vmin,
                    filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order
                )