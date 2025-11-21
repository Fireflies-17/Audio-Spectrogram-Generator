from func.input_func.process import process_csv_file, process_wav_file


def main():
    transform_method = 'cwt'  # 'stft' 或 'cwt'
    library = 'librosa'  # 'librosa' 或 'scipy'

    sample_rate = int(5e6)  # 采样率 (Hz)
    max_height = 4000  # 最大显示频率 (Hz)
    vmin = -60  # 颜色映射的最小值（dB），控制频谱图的动态范围

    n_fft = 32768 * 4  # FFT窗口大小
    win_length = 32768 * 4  # 窗口长度
    hop_length = win_length // 8  # 跳跃长度（帧移）
    window = 'hann'  # 窗口函数类型: 'hann', 'hamming', 'blackman', etc.
    n_mels = 256  # Mel频带数量（用于Mel频谱图）

    wavelet = 'morl'  # 小波基函数: 'morl'(Morlet), 'mexh'(墨西哥帽), 'cgau1'-'cgau8'(复高斯), 'cmor'(复Morlet)
    scale_min = 1  # 最小尺度值
    scale_max = 100000  # 最大尺度值
    scale_count = 256  # 尺度数量，影响频率分辨率

    filter_cutoff_freq = 20000  # 截止频率 (Hz)，设置为None表示不使用滤波
    filter_order = 4  # 滤波器阶数

    channel = 'CH1V'  # 通道选择: 'CH1V' 或 'CH2V'
    demodulated = True  # 是否进行希尔伯特解调

    process_csv_file(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, n_mels=n_mels, max_height=max_height,
        channel=channel, demodulated=demodulated, vmin=vmin,
        filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order,
        library=library, transform_method=transform_method,
        wavelet=wavelet, scale_min=scale_min, scale_max=scale_max, scale_count=scale_count
    )

    '''
    process_wav_file(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window, n_mels=n_mels, max_height=max_height,
        channel=channel, demodulated=demodulated, vmin=vmin,
        filter_cutoff_freq=filter_cutoff_freq, filter_order=filter_order,
        library=library, transform_method=transform_method,
        wavelet=wavelet, scale_min=scale_min, scale_max=scale_max, scale_count=scale_count
    )'''


if __name__ == "__main__":
    main()
