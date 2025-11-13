from func.input_func.process import process_csv_file, process_wav_file


def main():
    sample_rate = int(5e6) #int(input("Sample rate: "))
    n_fft = 100000 #int(input("FFT size (n_fft): "))
    win_length = 100000 #int(input("Window length: "))
    hop_length = win_length // 2 #int(input("Hop length: "))
    window = 'hann' #input("Window type: ")
    n_mels = 256 #int(input("Number of Mel bands: "))
    max_height = 3500  #int(input("Max frequency height: "))
    vmin = -45  # 颜色映射的最小值（dB），控制频谱图的动态范围

    process_csv_file(sample_rate, n_fft, hop_length, win_length, window, n_mels, max_height, 
                     channel='CH1V', demodulated=True, vmin=vmin)

    #process_wav_file(sample_rate, n_fft, hop_length, win_length, window, n_mels, max_height)


if __name__ == "__main__":
    main()
