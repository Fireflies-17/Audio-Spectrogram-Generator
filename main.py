from func.input_func.process import process_csv_file, process_wav_file


def main():
    sample_rate = int(2e6) #int(input("Sample rate: "))
    n_fft = 1024 #int(input("FFT size (n_fft): "))
    hop_length = 512 #int(input("Hop length: "))
    n_mels = 256 #int(input("Number of Mel bands: "))
    max_height = 3000  #int(input("Max frequency height: "))

    process_csv_file(sample_rate, n_fft, hop_length, n_mels, max_height, 
                     channel='CH1V', demodulated=False)

    #process_wav_file(sample_rate, n_fft, hop_length, n_mels, max_height)


if __name__ == "__main__":
    main()
