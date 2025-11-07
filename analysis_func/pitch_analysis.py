import numpy as np
import librosa


def analyze_fundamental_frequency(audio_data, sample_rate, fmin=50, fmax=2000, frame_length=2048, hop_length=512):
    """
    分析音频信号的基频（Fundamental Frequency）范围
    
    参数:
        audio_data (np.ndarray): 音频时域信号
        sample_rate (int): 采样率
        fmin (float): 最小检测频率（Hz），默认50Hz
        fmax (float): 最大检测频率（Hz），默认2000Hz
        frame_length (int): 帧长度，默认2048
        hop_length (int): 帧移，默认512
        
    返回:
        dict: 包含基频分析结果的字典
    """
    print("\n" + "="*60)
    print("Fundamental Frequency Analysis")
    print("="*60)
    
    # 根据采样率自动调整参数以避免警告
    # 确保至少有2个周期的fmin能够放入帧中，使用ceil确保足够的余量
    min_frame_length = int(np.ceil(2.0 * sample_rate / fmin)) + 2  # +2 提供额外余量
    if frame_length < min_frame_length:
        frame_length = min_frame_length
        hop_length = frame_length // 4  # 保持合理的帧移比例
        print(f"Auto-adjusted: frame_length={frame_length}, hop_length={hop_length}")
    
    try:
        # 使用librosa的pyin算法进行基频估计
        # pyin是一个概率YIN算法，对噪声更鲁棒
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio_data,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # 过滤掉未检测到基频的帧（NaN值）
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) == 0:
            print("Warning: No fundamental frequency detected!")
            print("Possible reasons:")
            print("  - Signal might be too noisy")
            print("  - Signal might not contain periodic components")
            print("  - Try adjusting fmin and fmax parameters")
            return None
        
        # 计算统计信息
        f0_min = np.min(valid_f0)
        f0_max = np.max(valid_f0)
        f0_mean = np.mean(valid_f0)
        f0_median = np.median(valid_f0)
        f0_std = np.std(valid_f0)
        
        # 计算有声帧的百分比
        voiced_percentage = (len(valid_f0) / len(f0)) * 100
        
        # 输出结果
        print(f"\nSample Rate: {sample_rate} Hz")
        print(f"Signal Duration: {len(audio_data)/sample_rate:.2f} seconds")
        print(f"Total Frames: {len(f0)}")
        print(f"Voiced Frames: {len(valid_f0)} ({voiced_percentage:.1f}%)")
        print(f"\n{'Fundamental Frequency Range:':-^60}")
        print(f"  Minimum F0:    {f0_min:>10.2f} Hz")
        print(f"  Maximum F0:    {f0_max:>10.2f} Hz")
        print(f"  Mean F0:       {f0_mean:>10.2f} Hz")
        print(f"  Range:         {f0_min:.2f} - {f0_max:.2f} Hz")
        print("="*60 + "\n")
        
        # 返回结果字典
        result = {
            'f0_values': f0,
            'valid_f0': valid_f0,
            'f0_min': f0_min,
            'f0_max': f0_max,
            'f0_mean': f0_mean,
            'f0_median': f0_median,
            'f0_std': f0_std,
            'voiced_percentage': voiced_percentage,
            'sample_rate': sample_rate,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs
        }
        
        return result
        
    except Exception as e:
        print(f"Error during pitch analysis: {e}")
        return None


def get_frequency_range_description(f0_min, f0_max):
    """
    根据基频范围给出描述性文字
    
    参数:
        f0_min (float): 最小基频
        f0_max (float): 最大基频
        
    返回:
        str: 描述性文字
    """
    descriptions = []
    
    # 判断频率范围所属的音域
    if f0_min < 85:
        descriptions.append("Contains very low frequencies (Sub-bass range)")
    if 85 <= f0_min <= 250 or (f0_max >= 85 and f0_max <= 250):
        descriptions.append("Contains bass frequencies (Male voice / Low instruments)")
    if 250 <= f0_min <= 500 or (f0_max >= 250 and f0_max <= 500):
        descriptions.append("Contains mid-range frequencies (Female voice / Mid instruments)")
    if f0_max > 500:
        descriptions.append("Contains high frequencies (High voice / High-pitched sounds)")
    
    return " | ".join(descriptions) if descriptions else "Frequency range analysis"
