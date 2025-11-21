import numpy as np
import matplotlib.pyplot as plt


def cwt_plot_scalogram(coefficients, frequencies, audio_data, sample_rate,
                       wavelet, scales, max_len, save_path=None, cmap='jet', vmin=-80,
                       scale_min=1, scale_max=128, scale_count=256,
                       filter_cutoff_freq=None, filter_order=5):
    """
    绘制CWT频谱图（Scalogram）

    参数:
        coefficients (np.ndarray): CWT系数矩阵
        frequencies (np.ndarray): 频率数组
        audio_data (np.ndarray): 原始音频数据
        sample_rate (int): 采样率
        wavelet (str): 使用的小波基函数
        scales (np.ndarray): 尺度数组
        max_len (int): 最大显示频率
        save_path (str): 保存路径，如果为None则不保存
        cmap (str): 颜色映射方案，默认'jet'
        vmin (float): 颜色映射的最小值（dB），默认-80
        scale_min (int): 最小尺度值
        scale_max (int): 最大尺度值
        scale_count (int): 尺度数量
        filter_cutoff_freq (float): 低通滤波器截止频率 (Hz)
        filter_order (int): 低通滤波器阶数
    """
    plt.figure(figsize=(22, 18), dpi=400)

    # 计算功率谱并转换为dB刻度
    power = np.abs(coefficients) ** 2
    power_db = 10 * np.log10(power + 1e-12)  # 添加小常数避免log(0)
    
    # 归一化到0dB最大值
    power_db = power_db - np.max(power_db)

    # 计算时间轴
    duration = len(audio_data) / sample_rate
    time = np.linspace(0, duration, coefficients.shape[1])

    # 绘制频谱图
    img = plt.pcolormesh(
        time,
        frequencies,
        power_db,
        cmap=cmap,
        vmin=vmin,
        vmax=0,
        shading='auto'
    )

    # 调整colorbar(图例栏)
    cbar = plt.colorbar(img, format='%+2.0f dB')
    cbar.ax.tick_params(labelsize=24)

    plt.xlabel('Time(s)', fontsize=36, labelpad=16)
    plt.ylabel('Frequency(Hz)', fontsize=36, labelpad=16)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)

    # 简化标题
    plt.title("CWT频谱图", fontsize=40, pad=20)
    plt.ylim(0, max_len)  # 限制显示频率范围

    # 在图形底部添加参数说明（白色背景，无边框）
    filter_text = f'  |  Filter: {filter_cutoff_freq} Hz (Order {filter_order})' if filter_cutoff_freq else ''
    param_text = f'Sample Rate = {sample_rate} Hz  |  Wavelet = {wavelet}  |  Scales = {scale_count} ({scale_min}-{scale_max}){filter_text}'
    plt.figtext(0.5, 0.015, param_text,
                ha='center', fontsize=28,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=5))

    plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为底部文字留出更多空间

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()
