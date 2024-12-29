import matplotlib.pyplot as plt
import numpy as np


def plot_custom_control_chart(mean, std_dev, samples, sample_size=5):
    """
    绘制自定义控制图，样本点之间连线，并标注异常点。

    :param mean: 给定的过程均值
    :param std_dev: 给定的过程标准差
    :param samples: 实际样本值列表
    :param sample_size: 每个样本的测量值数量
    """
    # 计算控制限
    z = 3  # 对应于99.73%的置信水平
    UCL = mean + z * std_dev / np.sqrt(sample_size)
    LCL = mean - z * std_dev / np.sqrt(sample_size)

    # 绘制样本点并连线
    sample_indices = np.arange(1, len(samples) + 1)
    plt.plot(sample_indices, samples, marker='o', linestyle='-', color='blue', label='Sample Points')

    # 绘制中心线(CL)、上控制限(UCL)和下控制限(LCL)
    plt.axhline(mean, color='green', linestyle='-', label='CL')
    plt.axhline(UCL, color='red', linestyle='--', label='UCL')
    plt.axhline(LCL, color='red', linestyle='--', label='LCL')

    # 异常点识别与标注
    anomalies = [i for i, sample in enumerate(samples, start=1) if sample > UCL or sample < LCL]
    if anomalies:
        print("异常点索引：", anomalies)
        plt.plot(anomalies, [samples[i - 1] for i in anomalies], marker='o', linestyle='none', color='magenta',
                 label='Anomalies')

    plt.xlabel('Sample Number')
    plt.ylabel('Value')
    plt.title('Custom Control Chart with Anomalies')

    # plt.legend()
    # plt.show()


