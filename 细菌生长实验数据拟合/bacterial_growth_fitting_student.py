import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def load_bacterial_data(file_path):
    """
    从文件中加载细菌生长实验数据

    参数:
        file_path (str): 数据文件路径

    返回:
        tuple: 包含时间和酶活性测量值的元组
    """
    # TODO: 实现数据加载功能 (大约3行代码)
    # [STUDENT_CODE_HERE]
    data = np.loadtxt(file_path, delimiter=',')
    t = data[:, 0]  # 时间
    activity = data[:, 1]  # 酶活性
    return t, activity


def V_model(t, tau):
    """
    V(t)模型函数

    参数:
        t (float or numpy.ndarray): 时间
        tau (float): 时间常数

    返回:
        float or numpy.ndarray: V(t)模型值
    """
    # TODO: 根据V(t) = 1 - e^(-t/τ)实现模型函数 (1行代码)
    # [STUDENT_CODE_HERE]
    return 1-np.exp(-t/tau)


def W_model(t, A, tau):
    """
    W(t)模型函数

    参数:
        t (float or numpy.ndarray): 时间
        A (float): 比例系数
        tau (float): 时间常数

    返回:
        float or numpy.ndarray: W(t)模型值
    """
    # TODO: 根据W(t) = A(e^(-t/τ) - 1 + t/τ)实现模型函数 (1行代码)
    # [STUDENT_CODE_HERE]
    return A*(np.exp(-t/tau)-1+t/tau)


def fit_model(t, data, model_func, p0):
    """
    使用curve_fit拟合模型

    参数:
        t (numpy.ndarray): 时间数据
        data (numpy.ndarray): 实验数据
        model_func (function): 模型函数
        p0 (list): 初始参数猜测

    返回:
        tuple: 拟合参数及其协方差矩阵
    """
    # TODO: 使用scipy.optimize.curve_fit进行拟合 (1行代码)
    # [STUDENT_CODE_HERE]
    popt, pcov = curve_fit(model_func, t, data, p0=p0)
    return popt, pcov


def plot_results(t, data, model_func, popt, title):
    """
    绘制实验数据与拟合曲线

    参数:
        t (numpy.ndarray): 时间数据
        data (numpy.ndarray): 实验数据
        model_func (function): 模型函数
        popt (numpy.ndarray): 拟合参数
        title (str): 图表标题
    """
    # TODO: 实现绘图功能 (约10行代码)
    # [STUDENT_CODE_HERE]
    plt.figure(figsize=(10, 6))

    # 绘制原始数据
    plt.scatter(t, data, color='b', label='Experimental data', alpha=0.7)

    # 生成拟合曲线
    t_fit = np.linspace(t.min(), t.max(), 1000)  # 生成密集的时间点用于绘制平滑曲线
    y_fit = model_func(t_fit, *popt)  # 使用拟合参数计算模型值

    # 绘制拟合曲线
    plt.plot(t_fit, y_fit, 'r-', linewidth=2, label='Model fit')

    # 添加标签和标题
    plt.xlabel('Time(s)')
    plt.ylabel('Activity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # 加载数据
    data_dir = "D:\Download"  # 请替换为你的数据目录
    t_V, V_data = load_bacterial_data(f"{data_dir}/g149novickA.txt")
    t_W, W_data = load_bacterial_data(f"{data_dir}/g149novickB.txt")

    # 拟合V(t)模型
    popt_V, pcov_V = fit_model(t_V, V_data, V_model, p0=[1.0])
    print(f"V(t)模型拟合参数: τ = {popt_V[0]:.3f}")

    # 拟合W(t)模型
    popt_W, pcov_W = fit_model(t_W, W_data, W_model, p0=[1.0, 1.0])
    print(f"W(t)模型拟合参数: A = {popt_W[0]:.3f}, τ = {popt_W[1]:.3f}")

    # 绘制结果
    plot_results(t_V, V_data, V_model, popt_V, 'V(t) Model Fit')
    plot_results(t_W, W_data, W_model, popt_W, 'W(t) Model Fit')
