import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_supernova_data(file_path):
    """
    从文件中加载超新星数据
    
    参数:
        file_path (str): 数据文件路径
        
    返回:
        tuple: 包含以下元素的元组
            - z (numpy.ndarray): 红移数据
            - mu (numpy.ndarray): 距离模数数据
            - mu_err (numpy.ndarray): 距离模数误差
    """
    # 加载CSV文件
    data = np.loadtxt(file_path, delimiter='\t', skiprows=6)
    
    # 提取红移、距离模数mu和标准差
    z = data[:, 0]      
    mu = data[:, 1]      
    mu_err = data[:, 2] 
    
    return z, mu, mu_err


def hubble_model(z, H0):
    """
    哈勃模型：距离模数与红移的关系
    
    参数:
        z (float or numpy.ndarray): 红移
        H0 (float): 哈勃常数 (km/s/Mpc)
        
    返回:
        float or numpy.ndarray: 距离模数
    """
    c = 299792.458
    
    # 距离模数 μ = 5*log10(c*z/H0) + 25，设减速参数q0=0（或a1=1）
    mu = 5 * np.log10(c * z / H0) + 25
    
    return mu

def hubble_model_with_deceleration(z, H0, a1):
    """
    包含减速参数的哈勃模型
    
    参数:
        z (float or numpy.ndarray): 红移
        H0 (float): 哈勃常数 (km/s/Mpc)
        a1 (float): 拟合参数，对应于减速参数q0
        
    返回:
        float or numpy.ndarray: 距离模数
    """
    c = 299792.458
    
    # 包含减速参数的距离模数
    # μ = 5*log10(c*z/H0 * (1 + 0.5*(1-a1)*z)) + 25
    mu = 5 * np.log10(c * z / H0 * (1 + 0.5 * (1 - a1) * z)) + 25
    
    return mu


def hubble_fit(z, mu, mu_err):
    """
    使用最小二乘法拟合哈勃常数
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        
    返回:
        tuple: 包含以下元素的元组
            - H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
            - H0_err (float): 哈勃常数的误差
    """
    # 哈勃常数初始值
    H0_guess = 70.0 
    
    # absolute_sigma=True即使用绝对误差而非相对
    popt, pcov = curve_fit(hubble_model, z, mu, p0=[H0_guess], sigma=mu_err, absolute_sigma=True)
    
    H0 = popt[0]  # 拟合参数
    H0_err = np.sqrt(pcov[0, 0])  # 参数标准差
    
    return H0, H0_err

def hubble_fit_with_deceleration(z, mu, mu_err):
    """
    使用最小二乘法拟合哈勃常数和减速参数
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        
    返回:
        tuple: 包含以下元素的元组
            - H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
            - H0_err (float): 哈勃常数的误差
            - a1 (float): 拟合得到的a1参数
            - a1_err (float): a1参数的误差
    """

    H0_guess = 70.0 
    a1_guess = 1.0   # 对应于q0=0
    
    popt, pcov = curve_fit(hubble_model_with_deceleration, z, mu, 
                          p0=[H0_guess, a1_guess], sigma=mu_err, absolute_sigma=True)
    
    # 从结果中提取参数及误差
    H0 = popt[0]
    a1 = popt[1]
    H0_err = np.sqrt(pcov[0, 0])
    a1_err = np.sqrt(pcov[1, 1])
    
    return H0, H0_err, a1, a1_err


def plot_hubble_diagram(z, mu, mu_err, H0):
    """
    绘制哈勃图（距离模数vs红移）
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', color='blue', markersize=5, 
                 ecolor='gray', elinewidth=1, capsize=2, label='Supernova data')
    #绘制拟合曲线
    z_fit = np.linspace(min(z), max(z), 1000)
    mu_fit = hubble_model(z_fit, H0)
    plt.plot(z_fit, mu_fit, '-', color='red', linewidth=2, 
             label=f'Best fit: $H_0$ = {H0:.1f} km/s/Mpc')
    
    plt.xlabel('Redshift z')
    plt.ylabel('Distance modulus μ')
    plt.title('Hubble Diagram')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

def plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1):
    """
    绘制包含减速参数的哈勃图
    
    参数:
        z (numpy.ndarray): 红移数据
        mu (numpy.ndarray): 距离模数数据
        mu_err (numpy.ndarray): 距离模数误差
        H0 (float): 拟合得到的哈勃常数 (km/s/Mpc)
        a1 (float): 拟合得到的a1参数
        
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    plt.figure(figsize=(10, 6))
    plt.errorbar(z, mu, yerr=mu_err, fmt='o', color='blue', markersize=5, 
                 ecolor='gray', elinewidth=1, capsize=2, label='Supernova data')
    z_fit = np.linspace(min(z), max(z), 1000)
    mu_fit = hubble_model_with_deceleration(z_fit, H0, a1)
    plt.plot(z_fit, mu_fit, '-', color='red', linewidth=2, 
             label=f'Best fit: $H_0$ = {H0:.1f} km/s/Mpc, $a_1$ = {a1:.2f}')
    plt.xlabel('Redshift z')
    plt.ylabel('Distance modulus μ')
    plt.title('Hubble Diagram with Deceleration Parameter')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()


if __name__ == "__main__":
    data_file = "/Users/lixh/Library/CloudStorage/OneDrive-个人/Code/cp2025-InterpolateFit-1/Supernova_Hubble_Fitting/data/supernova_data.txt"
    
    # 读取数据
    z, mu, mu_err = load_supernova_data(data_file)
    
    # 拟合H0
    H0, H0_err = hubble_fit(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    
    # 绘图
    fig = plot_hubble_diagram(z, mu, mu_err, H0)
    plt.show()
    H0, H0_err, a1, a1_err = hubble_fit_with_deceleration(z, mu, mu_err)
    print(f"拟合得到的哈勃常数: H0 = {H0:.2f} ± {H0_err:.2f} km/s/Mpc")
    print(f"拟合得到的a1参数: a1 = {a1:.2f} ± {a1_err:.2f}")
     # 绘制包含减速参数的图
    fig = plot_hubble_diagram_with_deceleration(z, mu, mu_err, H0, a1)
    plt.show()
