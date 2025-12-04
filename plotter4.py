import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# 假设这些是你自定义的模块，保持引用
from PartitalNN import PartialNN
from NN import Simple_NN
from Mie import Aerogel_Sample
from RTE_Truth_Model import RTE

# 设置保存路径
SAVE_DIR = 'plot_data'
SAVE_FILE = os.path.join(SAVE_DIR, 'real_pred.npy')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_and_save_data():
    """
    任务 1: 计算数据并保存
    包含：RTE (Ground Truth) 计算循环 和 ML 推理循环
    """
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    print("开始计算数据...")
    
    # 1. 初始化模型
    simple_model = Simple_NN(num_features=2).to(DEVICE)
    
    simple_model.load_state_dict(torch.load(
        'forward_models/epoch_380_.pth', 
        map_location=DEVICE
    ))
    simple_model.eval()

    # 2. 初始化气凝胶样本 (Sample C in Zhao's Paper)
    # 注意：build() 中通常会用到 wavelength，所以循环中需要更新
    aero = Aerogel_Sample(thickness_mm=5.26, density=293, optical_mean_r_nm=3.50, wavelength_nm=500, m0=1 - 2j)
    
    # 3. 定义波长范围
    wavelst_nm = np.linspace(200, 1000, 100)
    
    # 4. 初始化结果数组
    T_total_rte = np.zeros_like(wavelst_nm)
    T_diffuse_rte = np.zeros_like(wavelst_nm)
    T_total_ml = np.zeros_like(wavelst_nm)
    T_diffuse_ml = np.zeros_like(wavelst_nm)

    # 5. 循环计算
    for i, wl in enumerate(wavelst_nm):
        # --- 更新物理参数 ---
        aero.wavelength_nm = wl
        aero.build() 
        aero.build_m()
        xmax, omega, g = aero.to_opt_set()

        # --- A. RTE Solver (Ground Truth) ---
        # 实例化 RTE 类进行计算
        rte_obj = RTE(omega, 1, 0.99, xmax, 21, 21, phase_type='iso', g=g)
        rte_obj.build()
        T_rte, _ = rte_obj.hemi_props()
        
        T_total_rte[i] = T_rte
        T_diffuse_rte[i] = T_rte - np.exp(-xmax) # Diffuse = Total - Direct

        # --- B. ML Solver (Prediction) ---
        input_tensor = torch.tensor([[np.log(xmax), omega]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            T_R_vec = simple_model(input_tensor).cpu().numpy().flatten()
        
        T_total_ml[i] = T_R_vec[0]
        T_diffuse_ml[i] = T_R_vec[0] - np.exp(-xmax)

        if i % 10 == 0:
            print(f"Processing wavelength {wl:.1f} nm...")

    # 6. 打包并保存数据
    # 格式: [波长, T_RTE, T_diff_RTE, T_ML, T_diff_ML]
    datas = np.vstack([wavelst_nm, T_total_rte, T_diffuse_rte, T_total_ml, T_diffuse_ml])
    np.save(SAVE_FILE, datas)
    print(f"数据已保存至 {SAVE_FILE}")


def load_and_plot_data():
    """
    任务 2: 加载数据并画图
    """
    if not os.path.exists(SAVE_FILE):
        print(f"错误: 文件 {SAVE_FILE} 不存在，请先运行 compute_and_save_data()")
        return

    print(f"加载数据从 {SAVE_FILE}...")
    datas = np.load(SAVE_FILE)

    # 解析数据
    wavelst_nm = datas[0]
    T_total_rte = datas[1]
    T_diffuse_rte = datas[2]
    T_total_ml = datas[3]
    T_diffuse_ml = datas[4]

    # --- 数据切片 (可选) ---
    # 如果点太密，画散点图不好看，可以进行切片（例如原代码中的操作）
    # 这里我们画图时对散点进行切片，对曲线保持全精度
    # step = 5 表示每隔5个点画一个圈
    step = 4 
    
    lw = 2
    ms = 8

    plt.figure(figsize=(8, 6))

    # 1. Total Transmittance
    # RTE (Line)
    plt.plot(wavelst_nm, T_total_rte, '--', label='$T_{total}$ RTE', lw=lw, color='#4a90e2')
    # ML (Scatter points, thinned out)
    plt.plot(wavelst_nm[::step], T_total_ml[::step], 'o', label='$T_{total}$ ML', ms=ms, 
             color='#1f77b4', markerfacecolor='none', markeredgecolor='#1f77b4')

    # 2. Diffuse Transmittance
    # RTE (Line)
    plt.plot(wavelst_nm, T_diffuse_rte, '--', label='$T_{diffuse}$ RTE', lw=lw, color='#ffad66')
    # ML (Scatter points, thinned out)
    plt.plot(wavelst_nm[::step], T_diffuse_ml[::step], 'o', label='$T_{diffuse}$ ML', ms=ms, 
             color='#ff7f0e', markerfacecolor='none', markeredgecolor='#ff7f0e')

    plt.legend(loc='right')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance (T)')
    plt.title('T ~ Wavelength for Existing Sample', fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# --- 使用示例 ---
if __name__ == "__main__":
    compute_and_save_data()

    load_and_plot_data()