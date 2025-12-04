import numpy as np
import torch
import time
from scipy.optimize import minimize
from RTE_Truth_Model import RTE
from Mie import Aerogel_Sample
from PartitalNN import PartialNN

# --- Configuration ---
INVERSE_MODEL_PATH = r'models/partial_inverse.pth'  # 请确认路径
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running Benchmark on: {DEVICE}")

def benchmark_inverse_full_run():
    # 1. Setup Data (Sigmoid Spectrum, 50 points)
    waves = np.linspace(300, 1000, 50)
    target_T = 1 / (1 + np.exp(-(waves - 600) / 50))
    target_T = np.clip(target_T, 1e-3, 1 - 1e-3).astype(np.float32)
    
    # Pre-load to GPU (Exclude transfer time)
    target_T_tensor = torch.from_numpy(target_T).reshape(-1, 1).to(DEVICE)

    # 2. Load Model
    try:
        inv_model = PartialNN().to(DEVICE)
        inv_model.load_state_dict(torch.load(INVERSE_MODEL_PATH, map_location=DEVICE, weights_only=True))
        inv_model.eval()
    except Exception as e:
        print(f"Warning: Model load failed ({e}), using random init.")
        inv_model = PartialNN().to(DEVICE)
        inv_model.eval()

    num_repeats = 10
    results = {}

    print(f"\n========== Task 1: Optical Properties (T -> tau) [50 points, {num_repeats} repeats] ==========")
    
    # --- ML Approach (GPU) ---
    # Warm-up
    with torch.no_grad(): _ = inv_model(target_T_tensor)
    torch.cuda.synchronize()

    timings = []
    for _ in range(num_repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = inv_model(target_T_tensor) # 一次性推理所有50个点
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
    
    results['ml_opt_mean'] = np.mean(timings)
    results['ml_opt_std'] = np.std(timings)
    print(f"  ML (GPU): {np.mean(timings):.4f} ± {np.std(timings):.4f} ms")

    # --- NM Approach (CPU, Conventional) ---
    # 逻辑：对 50 个波长点，逐个运行 minimize 寻找 tau
    timings = []
    print("  NM (CPU): Running full optimization loop... (please wait)")
    
    for i in range(num_repeats):
        start = time.perf_counter()
        # 遍历所有 50 个点
        for idx in range(50):
            tgt_val = target_T[idx]
            def obj(tau):
                # 简单 RTE 计算 (iso 相函数)
                rte = RTE(0.9, 1, 0.99, tau[0], 21, 21, 'iso')
                rte.build()
                return (rte.hemi_props()[0] - tgt_val)**2
            
            # 不限制 maxiter，跑完全程
            minimize(obj, [1.0], method='Nelder-Mead', tol=1e-2)
            
        timings.append((time.perf_counter() - start) * 1000)
        # print(f"    Repeat {i+1}/{num_repeats} done.")

    results['nm_opt_mean'] = np.mean(timings)
    results['nm_opt_std'] = np.std(timings)
    print(f"  NM (CPU): {np.mean(timings):.4f} ± {np.std(timings):.4f} ms")


    print(f"\n========== Task 2: Material Properties (Spectrum -> rho, r) [Full RTE check, {num_repeats} repeats] ==========")

    # --- ML Approach (Hybrid: GPU Inference + CPU Mie Fit) ---
    timings = []
    for _ in range(num_repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # A. GPU Inference
        with torch.no_grad():
            pred_log_tau = inv_model(target_T_tensor).cpu().numpy().flatten()
        target_quasi_T = np.exp(-np.exp(pred_log_tau))
        
        # B. Mie Fit (Analytical, fast)
        def obj_mie(p):
            rho, r = p
            if rho < 10 or r < 0.1: return 1e6
            calc_quasi_T = []
            for wl in waves:
                aero = Aerogel_Sample(4.75, rho, optical_mean_r_nm=r, wavelength_nm=wl, m0=1.45-1e-4j)
                aero.build()
                x, w, _ = aero.to_opt_set()
                calc_quasi_T.append(np.exp(-aero.xmax))
            return np.mean((np.array(calc_quasi_T) - target_quasi_T)**2)

        minimize(obj_mie, [200, 5.0], method='Nelder-Mead', tol=1e-3, options={'maxiter': 20})
        
        timings.append((time.perf_counter() - start) * 1000)

    results['ml_mat_mean'] = np.mean(timings)
    results['ml_mat_std'] = np.std(timings)
    print(f"  ML (Hybrid): {np.mean(timings):.4f} ± {np.std(timings):.4f} ms")


    # --- NM Approach (Full RTE Loop) ---
    # 逻辑：直接优化 rho, r。Objective Function 内部对 50 个波长都跑 RTE。
    timings = []
    print("  NM (CPU): Running full RTE optimization... (this performs 50 RTE calls per iteration)")
    
    for i in range(num_repeats):
        start = time.perf_counter()
        
        def obj_full(p):
            rho, r = p
            if rho < 10 or r < 0.1: return 1e6
            loss = 0
            # 完整计算 50 个点
            for idx, wl in enumerate(waves):
                tgt = target_T[idx]
                aero = Aerogel_Sample(4.75, rho, optical_mean_r_nm=r, wavelength_nm=wl, m0=1.45-1e-4j)
                aero.build()
                x, w, _ = aero.to_opt_set()
                
                rte = RTE(w, 1, 0.99, x, 21, 21, 'iso')
                rte.build()
                loss += (rte.hemi_props()[0] - tgt)**2
            return loss

        # 不限制 maxiter
        minimize(obj_full, [200, 5.0], method='Nelder-Mead', tol=1e-2, options={'maxiter': 20})
        
        timings.append((time.perf_counter() - start) * 1000)
        # print(f"    Repeat {i+1}/{num_repeats} done.")

    results['nm_mat_mean'] = np.mean(timings)
    results['nm_mat_std'] = np.std(timings)
    print(f"  NM (CPU): {np.mean(timings):.4f} ± {np.std(timings):.4f} ms")

    # Save
    np.save('benchmark_inverse_results_full.npy', results)
    print("\nSaved to benchmark_inverse_results_full.npy")

if __name__ == '__main__':
    benchmark_inverse_full_run()