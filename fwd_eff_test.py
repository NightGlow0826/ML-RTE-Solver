import numpy as np
import torch
import time
from tqdm import tqdm
from scipy.optimize import minimize

# 引入您项目中的模块
from RTE_Truth_Model import RTE
from NN import Simple_NN
from Mie import Aerogel_Sample, f_n
from PartitalNN import PartialNN

# --- Configuration ---
FORWARD_MODEL_PATH = r'forward_models/epoch_380_.pth' 
DEVICE_GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_CPU = torch.device('cpu')

def benchmark_forward_workloads():
    print("========== Starting Forward Solver Efficiency Test (Fig E) with STD ==========")
    
    # 1. Setup Models
    # CPU Model
    net_cpu = Simple_NN(num_features=2).to(DEVICE_CPU)
    try:
        net_cpu.load_state_dict(torch.load(FORWARD_MODEL_PATH, map_location=DEVICE_CPU, weights_only=True))
    except:
        print(f"Warning: Could not load model from {FORWARD_MODEL_PATH}, using random weights.")
    net_cpu.eval()
    
    # GPU Model
    if torch.cuda.is_available():
        net_gpu = Simple_NN(num_features=2).to(DEVICE_GPU)
        net_gpu.load_state_dict(torch.load(FORWARD_MODEL_PATH, map_location=DEVICE_GPU, weights_only=True))
      
        net_gpu.eval()


    # 2. Define Workloads
    workload_sizes = [10, 100, 1000, 2500,  10000, 25000, 100000, 1000000]
    num_repeats = 10  # 重复次数，用于计算 mean 和 std
    
    results = {
        'sizes': workload_sizes,
        'cpu_times': [], 'cpu_std': [],
        'gpu_times': [], 'gpu_std': [],
        'rte_times': [], 'rte_std': []
    }

    for size in workload_sizes:
        print(f"\nTesting Workload Size: {size}")
        
        # --- Prepare Data (Exclude from timing) ---
        x_np = np.random.uniform(0.1, 5.0, size).astype(np.float32) 
        omega_np = np.random.uniform(0.1, 0.9, size).astype(np.float32)
        inputs_np = np.stack([np.log(x_np), omega_np], axis=1)
        
        inputs_cpu = torch.from_numpy(inputs_np).to(DEVICE_CPU)
        if net_gpu:
            inputs_gpu = torch.from_numpy(inputs_np).to(DEVICE_GPU)
        
        # ===========================
        # 1. ML CPU Benchmark
        # ===========================
        # Warmup
        with torch.no_grad():
            _ = net_cpu(inputs_cpu)
            
        timings = []
        for _ in range(num_repeats):
            start = time.perf_counter()
            with torch.no_grad():
                _ = net_cpu(inputs_cpu)
            end = time.perf_counter()
            timings.append((end - start) * 1000) # ms
        
        results['cpu_times'].append(np.mean(timings))
        results['cpu_std'].append(np.std(timings))
        print(f"  ML (CPU): {np.mean(timings):.4f} ± {np.std(timings):.4f} ms")

        # ===========================
        # 2. ML GPU Benchmark
        # ===========================
        with torch.no_grad():
            _ = net_gpu(inputs_gpu)
        torch.cuda.synchronize() 
        
        timings = []
        for _ in range(num_repeats):
            torch.cuda.synchronize() # Sync before start
            start = time.perf_counter()
            with torch.no_grad():
                _ = net_gpu(inputs_gpu)
            torch.cuda.synchronize() # Sync after end
            end = time.perf_counter()
            timings.append((end - start) * 1000) # ms
        
        results['gpu_times'].append(np.mean(timings))
        results['gpu_std'].append(np.std(timings))
        print(f"  ML (GPU): {np.mean(timings):.4f} ± {np.std(timings):.4f} ms")
        

        # ===========================
        # 3. Conventional RTE Benchmark
        # ===========================
        # 策略：如果 size 很大，只计算前 100 个样本的时间然后线性外推 (Scale up)
        # 为了计算 std，我们把这个"采样-外推"的过程也重复 num_repeats 次
        
        # if size <= 25000: 
        #     n_iter_run = size
        #     scale_factor = 1.0
        #     is_estimated = False
        # else:
        #     n_iter_run = 100 # 大任务只跑100个样本
        #     scale_factor = size / n_iter_run
        #     is_estimated = True
            
        # timings = []
        # for _ in range(num_repeats):
        #     start = time.perf_counter()
        #     for i in range(n_iter_run):
        #         rte = RTE(omega_np[i], 1, 0.99, x_np[i], 21, 21, 'iso')
        #         rte.build()
        #         _, _ = rte.hemi_props()
        #     end = time.perf_counter()
            
        #     run_time = (end - start) * 1000 # ms
        #     total_estimated_time = run_time * scale_factor
        #     timings.append(total_estimated_time)
            
        # results['rte_times'].append(np.mean(timings))
        # results['rte_std'].append(np.std(timings))
        
        # est_tag = "(Estimated)" if is_estimated else ""
        # print(f"  C-RTE   : {np.mean(timings):.4f} ± {np.std(timings):.4f} ms {est_tag}")

    # # Save Results including STD
    np.save('benchmark_forward_results_with_std_rte.npy', results)
    # np.save('benchmark_forward_results_with_std_ML.npy', results)
    # print("\nForward benchmark saved to benchmark_forward_results_with_std.npy")

if __name__ == '__main__':
    benchmark_forward_workloads()