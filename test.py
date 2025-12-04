import numpy as np

# a = np.load('benchmark_forward_results_with_std_rte.npy', allow_pickle=True).item()
# print(a['sizes'])
# print(a['rte_times'])

a = np.load('benchmark_inverse_results_full.npy', allow_pickle=True).item()
# print(a)
for keys in a.keys():
    if 'mean' in keys :
        print(f"{keys}: {a[keys]}")