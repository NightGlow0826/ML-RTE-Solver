import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')
import streamlit as st
from RTE_Truth_Model import RTE
from NN import Simple_NN
import torch
import numpy as np
import tqdm
from data.refractive import f_n, f_k
from Mie_1 import Aerogel_Sample
import scipy.optimize as optimize
from scipy.optimize import minimize
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.title("RTE Solver")



# with tab1:

tab1, tab2, tab3 = st.tabs(["Forward", 'Forward-Material Spectra', 'Inverse Spectra Fitting'])

##################### Part 1: Forward Model #####################
with tab1:

    if "t1_single_conv" not in st.session_state:
        st.session_state.t1_single_conv = None
    if "t1_single_ml" not in st.session_state:
        st.session_state.t1_single_ml = None
    if "t1_plot_conv" not in st.session_state:
        st.session_state.t1_plot_conv = None
    if "t1_plot_ml" not in st.session_state:
        st.session_state.t1_plot_ml = None

    # ==========================================
    # Part 1: Single Point Calculation
    # ==========================================
    st.markdown('### Single Point')
    col1, col2 = st.columns(2)

    with col1:
        albedo = st.slider("Albedo (0-1)", min_value=0.0, max_value=1.0, value=0.4, step=0.01)
    with col2:
        optical_depth = st.slider("Optical Depth (0-5)", min_value=0.0, max_value=5.0, value=0.6, step=0.01)
    
    phase = st.selectbox(
        "Phase function",
        options=['iso', 'ray', 'hg'],
        index=0,
        help="Phase functions: iso - isotropic, raileigh - Rayleigh scattering, hg - Henyey-Greenstein"
    )

    net = Simple_NN(num_features=2).to(device)
    forward_path = 'forward_models/model_Resnet_unif_5/iso/WithPINN_log_model/epoch_380.pth'

    try:
        net.load_state_dict(torch.load(forward_path, map_location=device, weights_only=True))
        net.eval()
    except FileNotFoundError:
        st.error(f"Model file not found at: {forward_path}")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        click_single_conv = st.button("Compute with Conventional Solver", key="btn_single_conv")
    with col_btn2:
        click_single_ml = st.button("Compute with ML", key="btn_single_ml")

    if click_single_conv:
        rte_obj = RTE(albedo, 1, 0.99, optical_depth, 21, 21, phase, g=0.5)
        rte_obj.build()
        T, R = rte_obj.hemi_props()
        st.session_state.t1_single_conv = (T, R)

    if click_single_ml:
        input_tensor = torch.tensor([[np.log(optical_depth), albedo]], dtype=torch.float32).to(device)
        with torch.no_grad():
            output = net(input_tensor).cpu().numpy()[0]
        T, R = output[0], output[1]
        st.session_state.t1_single_ml = (T, R)

    c_res1, c_res2 = st.columns(2)
    with c_res1:
        if st.session_state.t1_single_conv:
            T, R = st.session_state.t1_single_conv
            st.info(f"Conv Result:\nTransmittance: {T:.2f}, Reflectance: {R:.2f}")
    with c_res2:
        if st.session_state.t1_single_ml:
            T, R = st.session_state.t1_single_ml
            st.success(f"ML Result:\nTransmittance: {T:.2f}, Reflectance: {R:.2f}")
    st.markdown("---")

    # ==========================================
    # Part 2: Transmittance vs Optical Depth Plot
    # ==========================================
    st.markdown('### Transmittance versus Optical Depth')
    
    # Inputs
    num_points = st.number_input("Number of Points", min_value=10, max_value=1000, value=100, step=10)
  
    xmaxs = torch.logspace(np.log10(1e-3), np.log10(1e3), num_points)
    
    albedo_plot = st.slider("Albedo (0-1)", min_value=0.0, max_value=1.0, value=0.4, step=0.01, key="albedo_plot")
    omegas = torch.ones_like(xmaxs) * albedo_plot
    
    phase_plot = st.selectbox(
        "Phase function",
        options=['iso', 'ray', 'hg'],
        index=0,
        help="Phase functions...",
        key="phase_plot"
    )

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        clicked_plot_conv = st.button("Plot Conventional Curve", key="btn_plot_conv")
    with col_p2:
        clicked_plot_ml = st.button("Plot ML Curve", key="btn_plot_ml")
    
    my_bar = st.progress(0)

    if clicked_plot_conv:
        T_list = []
        # R_list = []
        for i, (xmax, omega) in enumerate(zip(xmaxs, omegas)):
            my_bar.progress((i + 1) / len(xmaxs))
            rte_obj = RTE(omega.item(), 1, 0.99, xmax.item(), 21, 21, phase_plot)
            rte_obj.build()
            T, R = rte_obj.hemi_props()
            T_list.append(T)
            # R_list.append(R)
        
        st.session_state.t1_plot_conv = {
            "x": xmaxs.cpu().numpy(),
            "T": np.array(T_list)
        }
        my_bar.empty()

    if clicked_plot_ml:
        input_tensor = torch.stack([torch.log(xmaxs), omegas], dim=1).to(device)
        with torch.no_grad():
            output = net(input_tensor).cpu().numpy()
        T_list = output[:, 0]
        
        st.session_state.t1_plot_ml = {
            "x": xmaxs.cpu().numpy(),
            "T": T_list
        }

    import matplotlib.pyplot as plt
    
    if st.session_state.t1_plot_conv is not None:
        data = st.session_state.t1_plot_conv
        fig, ax = plt.subplots()
        ax.plot(data["x"], data["T"], label='Total Transmittance (Conv)')
        ax.plot(data["x"], [T - np.exp(-x) for T, x in zip(data["T"], data["x"])], 
                label='Diffusive Transmittance', linestyle='--', alpha=0.5)
        
        ax.set_xscale('log')
        ax.set_xlabel('Optical Depth')
        ax.set_ylabel('Transmittance')
        ax.set_title("Conventional Solver")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        st.pyplot(fig)

    if st.session_state.t1_plot_ml is not None:
        data = st.session_state.t1_plot_ml
        fig, ax = plt.subplots()
        ax.plot(data["x"], data["T"], label='Total Transmittance (ML)')
        ax.plot(data["x"], [T - np.exp(-x) for T, x in zip(data["T"], data["x"])], 
                label='Diffusive Transmittance', linestyle='--', alpha=0.5)
        
        ax.set_xscale('log')
        ax.set_xlabel('Optical Depth')
        ax.set_ylabel('Transmittance')
        ax.set_title("ML Solver")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        st.pyplot(fig)
    

##################### Part 2: Material Spectra #####################
if "ml_data" not in st.session_state:
    st.session_state["ml_data"] = None
if "conv_data" not in st.session_state:
    st.session_state["conv_data"] = None


def plot_spectra_result(waves, T_total, T_direct, T_diffuse, title):
    fig, ax = plt.subplots()
    ax.plot(waves, T_total, label='Total T', linewidth=2)
    ax.plot(waves, T_direct, label='Direct T', linestyle='--')
    ax.plot(waves, T_diffuse, label='Diffuse T', linestyle='-.')
    
    ax.set_xlabel(r'Wavelength $\lambda$ (nm)')
    ax.set_ylabel('Transmittance')
    # ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend()
    return fig

# --- 3. 主程序逻辑 ---
with tab2:
    st.markdown("### Aerogel Material Spectra Simulation (Comparison)")

    # 输入区域
    col1, col2, col3 = st.columns(3)
    with col1:
        rho_input = st.number_input("Density (kg/m^3)", value=144.0, step=10.0, format="%.1f")
    with col2:
        thick_input = st.number_input("Thickness (mm)", value=4.75, step=0.1, format="%.2f")
    with col3:
        r_input = st.number_input("Mean Radius (nm)", value=3.05, step=0.1, format="%.2f")
        
    col_w1, col_w2, col_res = st.columns(3)
    with col_w1:
        wl_start = st.number_input("Start Wavelength (nm)", value=200., step=50.0, min_value=150.0)
    with col_w2:
        wl_end = st.number_input("End Wavelength (nm)", value=1000.0, step=50.0, min_value=600.0)
    with col_res:
        num_points = st.slider("Spectral Resolution", 100, 1000, 50)

    st.markdown("---")
    
    # --- 4. 操作按钮区域 ---
    # 使用两列放置两个独立的按钮
    btn_col1, btn_col2 = st.columns(2)
    
    # 预计算通用物理参数 (当任意按钮按下时调用)
    def prepare_optical_properties():
        waves_nm = np.linspace(wl_start, wl_end, num_points)
        xmax_list, omega_list, g_list, beta_list = [], [], [], []
        
        aero_sample = Aerogel_Sample(
            thickness_mm=thick_input,
            density=rho_input,
            optical_mean_r_nm=r_input,
            wavelength_nm=500, 
            m0=1.0 - 0j 
        )
        
        for i, wl in enumerate(waves_nm):
            aero_sample.wavelength_nm = wl
            n_val = f_n(wl * 1e-9) 
            aero_sample.m0 = n_val
            aero_sample.build()
            
            xmax, omega, g = aero_sample.to_opt_set()
            xmax_list.append(xmax)
            omega_list.append(omega)
            g_list.append(g)
            beta_list.append(aero_sample.beta)
            
        return waves_nm, np.array(xmax_list), np.array(omega_list), np.array(g_list)

    with btn_col1:
        if st.button("Run ML Solver", type="primary"):
            waves_nm, xmax_arr, omega_arr, g_arr = prepare_optical_properties()
            
            # ML Inference Logic
            net_spec = Simple_NN(num_features=2).to(device)
            net_spec.load_state_dict(torch.load(f'forward_models/model_Resnet_unif_5/iso/WithPINN_log_model/epoch_380.pth', weights_only=True, map_location=device))
            net_spec.eval()

            input_tensor = torch.stack([
                torch.log(torch.tensor(xmax_arr, dtype=torch.float32)), 
                torch.tensor(omega_arr, dtype=torch.float32)
            ], dim=1).to(device)
            
            with torch.no_grad():
                output = net_spec(input_tensor).cpu().numpy()
            
            T_total = output[:, 0]
            # R_total = output[:, 1] # 暂时不用
            
            T_direct = np.exp(-xmax_arr)
            T_diffuse = T_total - T_direct
            
            st.session_state["ml_data"] = {
                "waves": waves_nm, "total": T_total, "direct": T_direct, "diffuse": T_diffuse,
                "params": f"d={thick_input}, rho={rho_input}"
            }
            st.success("ML Inference Done!")

    # 右侧按钮：运行 Conventional Loop
    with btn_col2:
        if st.button("Run Conventional Loop "):
            waves_nm, xmax_arr, omega_arr, g_arr = prepare_optical_properties()
            
            T_total_list = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(len(waves_nm)):
                rte_obj = RTE(omega_arr[i], 1, 0.99, xmax_arr[i], 21, 21, phase_type='hg', g=g_arr[i])
                rte_obj.build()
                t_val, r_val = rte_obj.hemi_props()
                T_total_list.append(t_val)
                
                # Update progress
                if i % 10 == 0:
                    progress_bar.progress((i + 1) / len(waves_nm))
            
            progress_bar.empty()
            
            T_total = np.array(T_total_list)
            T_direct = np.exp(-xmax_arr)
            T_diffuse = T_total - T_direct
            
            # 存入 Session State
            st.session_state["conv_data"] = {
                "waves": waves_nm, "total": T_total, "direct": T_direct, "diffuse": T_diffuse,
                "params": f"d={thick_input}, rho={rho_input}"
            }
            st.success("Conventional Loop Done!")

    # --- 5. 结果展示区域 (两栏对比) ---
    st.markdown("### Results Comparison")
    res_col1, res_col2 = st.columns(2)

    # 左栏：展示 ML 结果
    with res_col1:
        st.markdown("**ML Solver Result**")
        data = st.session_state["ml_data"]
        if data is not None:
            fig_ml = plot_spectra_result(
                data["waves"], data["total"], data["direct"], data["diffuse"],
                f"ML Prediction ({data['params']})"
            )
            st.pyplot(fig_ml)
        else:
            st.info("No ML data. Click 'Run ML Solver'.")

    # 右栏：展示 Conventional 结果
    with res_col2:
        st.markdown("**Conventional Result**")
        data = st.session_state["conv_data"]
        if data is not None:
            fig_conv = plot_spectra_result(
                data["waves"], data["total"], data["direct"], data["diffuse"],
                f"RTE Loop ({data['params']})"
            )
            st.pyplot(fig_conv)
        else:
            st.info("No Conventional data. Click 'Run Conventional Loop'.")



##################### Part 3: Inverse Model #####################
with tab3:
    st.markdown("### Inverse Material Design")
    st.info("Optimization using Nelder-Mead algorithm to find Density and Radius.")
    
    import time
    from scipy.optimize import minimize
    from PartitalNN import PartialNN

    # --- 1. Session State Initialization ---
    # 初始化 Subtab 1 (RTE Fit) 的状态
    if "t3_s1_gt_data" not in st.session_state:
        st.session_state["t3_s1_gt_data"] = None # 存储 Ground Truth
    if "t3_s1_m1_res" not in st.session_state:
        st.session_state["t3_s1_m1_res"] = None  # 存储 Method 1 结果
    if "t3_s1_m2_res" not in st.session_state:
        st.session_state["t3_s1_m2_res"] = None  # 存储 Method 2 结果

    # 初始化 Subtab 2 (Artificial Fit) 的状态
    if "t3_s2_m1_res" not in st.session_state:
        st.session_state["t3_s2_m1_res"] = None
    if "t3_s2_m2_res" not in st.session_state:
        st.session_state["t3_s2_m2_res"] = None

    # --- 2. Helper Functions (Defined once) ---
    def get_mie_params(rho, r, wl_nm, thick_mm):
        try:
            n_val = f_n(wl_nm * 1e-9)
        except:
            n_val = 1.45
        aero = Aerogel_Sample(thick_mm, rho, optical_mean_r_nm=r, wavelength_nm=wl_nm, m0=n_val - 0j)
        aero.build()
        return aero.to_opt_set()

    def calc_rte_spectrum_loop(rho, r, thick_mm, wave_arr):
        T_list = []
        for wl in wave_arr:
            xmax, omega, g = get_mie_params(rho, r, wl, thick_mm)
            rte_obj = RTE(omega, 1, 0.99, xmax, 21, 21, phase_type='iso')
            rte_obj.build()
            T, _ = rte_obj.hemi_props()
            T_list.append(T)
        return np.array(T_list)

    def sigmoid_spectrum_func(w, center, smooth):
        val = 1 / (1 + np.exp(-(w - center) / smooth))
        return np.clip(val, 1e-3, 1 - 1e-3)

    # --- 3. Tabs Logic ---
    subtab_1, subtab_2 = st.tabs(["Fit RTE Spectrum", "Fit Artificial Spectrum"])

    # =========================================================================
    # Subtab 1: Fit RTE Generated Spectrum
    # =========================================================================
    with subtab_1:
        st.markdown("#### 1. Generate & Fit RTE Spectrum")
        
        # Inputs
        st.markdown("##### Ground Truth Parameters")
        c1, c2, c3, c4 = st.columns(4)
        gt_rho = c1.number_input("True Density (kg/m3)", 100.0, 500.0, 144.0, step=10.0)
        gt_r = c2.number_input("True Radius (nm)", 2.0, 10.0, 3.05, step=0.1)
        gt_thick = c3.number_input("Thickness (mm)", 1.0, 10.0, 4.75, step=0.1)
        num_points_fit = c4.slider("Wavelength Points", 10, 200, 15, help="Fewer points = Faster RTE optimization")

        c5, c6 = st.columns(2)
        with c5:
            w_1 = st.number_input("Start Wavelength (nm)", 150.0, 500.0, 200.0, step=10.0)
        with c6:
            w_2 = st.number_input("End Wavelength (nm)", 600.0, 1200.0, 1000.0, step=10.0)
        
        # Button to Generate GT
        if st.button("Generate Ground Truth Spectrum", type="primary"):
            waves = np.linspace(w_1, w_2, num_points_fit)
            with st.spinner("Generating GT using RTE..."):
                gt_T = calc_rte_spectrum_loop(gt_rho, gt_r, gt_thick, waves)
                # Store in Session State
                st.session_state["t3_s1_gt_data"] = {
                    "waves": waves,
                    "gt_T": gt_T,
                    "params": (gt_rho, gt_r, gt_thick)
                }
                # Reset previous results since GT changed
                st.session_state["t3_s1_m1_res"] = None
                st.session_state["t3_s1_m2_res"] = None
                st.success("GT Spectrum Generated and Cached!")

        st.markdown("---")
        
        # Only show methods if GT data exists
        if st.session_state["t3_s1_gt_data"] is not None:
            gt_data = st.session_state["t3_s1_gt_data"]
            waves_s1 = gt_data["waves"]
            gt_T_s1 = gt_data["gt_T"]
            
            col_m1, col_m2 = st.columns(2)

            # --- Method 1: Inverse NN ---
            with col_m1:
                st.subheader("Method 1: Inverse NN")
                st.caption("Step 1: Inverse NN predicts optical depth ($x_{max}$).\nStep 2: Optimize params to match $e^{-x_{max}}$.")
                
                if st.button("Run Method 1", key="btn_m1"):
                    inv_model = PartialNN().to(device)
                    inv_model.load_state_dict(torch.load(r'models/partial_inverse.pth', map_location=device, weights_only=True))
                    inv_model.eval()
                    
                    target_T_tensor = torch.tensor(gt_T_s1, dtype=torch.float32).reshape(-1, 1).to(device)
                    
                    t_start = time.time()
                    with torch.no_grad():
                        pred_log_xmax = inv_model(target_T_tensor).cpu().numpy().flatten()
                    t_1 = time.time()
                    
                    target_quasi_T = np.exp(-np.exp(pred_log_xmax)) # e^(-xmax)

                    def obj_method_1(params):
                        rho, r = params
                        if rho < 10 or r < 0.1: return 1e6
                        curr_quasi_T = []
                        for wl in waves_s1:
                            xmax, _, _ = get_mie_params(rho, r, wl, gt_thick)
                            curr_quasi_T.append(np.exp(-xmax))
                        return np.mean((np.array(curr_quasi_T) - target_quasi_T)**2)

                    t_2 = time.time()
                    res = minimize(obj_method_1, [200.0, 5.0], method='Nelder-Mead', tol=1e-4)
                    t_end = time.time()
                    
                    # Calculate final fit for plotting
                    fit_T = calc_rte_spectrum_loop(res.x[0], res.x[1], gt_thick, waves_s1)
                    
                    # Store Result
                    st.session_state["t3_s1_m1_res"] = {
                        "rho": res.x[0], "r": res.x[1],
                        "time_op": t_1 - t_start, "time_mat": t_end - t_2,
                        "fit_T": fit_T
                    }

                # Display Method 1 Results
                res_m1 = st.session_state["t3_s1_m1_res"]
                if res_m1:
                    st.success(f"Found: ρ={res_m1['rho']:.1f}, r={res_m1['r']:.2f}")
                    st.caption(f"Time: {res_m1['time_op']:.2f}s (Opt) + {res_m1['time_mat']:.2f}s (Mat)")
                    fig1, ax1 = plt.subplots()
                    ax1.plot(waves_s1, gt_T_s1, '-', label='GT (RTE)',)
                    ax1.plot(waves_s1, res_m1['fit_T'], 'o', label='Method 1 Fit', markersize=4)
                    ax1.legend()
                    st.pyplot(fig1)

            # --- Method 2: Direct RTE ---
            with col_m2:
                st.subheader("Method 2: Direct RTE")
                st.caption("Directly runs RTE solver inside the optimization loop. Very slow.")
                
                if st.button("Run Method 2", key="btn_m2"):
                    status_box = st.empty()
                    
                    def obj_method_2(params):
                        rho, r = params
                        if rho < 10 or r < 0.1: return 1e6
                        curr_T = calc_rte_spectrum_loop(rho, r, gt_thick, waves_s1)
                        loss = np.mean((curr_T - gt_T_s1)**2)
                        status_box.text(f"Optimizing... ρ={rho:.1f}, r={r:.2f}, Loss={loss:.2e}")
                        return loss

                    x0 = [200.0, 5.0]
                    t_start = time.time()
                    with st.spinner("Optimizing (Direct RTE)..."):
                        res = minimize(obj_method_2, x0, method='Nelder-Mead', 
                                       options={'maxiter': 30, 'xatol': 1.0}, tol=1e-4)
                    t_end = time.time()
                    status_box.empty()
                    
                    fit_T = calc_rte_spectrum_loop(res.x[0], res.x[1], gt_thick, waves_s1)
                    
                    # Store Result
                    st.session_state["t3_s1_m2_res"] = {
                        "rho": res.x[0], "r": res.x[1],
                        "time": t_end - t_start,
                        "fit_T": fit_T
                    }

                # Display Method 2 Results
                res_m2 = st.session_state["t3_s1_m2_res"]
                if res_m2:
                    st.success(f"Found: ρ={res_m2['rho']:.1f}, r={res_m2['r']:.2f}")
                    st.caption(f"Time: {res_m2['time']:.2f}s")
                    fig2, ax2 = plt.subplots()
                    ax2.plot(waves_s1, gt_T_s1, '-', label='GT (RTE)',)
                    ax2.plot(waves_s1, res_m2['fit_T'], 'x', label='Method 2 Fit', )
                    ax2.legend()
                    st.pyplot(fig2)
        else:
            st.info("Please click 'Generate Ground Truth Spectrum' first.")


    # =========================================================================
    # Subtab 2: Fit Artificial Sigmoid Spectrum
    # =========================================================================
    with subtab_2:
        st.markdown("#### 2. Fit Artificial Sigmoid Spectrum")
        
        # Inputs
        st.markdown("##### Target Spectrum Parameters")
        c1, c2, c3, c4 = st.columns(4)
        cutoff_wl = c1.number_input("Cutoff Wavelength (nm)", 300.0, 900.0, 600.0, step=10.0)
        smoothness = c2.number_input("Smoothness", 10.0, 200.0, 50.0, step=10.0)
        design_thick = c3.number_input("Design Thickness (mm)", 1.0, 10.0, 4.75, step=0.1)
        num_points_fit_s2 = c4.slider("Wavelength Points", 10, 200, 15, key="slider_s2")

        c5, c6 = st.columns(2)
        with c5:
            w_1_s2 = st.number_input("Start Wavelength (nm)", 150.0, 500.0, 300.0, step=10.0, key="start_w_s2")
        with c6:
            w_2_s2 = st.number_input("End Wavelength (nm)", 600.0, 1200.0, 1000.0, step=10.0, key="end_w_s2")
        
        # 实时计算 Target (因为计算很快，不需要单独按钮生成)
        waves_s2 = np.linspace(w_1_s2, w_2_s2, num_points_fit_s2)
        target_T_sigmoid = sigmoid_spectrum_func(waves_s2, cutoff_wl, smoothness)
        
        st.markdown("---")
        
        col_m1_s2, col_m2_s2 = st.columns(2)

        # --- Method 1 (Subtab 2) ---
        with col_m1_s2:
            st.subheader("Method 1: Inverse NN")
            if st.button("Run Method 1", key="btn_m1_s2"):
                inv_model = PartialNN().to(device)
                inv_model.load_state_dict(torch.load(r'models/partial_inverse.pth', map_location=device, weights_only=True))
                inv_model.eval()
                
                target_T_tensor = torch.tensor(target_T_sigmoid, dtype=torch.float32).reshape(-1, 1).to(device)
                
                t_start = time.time()
                with torch.no_grad():
                    pred_log_xmax = inv_model(target_T_tensor).cpu().numpy().flatten()
                t_1 = time.time()
                
                target_quasi_T = np.exp(-np.exp(pred_log_xmax)) 

                def obj_method_1_s2(params):
                    rho, r = params
                    if rho < 10 or r < 0.1: return 1e6
                    curr_quasi_T = []
                    for wl in waves_s2:
                        xmax, _, _ = get_mie_params(rho, r, wl, design_thick)
                        curr_quasi_T.append(np.exp(-xmax))
                    return np.mean((np.array(curr_quasi_T) - target_quasi_T)**2)

                t_2 = time.time()
                res = minimize(obj_method_1_s2, [200.0, 5.0], method='Nelder-Mead', tol=1e-4)
                t_end = time.time()
                
                fit_T = calc_rte_spectrum_loop(res.x[0], res.x[1], design_thick, waves_s2)
                
                st.session_state["t3_s2_m1_res"] = {
                    "rho": res.x[0], "r": res.x[1],
                    "time": t_end - t_2,
                    "fit_T": fit_T,
                    "target_T": target_T_sigmoid, # 保存当时的Target以防Slider变动导致图不匹配
                    "waves": waves_s2
                }
            
            # Display Method 1
            res_s2_m1 = st.session_state["t3_s2_m1_res"]
            if res_s2_m1:
                st.success(f"Found: ρ={res_s2_m1['rho']:.1f}, r={res_s2_m1['r']:.2f}")
                fig1, ax1 = plt.subplots()
                # 使用保存的 waves 和 target，确保图和数据对应
                ax1.plot(res_s2_m1['waves'], res_s2_m1['target_T'], '-', label='Target')
                ax1.plot(res_s2_m1['waves'], res_s2_m1['fit_T'], 'o', label='Method 1 Fit')
                ax1.legend()
                st.pyplot(fig1)

        # --- Method 2 (Subtab 2) ---
        with col_m2_s2:
            st.subheader("Method 2: Direct RTE")
            if st.button("Run Method 2", key="btn_m2_s2"):
                status_box = st.empty()
                
                def obj_method_2_s2(params):
                    rho, r = params
                    if rho < 10 or r < 0.1: return 1e6
                    curr_T = calc_rte_spectrum_loop(rho, r, design_thick, waves_s2)
                    loss = np.mean((curr_T - target_T_sigmoid)**2)
                    status_box.text(f"Optimizing... ρ={rho:.1f}, r={r:.2f}, Loss={loss:.2e}")
                    return loss

                x0 = [200.0, 5.0]
                t_start = time.time()
                with st.spinner("Optimizing..."):
                    res = minimize(obj_method_2_s2, x0, method='Nelder-Mead', 
                                   options={'maxiter': 30, 'xatol': 1.0}, tol=1e-4)
                t_end = time.time()
                status_box.empty()
                
                fit_T = calc_rte_spectrum_loop(res.x[0], res.x[1], design_thick, waves_s2)
                
                st.session_state["t3_s2_m2_res"] = {
                    "rho": res.x[0], "r": res.x[1],
                    "time": t_end - t_start,
                    "fit_T": fit_T,
                    "target_T": target_T_sigmoid,
                    "waves": waves_s2
                }

            # Display Method 2
            res_s2_m2 = st.session_state["t3_s2_m2_res"]
            if res_s2_m2:
                st.success(f"Found: ρ={res_s2_m2['rho']:.1f}, r={res_s2_m2['r']:.2f}")
                fig2, ax2 = plt.subplots()
                ax2.plot(res_s2_m2['waves'], res_s2_m2['target_T'], '-', label='Target', )
                ax2.plot(res_s2_m2['waves'], res_s2_m2['fit_T'], 'x', label='Method 2 Fit', )
                ax2.legend()
                st.pyplot(fig2)