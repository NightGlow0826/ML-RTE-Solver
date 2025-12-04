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
        ax.plot(data["x"], data["T"], label='Total Transmittance (Conv)', color='blue')
        ax.plot(data["x"], [T - np.exp(-x) for T, x in zip(data["T"], data["x"])], 
                label='Diffusive Transmittance', linestyle='--', color='blue', alpha=0.5)
        
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
        ax.plot(data["x"], data["T"], label='Total Transmittance (ML)', color='red')
        ax.plot(data["x"], [T - np.exp(-x) for T, x in zip(data["T"], data["x"])], 
                label='Diffusive Transmittance', linestyle='--', color='red', alpha=0.5)
        
        ax.set_xscale('log')
        ax.set_xlabel('Optical Depth')
        ax.set_ylabel('Transmittance')
        ax.set_title("ML Solver")
        ax.legend()
        ax.grid(True, which="both", ls="-", alpha=0.2)
        st.pyplot(fig)
    # with tab2:
    

##################### Part 2: Material Spectra #####################
with tab2:
    st.markdown("### Aerogel Material Spectra Simulation")

    col1, col2, col3 = st.columns(3)
    with col1:
        rho_input = st.number_input("Density (kg/m^3)", value=144.0, step=10.0, format="%.1f")
    with col2:
        thick_input = st.number_input("Thickness (mm)", value=4.75, step=0.1, format="%.2f")
    with col3:
        r_input = st.number_input("Mean Radius (nm)", value=3.05, step=0.1, format="%.2f")
    col_w1, col_w2, col_mode = st.columns(3)
    with col_w1:
        wl_start = st.number_input("Start Wavelength (nm)", value=200., step=50.0, min_value=150.0)
    with col_w2:
        wl_end = st.number_input("End Wavelength (nm)", value=1000.0, step=50.0, min_value=600.0)
    with col_mode:
        solver_mode = st.radio("Solver", ["ML Solver (Batch)", "Conventional (Loop)"], index=0)
    
    num_points = st.slider("Spectral Resolution", 100, 1000, 50)
    
    run_calc = st.button("Calculate Spectra", key="calc_spectra")

    if run_calc:
        waves_nm = np.linspace(wl_start, wl_end, num_points)
        
        xmax_list, omega_list, g_list = [], [], []
        beta_list = [] 

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
            

        xmax_arr = np.array(xmax_list)
        omega_arr = np.array(omega_list)
        beta_arr = np.array(beta_list)
        g_arr = np.array(g_list)
        
        T_total, R_total = [], []

        if "ML" in solver_mode:
            st.info("Using Vectorized ML Inference...")
            # phase_type = 'hg' 
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
            R_total = output[:, 1]
                
            
                
        else:
            my_bar = st.progress(0)
            st.info("Using Conventional RTE Loop...")
            for i in range(len(waves_nm)):
                rte_obj = RTE(omega_arr[i], 1, 0.99, xmax_arr[i], 21, 21, phase_type='hg', g=g_arr[i])
                rte_obj.build()
                t_val, r_val = rte_obj.hemi_props()
                T_total.append(t_val)
                R_total.append(r_val)
                my_bar.progress((i + 1) / len(waves_nm))
            T_total = np.array(T_total)
            R_total = np.array(R_total)

        # T_direct = exp(-tau)
        T_direct = np.exp(-xmax_arr)
        # T_diffuse = T_total - T_direct
        T_diffuse = T_total - T_direct

        fig, ax = plt.subplots()
        
        ax.plot(waves_nm, T_total, label='Total Transmittance', color='black', linewidth=2)
        ax.plot(waves_nm, T_direct, label='Direct Transmittance', linestyle='--', color='blue')
        ax.plot(waves_nm, T_diffuse, label='Diffuse Transmittance', linestyle='-.', color='red')
        
        ax.set_xlabel(r'Wavelength $\lambda$ (nm)')
        ax.set_ylabel('Transmittance')
        ax.set_title(f'Spectra (d={thick_input:.2f}mm, ' + r'$\rho$=' + f'{rho_input}, r={r_input}nm)')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
        st.pyplot(fig)



##################### Part 3: Inverse Model #####################
with tab3:
    st.markdown("### Inverse Material Design")
    st.info("Optimization using Nelder-Mead algorithm to find Density and Radius.")

    from PartitalNN import PartialNN


    def sigmoid_spectrum(waves_nm, sep_wl_nm, smooth=50):
        target_T = 1 / (1 + np.exp(-1 / smooth * (waves_nm - sep_wl_nm)))
        return np.clip(target_T, 1e-3, 1 - 1e-3)

    def get_mie_xmax_batch(rho, r_nm, wave_nm_arr, thick_mm):
        
        xmax_list = []
        temp_aero = Aerogel_Sample(thick_mm, rho, r_nm, 500, m0=1.0)
        
        for wl in wave_nm_arr:
            temp_aero.wavelength_nm = wl
            try:
                n_val = f_n(wl * 1e-9)
            except:
                n_val = 1.45 # Fallback
            temp_aero.m0 = n_val - 1e-4j
            temp_aero.build()
            x, _, _ = temp_aero.to_opt_set()
            xmax_list.append(x)
        return np.array(xmax_list)

  

    subtab_1, subtab_2 = st.tabs(["Fit RTE Spectrum", "Fit Artificial Spectrum"])

    # ---------------- Task 1: Fit RTE Generated Spectrum ----------------
    with subtab_1:

        st.markdown("#### 1. Generate & Fit RTE Spectrum")
        
        st.markdown("##### Ground Truth Parameters")
        c1, c2, c3, c4 = st.columns(4)
        gt_rho = c1.number_input("True Density (kg/m3)", 100.0, 500.0, 144.0, step=10.0)
        gt_r = c2.number_input("True Radius (nm)", 2.0, 10.0, 3.05, step=0.1)
        gt_thick = c3.number_input("Thickness (mm)", 1.0, 10.0, 4.75, step=0.1)
        num_points_fit = c4.slider("Wavelength Points", 10, 200, 10, help="Fewer points = Faster RTE optimization")

        c5, c6 = st.columns(2)
        with c5:
            w_1 = st.number_input("Start Wavelength (nm)", 150.0, 500.0, 200.0, step=10.0)
        with c6:
            w_2 = st.number_input("End Wavelength (nm)", 600.0, 1200.0, 1000.0, step=10.0)
        waves = np.linspace(w_1, w_2, num_points_fit)

        def get_mie_params(rho, r, wl_nm, thick_mm):
            try:
                n_val = f_n(wl_nm * 1e-9)
            except:
                n_val = 1.45
            aero = Aerogel_Sample(thick_mm, rho, optical_mean_r_nm=r,wavelength_nm= wl_nm, m0=n_val - 0j)
            aero.build()
            return aero.to_opt_set() # returns xmax, omega, g

        def calc_rte_spectrum_loop(rho, r, thick_mm, wave_arr):
           
            T_list = []
            # t1 = time.time()
            for wl in wave_arr:
                xmax, omega, g = get_mie_params(rho, r, wl, thick_mm)
                rte_obj = RTE(omega, 1, 0.99, xmax, 21, 21, phase_type='iso')
                rte_obj.build()
                T, R = rte_obj.hemi_props()
                T_list.append(T)
                # t2 = time.time()
                # st.text(f"RTE Spectrum Generation Time: {t2 - t1:.2f}s")
            return np.array(T_list)

        st.markdown("---")
        col_m1, col_m2 = st.columns(2)

        gt_T = calc_rte_spectrum_loop(gt_rho, gt_r, gt_thick, waves)
        # ==========================================
        # Method 1: Inverse NN -> Quasi-T -> Optimize
        # ==========================================
        with col_m1:
            st.subheader("Method 1: Inverse NN")
            st.caption("Step 1: Inverse NN predicts optical depth ($x_{max}$).\nStep 2: Optimize params to match $e^{-x_{max}}$.")
            
            if st.button("Run Method 1", key="btn_m1"):
                

                from PartitalNN import PartialNN
                inv_model = PartialNN().to(device)
                inv_model.load_state_dict(torch.load(r'models/partial_inverse.pth', map_location=device))
                inv_model.eval()
                target_T_tensor = torch.tensor(gt_T, dtype=torch.float32).reshape(-1, 1).to(device)
                t_start = time.time()
                with torch.no_grad():
                    pred_log_xmax = inv_model(target_T_tensor).cpu().numpy().flatten()
                t_1 = time.time()
                target_quasi_T = np.exp(-np.exp(pred_log_xmax)) # e^(-xmax)

                def obj_method_1(params):
                    rho, r = params
                    if rho < 10 or r < 0.1: return 1e6
                    curr_quasi_T = []
                    for wl in waves:
                        xmax, _, _ = get_mie_params(rho, r, wl, gt_thick)
                        curr_quasi_T.append(np.exp(-xmax))
                    return np.mean((np.array(curr_quasi_T) - target_quasi_T)**2)
                t_2 = time.time()
                res = minimize(obj_method_1, [200.0, 5.0], method='Nelder-Mead', tol=1e-4)
                
                t_end = time.time()
                
                st.success(f"Found optical properties in {t_1 - t_start:.2f}s, material properties in {t_end - t_2:.2f}s")
                st.write(f"**Found:** ρ={res.x[0]:.1f}, r={res.x[1]:.2f}")
                # Compute MSE Loss Between * spectra
                ##########
                
                fit_T = calc_rte_spectrum_loop(res.x[0], res.x[1], gt_thick, waves)
                fig1, ax1 = plt.subplots()
                ax1.plot(waves, gt_T, '-', label='GT (RTE)')
                ax1.plot(waves, fit_T, '.', label='Method 1')
                ax1.legend()
                st.pyplot(fig1)

        # ==========================================
        # Method 2: Direct RTE Optimization
        # ==========================================
        with col_m2:
            st.subheader("Method 2: Direct RTE")
            st.caption("Directly runs RTE solver inside the optimization loop. Very slow but physically rigorous.")
            
            if st.button("Run Method 2", key="btn_m2"):
                t_start = time.time()
                
                with st.spinner("Generating GT using RTE..."):
                    gt_T = calc_rte_spectrum_loop(gt_rho, gt_r, gt_thick, waves)

                status_box = st.empty()
                
                def obj_method_2(params):
                    rho, r = params
                    if rho < 10 or r < 0.1: return 1e6
                    curr_T = calc_rte_spectrum_loop(rho, r, gt_thick, waves)
                    loss = np.mean((curr_T - gt_T)**2)
                    status_box.text(f"Optimizing... ρ={rho:.1f}, r={r:.2f}, Loss={loss:.2e}")
                    return loss

                x0 = [200.0, 5.0]
                with st.spinner("Optimizing (this may take a while)..."):
                    res = minimize(obj_method_2, x0, method='Nelder-Mead', 
                                   options={'maxiter': 30, 'xatol': 1.0}, tol=1e-4)
                
                t_end = time.time()

                st.success(f"Done in {t_end - t_start:.2f}s")
                st.write(f"**Found:** ρ={res.x[0]:.1f}, r={res.x[1]:.2f}")

                fit_T = calc_rte_spectrum_loop(res.x[0], res.x[1], gt_thick, waves)
                fig2, ax2 = plt.subplots()
                ax2.plot(waves, gt_T, '-', label='GT (RTE)')
                ax2.plot(waves, fit_T, '.', label='Method 2')
                ax2.legend()
                st.pyplot(fig2)


    with subtab_2:
        st.markdown("#### 2. Fit Artificial Sigmoid Spectrum")
        
        st.markdown("##### Target Spectrum Parameters")
        c1, c2, c3, c4 = st.columns(4)
        cutoff_wl = c1.number_input("Cutoff Wavelength (nm)", 300.0, 900.0, 600.0, step=10.0)
        smoothness = c2.number_input("Smoothness", 10.0, 200.0, 50.0, step=10.0, help="Larger = Smoother transition")
        design_thick = c3.number_input("Design Thickness (mm)", 1.0, 10.0, 4.75, step=0.1)
        num_points_fit = c4.slider("Wavelength Points", 10, 200, 10, key="slider_s2", help="Fewer points = Faster RTE optimization")

        c5, c6 = st.columns(2)
        with c5:
            w_1 = st.number_input("Start Wavelength (nm)", 150.0, 500.0, 300.0, step=10.0, key="start_w_s2")
        with c6:
            w_2 = st.number_input("End Wavelength (nm)", 600.0, 1200.0, 1000.0, step=10.0, key="end_w_s2")
        waves = np.linspace(w_1, w_2, num_points_fit)

        def sigmoid_spectrum(w, center, smooth):
            val = 1 / (1 + np.exp(-(w - center) / smooth))
            return np.clip(val, 1e-3, 1 - 1e-3)

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

        st.markdown("---")
        
        target_T_sigmoid = sigmoid_spectrum(waves, cutoff_wl, smoothness)

        col_m1, col_m2 = st.columns(2)

        # ==========================================
        # Method 1: Inverse NN -> Quasi-T -> Optimize
        # ==========================================
        with col_m1:
            st.subheader("Method 1: Inverse NN")
            st.caption("Step 1: Inverse NN predicts optical depth ($x_{max}$).\nStep 2: Optimize params to match $e^{-x_{max}}$.")
            
            if st.button("Run Method 1", key="btn_m1_s2"):
                
                from PartitalNN import PartialNN
                inv_model = PartialNN().to(device)
                inv_model.load_state_dict(torch.load(r'models/partial_inverse.pth', map_location=device))
                inv_model.eval()
                
                target_T_tensor = torch.tensor(target_T_sigmoid, dtype=torch.float32).reshape(-1, 1).to(device)
                
                t_start = time.time()
                
                with torch.no_grad():
                    pred_log_xmax = inv_model(target_T_tensor).cpu().numpy().flatten()
                
                t_1 = time.time()
                
                target_quasi_T = np.exp(-np.exp(pred_log_xmax)) 

                def obj_method_1(params):
                    rho, r = params
                    if rho < 10 or r < 0.1: return 1e6
                    curr_quasi_T = []
                    for wl in waves:
                        xmax, _, _ = get_mie_params(rho, r, wl, design_thick)
                        curr_quasi_T.append(np.exp(-xmax))
                    return np.mean((np.array(curr_quasi_T) - target_quasi_T)**2)

                t_2 = time.time()
                res = minimize(obj_method_1, [200.0, 5.0], method='Nelder-Mead', tol=1e-4)
                t_end = time.time()
                
                st.success(f"Found optical properties in {t_1 - t_start:.2f}s, material properties in {t_end - t_2:.2f}s")
                st.write(f"**Found:** ρ={res.x[0]:.1f}, r={res.x[1]:.2f}")
                
                fit_T = calc_rte_spectrum_loop(res.x[0], res.x[1], design_thick, waves)
                
                fig1, ax1 = plt.subplots()
                ax1.plot(waves, target_T_sigmoid, '-', label='Target (Sigmoid)')
                ax1.plot(waves, fit_T, '.', label='Method 1 Fit')
                ax1.legend()
                st.pyplot(fig1)

        # ==========================================
        # Method 2: Direct RTE Optimization
        # ==========================================
        with col_m2:
            st.subheader("Method 2: Direct RTE")
            st.caption("Directly runs RTE solver inside the optimization loop. Very slow.")
            
            if st.button("Run Method 2", key="btn_m2_s2"):
                t_start = time.time()
                
                status_box = st.empty()
                
                def obj_method_2(params):
                    rho, r = params
                    if rho < 10 or r < 0.1: return 1e6
                    curr_T = calc_rte_spectrum_loop(rho, r, design_thick, waves)
                    loss = np.mean((curr_T - target_T_sigmoid)**2)
                    status_box.text(f"Optimizing... ρ={rho:.1f}, r={r:.2f}, Loss={loss:.2e}")
                    return loss

                x0 = [200.0, 5.0]
                with st.spinner("Optimizing (this may take a while)..."):
                    res = minimize(obj_method_2, x0, method='Nelder-Mead', 
                                   options={'maxiter': 30, 'xatol': 1.0}, tol=1e-4)
                
                t_end = time.time()

                st.success(f"Done in {t_end - t_start:.2f}s")
                st.write(f"**Found:** ρ={res.x[0]:.1f}, r={res.x[1]:.2f}")

                fit_T = calc_rte_spectrum_loop(res.x[0], res.x[1], design_thick, waves)
                
                fig2, ax2 = plt.subplots()
                ax2.plot(waves, target_T_sigmoid, '-', label='Target (Sigmoid)')
                ax2.plot(waves, fit_T, '.', label='Method 2 Fit')
                ax2.legend()
                st.pyplot(fig2)