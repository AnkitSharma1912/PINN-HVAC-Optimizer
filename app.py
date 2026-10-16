import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- 1. PAGE SETUP & MODE SELECTION ---
st.set_page_config(page_title="PINN HVAC Optimizer (2D & 3D)", layout="wide")
st.title("🔥 Physics-Informed AI: Smart HVAC Placement (2D & 3D)")

st.sidebar.header("🌐 Simulation Mode")
sim_mode = st.sidebar.radio("Select Room Dimension:", ["2D Room (X, Y)", "3D Room (X, Y, Z)"])
is_3d = (sim_mode == "3D Room (X, Y, Z)")

st.sidebar.header("📍 Initial Heater Positions")

if not is_3d:
    col1, col2 = st.sidebar.columns(2)
    start_h1_x = col1.number_input("H1 X (m)", min_value=0.5, max_value=9.5, value=9.0)
    start_h1_y = col2.number_input("H1 Y (m)", min_value=0.5, max_value=9.5, value=8.0)

    col3, col4 = st.sidebar.columns(2)
    start_h2_x = col3.number_input("H2 X (m)", min_value=0.5, max_value=9.5, value=9.0)
    start_h2_y = col4.number_input("H2 Y (m)", min_value=0.5, max_value=9.5, value=9.0)
else:
    st.sidebar.markdown("**Heater 1 (X, Y, Z)**")
    c1, c2, c3 = st.sidebar.columns(3)
    start_h1_x = c1.number_input("H1 X", min_value=0.5, max_value=9.5, value=9.0)
    start_h1_y = c2.number_input("H1 Y", min_value=0.5, max_value=9.5, value=8.0)
    start_h1_z = c3.number_input("H1 Z", min_value=0.5, max_value=9.5, value=2.0)

    st.sidebar.markdown("**Heater 2 (X, Y, Z)**")
    c4, c5, c6 = st.sidebar.columns(3)
    start_h2_x = c4.number_input("H2 X", min_value=0.5, max_value=9.5, value=9.0)
    start_h2_y = c5.number_input("H2 Y", min_value=0.5, max_value=9.5, value=9.0)
    start_h2_z = c6.number_input("H2 Z", min_value=0.5, max_value=9.5, value=8.0)

st.sidebar.header("🌡️ Target & Training")
target_temp = st.sidebar.number_input("Target Room Temp (°C)", value=22.0)
EPOCHS_P1 = st.sidebar.slider("Phase 1 Epochs (Physics Training)", 1000, 20000, 4500, step=500)
EPOCHS_P2 = st.sidebar.slider("Phase 2 Epochs (Optimization)", 100, 2000, 1000, step=100)

# --- 2. PHYSICS CONSTANTS & DATA GENERATORS ---
ROOM_SIZE = 10.0  # 10m x 10m (2D) or 10m x 10m x 10m (3D)
N_INT = 1000
N_BND = 600 if is_3d else 400
WALL_TEMP, HEATER_POWER, HEATER_SPREAD = 10.0, 50.0, 1.5

def generate_data_2d():
    x_int = (torch.rand(N_INT, 1) * ROOM_SIZE).requires_grad_(True)
    y_int = (torch.rand(N_INT, 1) * ROOM_SIZE).requires_grad_(True)
    pts = N_BND // 4
    r = torch.rand(pts, 1) * ROOM_SIZE
    z = torch.zeros(pts, 1)
    f = torch.full((pts, 1), ROOM_SIZE)
    
    x_bnd = torch.cat([r, r, z, f])
    y_bnd = torch.cat([z, f, r, r])
    return x_int, y_int, x_bnd, y_bnd

def generate_data_3d():
    x_int = (torch.rand(N_INT, 1) * ROOM_SIZE).requires_grad_(True)
    y_int = (torch.rand(N_INT, 1) * ROOM_SIZE).requires_grad_(True)
    z_int = (torch.rand(N_INT, 1) * ROOM_SIZE).requires_grad_(True)
    
    pts = N_BND // 6  # 6 surfaces: 4 walls + floor + ceiling
    r1 = torch.rand(pts, 1) * ROOM_SIZE
    r2 = torch.rand(pts, 1) * ROOM_SIZE
    z = torch.zeros(pts, 1)
    f = torch.full((pts, 1), ROOM_SIZE)
    
    # x=0, x=10, y=0, y=10, z=0 (floor), z=10 (ceiling)
    x_bnd = torch.cat([z, f, r1, r1, r1, r1])
    y_bnd = torch.cat([r1, r1, z, f, r2, r2])
    z_bnd = torch.cat([r2, r2, r2, r2, z, f])
    return x_int, y_int, z_int, x_bnd, y_bnd, z_bnd

# --- 3. MODULAR PINN ARCHITECTURE (6-Input for 2D, 9-Input for 3D) ---
class SmartHeatPINN(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, *args):
        inputs = torch.cat(args, dim=1)
        return self.net(inputs)

# --- 4. EXECUTION ENGINE ---
if st.button("🚀 Run Diagnosis & Optimization", type="primary"):
    st.write(f"### ⚙️ Initializing {sim_mode} AI Engine...")
    
    if not is_3d:
        # ================= 2D EXECUTION =================
        x_int, y_int, x_bnd, y_bnd = generate_data_2d()
        model = SmartHeatPINN(in_features=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        st.write(f"### 🧠 Phase 1: Teaching 2D Thermodynamics ({EPOCHS_P1} Epochs)")
        p1_bar = st.progress(0)
        p1_text = st.empty()

        for epoch in range(EPOCHS_P1):
            optimizer.zero_grad()
            h1x_r, h1y_r = torch.rand_like(x_int) * ROOM_SIZE, torch.rand_like(x_int) * ROOM_SIZE
            h2x_r, h2y_r = torch.rand_like(x_int) * ROOM_SIZE, torch.rand_like(x_int) * ROOM_SIZE
            
            T_int = model(x_int, y_int, h1x_r, h1y_r, h2x_r, h2y_r)
            dT_dx = torch.autograd.grad(T_int, x_int, torch.ones_like(T_int), create_graph=True)[0]
            dT_dy = torch.autograd.grad(T_int, y_int, torch.ones_like(T_int), create_graph=True)[0]
            d2T_dx2 = torch.autograd.grad(dT_dx, x_int, torch.ones_like(dT_dx), create_graph=True)[0]
            d2T_dy2 = torch.autograd.grad(dT_dy, y_int, torch.ones_like(dT_dy), create_graph=True)[0]

            dist1 = (x_int - h1x_r)**2 + (y_int - h1y_r)**2
            dist2 = (x_int - h2x_r)**2 + (y_int - h2y_r)**2
            Q = HEATER_POWER * torch.exp(-dist1 / (2 * HEATER_SPREAD**2)) + \
                HEATER_POWER * torch.exp(-dist2 / (2 * HEATER_SPREAD**2))
            pde_loss = torch.mean((d2T_dx2 + d2T_dy2 + Q)**2)

            h1x_b, h1y_b = torch.rand_like(x_bnd) * ROOM_SIZE, torch.rand_like(x_bnd) * ROOM_SIZE
            h2x_b, h2y_b = torch.rand_like(x_bnd) * ROOM_SIZE, torch.rand_like(x_bnd) * ROOM_SIZE
            T_bnd = model(x_bnd, y_bnd, h1x_b, h1y_b, h2x_b, h2y_b)
            bc_loss = torch.mean((T_bnd - WALL_TEMP)**2)

            loss = pde_loss + 10.0 * bc_loss
            loss.backward()
            optimizer.step()

            if epoch % (EPOCHS_P1 // 10) == 0:
                p1_bar.progress(epoch / EPOCHS_P1)
                p1_text.text(f"Epoch {epoch} | 2D Physics Loss: {loss.item():.2f}")

        p1_bar.progress(1.0)
        p1_text.text(f"Phase 1 Complete! Final Physics Loss: {loss.item():.2f}")

        # Phase 2 (2D)
        st.write(f"### 🎯 Phase 2: Optimizing 2D Placement ({EPOCHS_P2} Epochs)")
        for param in model.parameters():
            param.requires_grad = False

        h_params = nn.Parameter(torch.tensor([start_h1_x, start_h1_y, start_h2_x, start_h2_y], dtype=torch.float32))
        h_optimizer = torch.optim.Adam([h_params], lr=0.1)
        h1_path_x, h1_path_y, h2_path_x, h2_path_y = [], [], [], []
        p2_bar = st.progress(0)
        p2_text = st.empty()

        for epoch in range(EPOCHS_P2):
            h_optimizer.zero_grad()
            h1_path_x.append(h_params[0].item())
            h1_path_y.append(h_params[1].item())
            h2_path_x.append(h_params[2].item())
            h2_path_y.append(h_params[3].item())

            hx1, hy1 = h_params[0] * torch.ones_like(x_int), h_params[1] * torch.ones_like(x_int)
            hx2, hy2 = h_params[2] * torch.ones_like(x_int), h_params[3] * torch.ones_like(x_int)
            T_pred = model(x_int, y_int, hx1, hy1, hx2, hy2)
            loss_opt = torch.mean((T_pred - target_temp)**2)
            loss_opt.backward()
            h_optimizer.step()
            h_params.data.clamp_(0.5, ROOM_SIZE - 0.5)

            if epoch % (EPOCHS_P2 // 10) == 0:
                p2_bar.progress(epoch / EPOCHS_P2)
                p2_text.text(f"Epoch {epoch} | Target Diff Loss: {loss_opt.item():.2f}")

        p2_bar.progress(1.0)
        st.success("✅ 2D Diagnosis & Optimization Complete!")

        colA, colB = st.columns(2)
        colA.metric("Optimal Heater 1 (2D)", f"X: {h_params[0].item():.2f}m, Y: {h_params[1].item():.2f}m")
        colB.metric("Optimal Heater 2 (2D)", f"X: {h_params[2].item():.2f}m, Y: {h_params[3].item():.2f}m")

        # 2D Plotting
        st.write("### 📊 Final 2D Thermal Heatmap & Path")
        grid_pts = np.linspace(0, ROOM_SIZE, 100)
        X, Y = np.meshgrid(grid_pts, grid_pts)
        X_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
        Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            T_final = model(
                X_flat, Y_flat,
                h_params[0] * torch.ones_like(X_flat), h_params[1] * torch.ones_like(X_flat),
                h_params[2] * torch.ones_like(X_flat), h_params[3] * torch.ones_like(X_flat)
            ).numpy().reshape(100, 100)

        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(X, Y, T_final, levels=50, cmap=cm.inferno)
        cbar = fig.colorbar(contour)
        cbar.set_label('Temperature (°C)')
        ax.plot(h1_path_x, h1_path_y, color='cyan', linestyle='--', linewidth=2, label='H1 Path')
        ax.plot(h2_path_x, h2_path_y, color='lime', linestyle='--', linewidth=2, label='H2 Path')
        ax.scatter([h1_path_x[0], h2_path_x[0]], [h1_path_y[0], h2_path_y[0]], color='blue', s=100, label='Start Position')
        ax.scatter([h_params[0].item(), h_params[2].item()], [h_params[1].item(), h_params[3].item()], color='red', marker='*', s=250, edgecolor='white', label='Optimal Placement')
        ax.set_xlabel("Room Width X (m)")
        ax.set_ylabel("Room Length Y (m)")
        ax.legend(loc='lower right')
        st.pyplot(fig)

    else:
        # ================= 3D EXECUTION =================
        x_int, y_int, z_int, x_bnd, y_bnd, z_bnd = generate_data_3d()
        model = SmartHeatPINN(in_features=9)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        st.write(f"### 🧠 Phase 1: Teaching 3D Volumetric Thermodynamics ({EPOCHS_P1} Epochs)")
        p1_bar = st.progress(0)
        p1_text = st.empty()

        for epoch in range(EPOCHS_P1):
            optimizer.zero_grad()
            h1x_r, h1y_r, h1z_r = torch.rand_like(x_int)*ROOM_SIZE, torch.rand_like(x_int)*ROOM_SIZE, torch.rand_like(x_int)*ROOM_SIZE
            h2x_r, h2y_r, h2z_r = torch.rand_like(x_int)*ROOM_SIZE, torch.rand_like(x_int)*ROOM_SIZE, torch.rand_like(x_int)*ROOM_SIZE

            T_int = model(x_int, y_int, z_int, h1x_r, h1y_r, h1z_r, h2x_r, h2y_r, h2z_r)
            dT_dx = torch.autograd.grad(T_int, x_int, torch.ones_like(T_int), create_graph=True)[0]
            dT_dy = torch.autograd.grad(T_int, y_int, torch.ones_like(T_int), create_graph=True)[0]
            dT_dz = torch.autograd.grad(T_int, z_int, torch.ones_like(T_int), create_graph=True)[0]
            
            d2T_dx2 = torch.autograd.grad(dT_dx, x_int, torch.ones_like(dT_dx), create_graph=True)[0]
            d2T_dy2 = torch.autograd.grad(dT_dy, y_int, torch.ones_like(dT_dy), create_graph=True)[0]
            d2T_dz2 = torch.autograd.grad(dT_dz, z_int, torch.ones_like(dT_dz), create_graph=True)[0]

            dist1 = (x_int - h1x_r)**2 + (y_int - h1y_r)**2 + (z_int - h1z_r)**2
            dist2 = (x_int - h2x_r)**2 + (y_int - h2y_r)**2 + (z_int - h2z_r)**2
            Q = HEATER_POWER * torch.exp(-dist1 / (2 * HEATER_SPREAD**2)) + \
                HEATER_POWER * torch.exp(-dist2 / (2 * HEATER_SPREAD**2))
            
            # 3D Poisson Heat Equation: d2T/dx2 + d2T/dy2 + d2T/dz2 + Q = 0
            pde_loss = torch.mean((d2T_dx2 + d2T_dy2 + d2T_dz2 + Q)**2)

            h1x_b, h1y_b, h1z_b = torch.rand_like(x_bnd)*ROOM_SIZE, torch.rand_like(x_bnd)*ROOM_SIZE, torch.rand_like(x_bnd)*ROOM_SIZE
            h2x_b, h2y_b, h2z_b = torch.rand_like(x_bnd)*ROOM_SIZE, torch.rand_like(x_bnd)*ROOM_SIZE, torch.rand_like(x_bnd)*ROOM_SIZE
            T_bnd = model(x_bnd, y_bnd, z_bnd, h1x_b, h1y_b, h1z_b, h2x_b, h2y_b, h2z_b)
            bc_loss = torch.mean((T_bnd - WALL_TEMP)**2)

            loss = pde_loss + 10.0 * bc_loss
            loss.backward()
            optimizer.step()

            if epoch % (EPOCHS_P1 // 10) == 0:
                p1_bar.progress(epoch / EPOCHS_P1)
                p1_text.text(f"Epoch {epoch} | 3D Physics Loss: {loss.item():.2f}")

        p1_bar.progress(1.0)
        p1_text.text(f"Phase 1 Complete! Final 3D Physics Loss: {loss.item():.2f}")

        # Phase 2 (3D)
        st.write(f"### 🎯 Phase 2: Optimizing 3D Coordinates ({EPOCHS_P2} Epochs)")
        for param in model.parameters():
            param.requires_grad = False

        h_params = nn.Parameter(torch.tensor([start_h1_x, start_h1_y, start_h1_z, start_h2_x, start_h2_y, start_h2_z], dtype=torch.float32))
        h_optimizer = torch.optim.Adam([h_params], lr=0.1)

        h1_px, h1_py, h1_pz = [], [], []
        h2_px, h2_py, h2_pz = [], [], []
        p2_bar = st.progress(0)
        p2_text = st.empty()

        for epoch in range(EPOCHS_P2):
            h_optimizer.zero_grad()
            h1_px.append(h_params[0].item())
            h1_py.append(h_params[1].item())
            h1_pz.append(h_params[2].item())
            h2_px.append(h_params[3].item())
            h2_py.append(h_params[4].item())
            h2_pz.append(h_params[5].item())

            hx1, hy1, hz1 = h_params[0]*torch.ones_like(x_int), h_params[1]*torch.ones_like(x_int), h_params[2]*torch.ones_like(x_int)
            hx2, hy2, hz2 = h_params[3]*torch.ones_like(x_int), h_params[4]*torch.ones_like(x_int), h_params[5]*torch.ones_like(x_int)

            T_pred = model(x_int, y_int, z_int, hx1, hy1, hz1, hx2, hy2, hz2)
            loss_opt = torch.mean((T_pred - target_temp)**2)
            loss_opt.backward()
            h_optimizer.step()
            h_params.data.clamp_(0.5, ROOM_SIZE - 0.5)

            if epoch % (EPOCHS_P2 // 10) == 0:
                p2_bar.progress(epoch / EPOCHS_P2)
                p2_text.text(f"Epoch {epoch} | 3D Target Diff Loss: {loss_opt.item():.2f}")

        p2_bar.progress(1.0)
        st.success("✅ 3D Volumetric Diagnosis & Optimization Complete!")

        colA, colB = st.columns(2)
        colA.metric("Optimal Heater 1 (3D)", f"X: {h_params[0].item():.2f}m, Y: {h_params[1].item():.2f}m, Z: {h_params[2].item():.2f}m")
        colB.metric("Optimal Heater 2 (3D)", f"X: {h_params[3].item():.2f}m, Y: {h_params[4].item():.2f}m, Z: {h_params[5].item():.2f}m")

        # 3D Volumetric Plotting
        st.write("### 📊 Final 3D Volumetric Thermal Field & Trajectories")
        grid_3d = np.linspace(0, ROOM_SIZE, 14)  # 14x14x14 volumetric grid for smooth rendering
        X, Y, Z = np.meshgrid(grid_3d, grid_3d, grid_3d)
        X_f = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
        Y_f = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
        Z_f = torch.tensor(Z.flatten(), dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            T_final_3d = model(
                X_f, Y_f, Z_f,
                h_params[0]*torch.ones_like(X_f), h_params[1]*torch.ones_like(X_f), h_params[2]*torch.ones_like(X_f),
                h_params[3]*torch.ones_like(X_f), h_params[4]*torch.ones_like(X_f), h_params[5]*torch.ones_like(X_f)
            ).numpy().flatten()

        fig = plt.figure(figsize=(11, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Render semi-transparent 3D thermal cloud
        scatter_cloud = ax.scatter(X.flatten(), Y.flatten(), Z.flatten(), c=T_final_3d, cmap=cm.inferno, alpha=0.12, s=18)
        cbar = fig.colorbar(scatter_cloud, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Temperature (°C)')

        # Render 3D Gradient Descent Paths
        ax.plot(h1_px, h1_py, h1_pz, color='cyan', linestyle='--', linewidth=3, label='H1 3D Path')
        ax.plot(h2_px, h2_py, h2_pz, color='lime', linestyle='--', linewidth=3, label='H2 3D Path')

        # Start & Optimal 3D Markers
        ax.scatter([h1_px[0], h2_px[0]], [h1_py[0], h2_py[0]], [h1_pz[0], h2_pz[0]], color='blue', s=120, label='Start Position', depthshade=False)
        ax.scatter([h_params[0].item(), h_params[3].item()], [h_params[1].item(), h_params[4].item()], [h_params[2].item(), h_params[5].item()], color='red', marker='*', s=350, edgecolor='black', label='Optimal 3D Placement', depthshade=False)

        ax.set_title(f"3D Optimized HVAC Placement (Target: {target_temp}°C)", fontsize=15)
        ax.set_xlabel("Width X (m)")
        ax.set_ylabel("Length Y (m)")
        ax.set_zlabel("Height Z (m)")
        ax.set_xlim(0, ROOM_SIZE)
        ax.set_ylim(0, ROOM_SIZE)
        ax.set_zlim(0, ROOM_SIZE)
        ax.legend(loc='upper left')
        st.pyplot(fig)
