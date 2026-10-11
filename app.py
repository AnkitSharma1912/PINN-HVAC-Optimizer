import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

st.set_page_config(page_title="PINN HVAC Optimizer", layout="wide")
st.title("🔥 Physics-Informed AI: Smart HVAC Placement")

st.sidebar.header("📍 Initial Heater Positions")
col1, col2 = st.sidebar.columns(2)
start_h1_x = col1.number_input("H1 X", min_value=0.5, max_value=9.5, value=1.0)
start_h1_y = col2.number_input("H1 Y", min_value=0.5, max_value=9.5, value=1.0)

col3, col4 = st.sidebar.columns(2)
start_h2_x = col3.number_input("H2 X", min_value=0.5, max_value=9.5, value=9.0)
start_h2_y = col4.number_input("H2 Y", min_value=0.5, max_value=9.5, value=9.0)

target_temp = st.sidebar.number_input("Target Room Temp (°C)", value=22.0)
EPOCHS_P1 = st.sidebar.slider("Phase 1 Epochs (Physics Training)", 1000, 20000, 4500, step=500)
EPOCHS_P2 = st.sidebar.slider("Phase 2 Epochs (Optimization)", 100, 2000, 1000, step=100)

ROOM_WIDTH, ROOM_LENGTH = 10.0, 10.0
N_INT, N_BND = 1000, 400
WALL_TEMP, HEATER_POWER, HEATER_SPREAD = 10.0, 50.0, 1.5

def generate_room_data():
    x_int = (torch.rand(N_INT, 1) * ROOM_WIDTH).requires_grad_(True)
    y_int = (torch.rand(N_INT, 1) * ROOM_LENGTH).requires_grad_(True)
    pts_per_wall = N_BND // 4
    rand_pos = torch.rand(pts_per_wall, 1)
    x_bnd = torch.cat([rand_pos * ROOM_WIDTH, rand_pos * ROOM_WIDTH, torch.zeros(pts_per_wall, 1), torch.full((pts_per_wall, 1), ROOM_WIDTH)])
    y_bnd = torch.cat([torch.zeros(pts_per_wall, 1), torch.full((pts_per_wall, 1), ROOM_LENGTH), rand_pos * ROOM_LENGTH, rand_pos * ROOM_LENGTH])
    return x_int, y_int, x_bnd, y_bnd

class SmartHeatPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x, y, h1x, h1y, h2x, h2y):
        inputs = torch.cat([x, y, h1x, h1y, h2x, h2y], dim=1)
        return self.net(inputs)

if st.button("🚀 Run Diagnosis & Optimization", type="primary"):
    st.write("### ⚙️ Initializing AI Engine...")
    x_int, y_int, x_bnd, y_bnd = generate_room_data()
    model = SmartHeatPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    st.write(f"### 🧠 Phase 1: Teaching AI Thermodynamics ({EPOCHS_P1} Epochs)")
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(EPOCHS_P1):
        optimizer.zero_grad()
        h1x_r, h1y_r = torch.rand_like(x_int) * ROOM_WIDTH, torch.rand_like(x_int) * ROOM_LENGTH
        h2x_r, h2y_r = torch.rand_like(x_int) * ROOM_WIDTH, torch.rand_like(x_int) * ROOM_LENGTH
        T_int = model(x_int, y_int, h1x_r, h1y_r, h2x_r, h2y_r)
        dT_dx = torch.autograd.grad(T_int, x_int, torch.ones_like(T_int), create_graph=True)[0]
        dT_dy = torch.autograd.grad(T_int, y_int, torch.ones_like(T_int), create_graph=True)[0]
        d2T_dx2 = torch.autograd.grad(dT_dx, x_int, torch.ones_like(dT_dx), create_graph=True)[0]
        d2T_dy2 = torch.autograd.grad(dT_dy, y_int, torch.ones_like(dT_dy), create_graph=True)[0]

        dist1 = (x_int - h1x_r)**2 + (y_int - h1y_r)**2
        dist2 = (x_int - h2x_r)**2 + (y_int - h2y_r)**2
        Q = HEATER_POWER * torch.exp(-dist1 / (2 * HEATER_SPREAD**2)) + HEATER_POWER * torch.exp(-dist2 / (2 * HEATER_SPREAD**2))
        pde_loss = torch.mean((d2T_dx2 + d2T_dy2 + Q)**2)

        h1x_b, h1y_b = torch.rand_like(x_bnd) * ROOM_WIDTH, torch.rand_like(x_bnd) * ROOM_LENGTH
        h2x_b, h2y_b = torch.rand_like(x_bnd) * ROOM_WIDTH, torch.rand_like(x_bnd) * ROOM_LENGTH
        T_bnd = model(x_bnd, y_bnd, h1x_b, h1y_b, h2x_b, h2y_b)
        bc_loss = torch.mean((T_bnd - WALL_TEMP)**2)

        loss = pde_loss + 10.0 * bc_loss
        loss.backward()
        optimizer.step()

        if epoch % (EPOCHS_P1 // 10) == 0:
            progress_bar.progress(epoch / EPOCHS_P1)
            status_text.text(f"Epoch {epoch} | Total Physics Loss: {loss.item():.2f}")
    
    progress_bar.progress(1.0)
    status_text.text(f"Phase 1 Complete! Final Physics Loss: {loss.item():.2f}")

    st.write(f"### 🎯 Phase 2: Optimizing Heater Placement ({EPOCHS_P2} Epochs)")
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
        h_params.data.clamp_(0.5, ROOM_WIDTH - 0.5)

        if epoch % (EPOCHS_P2 // 10) == 0:
            p2_bar.progress(epoch / EPOCHS_P2)
            p2_text.text(f"Epoch {epoch} | Target Diff Loss: {loss_opt.item():.2f} | H1: ({h_params[0]:.2f}, {h_params[1]:.2f})")

    p2_bar.progress(1.0)
    st.success("✅ Diagnosis & Optimization Complete!")

    colA, colB = st.columns(2)
    colA.metric("Optimal Heater 1", f"X: {h_params[0].item():.2f}, Y: {h_params[1].item():.2f}")
    colB.metric("Optimal Heater 2", f"X: {h_params[2].item():.2f}, Y: {h_params[3].item():.2f}")

    st.write("### 📊 Final Thermal Heatmap & Path")
    x_grid = np.linspace(0, ROOM_WIDTH, 100)
    y_grid = np.linspace(0, ROOM_LENGTH, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    X_flat = torch.tensor(X.flatten(), dtype=torch.float32).unsqueeze(1)
    Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32).unsqueeze(1)
    hx1_f, hy1_f = h_params[0] * torch.ones_like(X_flat), h_params[1] * torch.ones_like(X_flat)
    hx2_f, hy2_f = h_params[2] * torch.ones_like(X_flat), h_params[3] * torch.ones_like(X_flat)

    with torch.no_grad():
        T_final = model(X_flat, Y_flat, hx1_f, hy1_f, hx2_f, hy2_f).numpy().reshape(100, 100)

    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(X, Y, T_final, levels=50, cmap=cm.inferno)
    cbar = fig.colorbar(contour)
    cbar.set_label('Temperature (°C)')

    ax.plot(h1_path_x, h1_path_y, color='cyan', linestyle='--', label='H1 Path')
    ax.plot(h2_path_x, h2_path_y, color='lime', linestyle='--', label='H2 Path')
    ax.scatter(h1_path_x[0], h1_path_y[0], color='blue', s=100, label='Start Position')
    ax.scatter(h2_path_x[0], h2_path_y[0], color='blue', s=100)
    ax.scatter(h_params[0].item(), h_params[1].item(), color='red', marker='*', s=250, edgecolor='white', label='Optimal Placement')
    ax.scatter(h_params[2].item(), h_params[3].item(), color='red', marker='*', s=250, edgecolor='white')

    ax.set_xlabel("Room Width (m)")
    ax.set_ylabel("Room Length (m)")
    ax.legend(loc='lower right')
    st.pyplot(fig)
