# DiffPhoton: Differentiable Optical Neural Network Simulator
**Physics-aware AI accelerator design based on Hybrid Pockels Phase Shifters.**

## 📖 Overview
**DiffPhoton** is a high-speed, differentiable simulator for Photonic Integrated Circuits (PICs), specifically designed to validate the performance of **Takenaka Lab's Hybrid Phase Shifters**. 
Powered by **JAX**, it enables end-to-end optimization of optical circuit parameters (control voltages) for deep learning tasks.

## 🚀 Key Features
- **Physical Modeling**: Simulates the electro-optic **Pockels effect** ($\Delta n \propto V$) for ultra-low voltage operation.
- **Universal Mesh**: Implements a 6-MZI architecture capable of realizing arbitrary $4 \times 4$ unitary matrices (Reck/Clements topology).
- **Gradient-Based Learning**: Uses automatic differentiation to train the optical circuit like a neural network.
- **Robustness**: Proved resilience against **10% thermal/electrical crosstalk** via in-situ training.

## 🔬 Simulation Results

### 1. Device Characterization & Robustness
Demonstrated 100% accuracy in image classification even under **10% voltage leakage (crosstalk)** conditions. The AI automatically compensates for hardware imperfections.
![Crosstalk Result](output/final_battle_crosstalk.png)

### 2. Matrix Reproduction
Successfully reproduced arbitrary random orthogonal matrices with **Loss $\approx$ 0.0**.
![Matrix Result](output/perfect_result.png)

---

## 📊 Feasibility Study: Inverse Design for Material Specs

We performed a reverse-engineering analysis to determine the minimum required Pockels coefficient ($r$) for CMOS-compatible operation.

**Goal:** Find the material specs required to operate the circuit below **1.0V**.

- **Result**: A material with **$r \ge 40$ pm/V** is sufficient to operate within a **1.0V** limit. This suggests that ultra-high performance ($r=100$) is not strictly necessary for practical applications.
![Material Spec Tradeoff](output/material_spec_tradeoff.png)

---

## 🛠 Technology Stack
- **Language**: Python 3.10+
- **Core Engine**: Google JAX (Just-In-Time compilation)
- **Optimization**: Optax (Adam Optimizer)
- **Physics**: Wave Optics / Transfer Matrix Method (TMM)

## ��‍💻 Author
**DiffPhoton Project** (Developed during Takenaka Lab Research Simulation)
