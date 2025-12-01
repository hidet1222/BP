import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import value_and_grad, jit
import optax
import os

# --- 1. エンジンを「r可変型」に定義 ---
def create_engine_with_r(r_value):
    """ 指定された r (pm/V) を持つ物理エンジンを返す """
    
    def directional_coupler():
        val = 1.0 / jnp.sqrt(2.0)
        return jnp.array([[val, val * 1j], [val * 1j, val]])

    def pockels_phase_shifter(voltage):
        # 物理定数
        L = 2000e-6
        d = 0.3e-6
        wl = 1.55e-6
        n = 3.5
        r = r_value * 1e-12 # pm/V -> m/V に変換
        
        E = voltage / d
        dn = 0.5 * (n**3) * r * E
        phi = (2 * jnp.pi / wl) * dn * L
        
        return jnp.array([[jnp.exp(1j * phi), 0], [0, 1.0 + 0j]])

    def mzi_switch(voltage):
        DC = directional_coupler()
        PS = pockels_phase_shifter(voltage)
        return jnp.dot(DC, jnp.dot(PS, DC))

    @jit
    def simulate_mesh(voltages):
        # 6-MZI Universal Mesh
        T0 = mzi_switch(voltages[0])
        T1 = mzi_switch(voltages[1])
        L1 = jnp.block([[T0, jnp.zeros((2,2))], [jnp.zeros((2,2)), T1]])
        T2 = mzi_switch(voltages[2])
        L2 = jnp.eye(4, dtype=complex)
        L2 = L2.at[1:3, 1:3].set(T2)
        T3 = mzi_switch(voltages[3])
        T4 = mzi_switch(voltages[4])
        L3 = jnp.block([[T3, jnp.zeros((2,2))], [jnp.zeros((2,2)), T4]])
        T5 = mzi_switch(voltages[5])
        L4 = jnp.eye(4, dtype=complex)
        L4 = L4.at[1:3, 1:3].set(T5)
        U = jnp.dot(L4, jnp.dot(L3, jnp.dot(L2, L1)))
        return U

    return simulate_mesh

def run_spec_explorer():
    print("🚀 DiffPhoton: Material Spec Explorer (Reverse Engineering)...")
    print("   Goal: Find the minimum required 'r' coefficient for a given voltage limit.")

    # データセット: 2x2画像 (0 vs 1)
    img_1 = jnp.array([0.0, 0.707, 0.0, 0.707]) + 0j
    target_1 = jnp.array([0.0, 1.0, 0.0, 0.0])
    img_0 = jnp.array([0.5, 0.5, 0.5, -0.5]) + 0j
    target_0 = jnp.array([1.0, 0.0, 0.0, 0.0])

    # テストする r の値 (pm/V): 200から10まで下げていく
    r_candidates = [200, 150, 100, 80, 60, 40, 30, 20, 10]
    required_voltages = []

    for r_val in r_candidates:
        print(f"   Testing Material r = {r_val:3} pm/V ...", end="", flush=True)
        
        # エンジン生成
        mesh_fn = create_engine_with_r(float(r_val))

        @jit
        def predict(voltages, input_vec):
            U = mesh_fn(voltages)
            return jnp.abs(jnp.dot(U, input_vec))**2

        # Loss関数 (精度 + 電圧最小化)
        @jit
        def loss_fn(params):
            pred_0 = predict(params, img_0)
            pred_1 = predict(params, img_1)
            acc_loss = jnp.mean((pred_0 - target_0)**2) + jnp.mean((pred_1 - target_1)**2)
            # 正則化を強めにして、サボらずに一番低い電圧を探させる
            volt_reg = 1e-4 * jnp.mean(params**2)
            return acc_loss + volt_reg

        # 学習実行
        key = jax.random.PRNGKey(0)
        params = jax.random.uniform(key, shape=(6,), minval=-0.1, maxval=0.1)
        optimizer = optax.adam(learning_rate=0.05)
        opt_state = optimizer.init(params)

        # 500回学習
        for i in range(500):
            grads = jax.grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
        
        # 学習完了後、一番大きかった電圧の絶対値を取得
        max_v = jnp.max(jnp.abs(params))
        required_voltages.append(max_v)
        print(f" -> Needed {max_v:.4f} V")

    # --- グラフ作成 ---
    plt.figure(figsize=(10, 6))
    
    # トレードオフ曲線
    plt.plot(r_candidates, required_voltages, 'o-', linewidth=3, color='crimson')
    
    # ガイドライン (1.0V)
    plt.axhline(y=1.0, color='gray', linestyle='--', label='CMOS Driver Limit (1.0V)')
    # ガイドライン (3.0V)
    plt.axhline(y=3.0, color='blue', linestyle=':', label='High Voltage Amp (3.0V)')

    plt.title("Material Spec vs. Required Drive Voltage", fontsize=14)
    plt.xlabel("Pockels Coefficient r (pm/V) [Material Quality]", fontsize=12)
    plt.ylabel("Required Control Voltage (V) [Circuit Cost]", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    # 軸を反転 (右に行くほど高性能な材料)
    plt.gca().invert_xaxis()
    
    output_path = "output/material_spec_tradeoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Spec Analysis Complete.")
    print(f"   Graph saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_spec_explorer()
