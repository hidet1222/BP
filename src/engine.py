import jax.numpy as jnp
from jax import jit

# -- 1. Basic Components (Unit Cells) --

def directional_coupler():
    """Fixed 50:50 Splitter"""
    val = 1.0 / jnp.sqrt(2.0)
    return jnp.array([
        [val, val * 1j],
        [val * 1j, val]
    ])

def pockels_phase_shifter(voltage):
    """
    Hybrid Pockels Phase Shifter
    Simulating the r_33 electro-optic effect
    """
    # Parameters based on high-performance organic material
    r_pmV = 100.0  # 100 pm/V
    L = 2000e-6    # 2mm Length
    d = 0.3e-6     # 300nm Gap
    wl = 1.55e-6   # 1550nm Wavelength
    n = 3.5        # Refractive Index
    
    r = r_pmV * 1e-12
    E = voltage / d
    dn = 0.5 * (n**3) * r * E
    phi = (2 * jnp.pi / wl) * dn * L
    
    # Phase shift matrix
    return jnp.array([
        [jnp.exp(1j * phi), 0],
        [0, 1.0 + 0j]
    ])

def mzi_switch(voltage):
    """MZI = DC + PS + DC"""
    DC = directional_coupler()
    PS = pockels_phase_shifter(voltage)
    return jnp.dot(DC, jnp.dot(PS, DC))

# -- 2. Universal 4x4 Mesh Circuit --

@jit
def simulate_4x4_mesh(voltages):
    """
    6-MZI Universal Mesh Architecture
    Voltages: Array of 6 control values
    """
    # Layer 1
    T0 = mzi_switch(voltages[0])
    T1 = mzi_switch(voltages[1])
    L1 = jnp.block([[T0, jnp.zeros((2,2))], [jnp.zeros((2,2)), T1]])
    
    # Layer 2
    T2 = mzi_switch(voltages[2])
    L2 = jnp.eye(4, dtype=complex)
    L2 = L2.at[1:3, 1:3].set(T2)

    # Layer 3
    T3 = mzi_switch(voltages[3])
    T4 = mzi_switch(voltages[4])
    L3 = jnp.block([[T3, jnp.zeros((2,2))], [jnp.zeros((2,2)), T4]])
    
    # Layer 4
    T5 = mzi_switch(voltages[5])
    L4 = jnp.eye(4, dtype=complex)
    L4 = L4.at[1:3, 1:3].set(T5)
    
    # Total Transfer Matrix
    U_total = jnp.dot(L4, jnp.dot(L3, jnp.dot(L2, L1)))
    
    # Return Intensity (Power)
    return jnp.abs(U_total)**2
