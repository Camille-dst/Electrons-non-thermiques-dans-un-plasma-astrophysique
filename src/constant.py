import numpy as np

# ==========================================
# 1. Constantes physiques fondamentales (CGS)
# ==========================================
sigma_t = 6.6524e-25  # Section efficace de Thomson (cm²)
m_e = 9.1094e-28      # Masse de l'électron (g)
C = 2.9979e10         # Vitesse de la lumière (cm/s)

# ==========================================
# 2. Paramètres du modèle astrophysique
# ==========================================
B = 1.0               # Champ magnétique (Gauss)
R = 10e15             # Taille de la zone d'émission (cm)
s = 2.0               # Indice spectral d'injection
gamma_min = 10.0      # Facteur de Lorentz min
gamma_max = 10e6      # Facteur de Lorentz max
Q_0 = 1.0             # Normalisation de l'injection

# ==========================================
# 3. Grandeurs physiques calculées
# ==========================================
u_b = B**2 / (8 * np.pi)                        # Densité d'énergie magnétique (erg/cm³)
K = (4 * u_b * sigma_t) / (3 * m_e * C)          # Coefficient de perte synchrotron (s⁻¹)
T_ad = R / c                                     # Temps d'échappement (s)
gamma_c = 1.0 / (K * T_ad)                       # Facteur de Lorentz critique