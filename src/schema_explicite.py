"""
Schéma explicite (FTCS) — EDP (E1)
====================================
    u_t = u_xx - alpha*u,   x in ]0,1[, t > 0
    u(0,t) = u(1,t) = 0
    u(x,0) = f(x)

Discrétisation :
    u_t  ← (u_j^{n+1} - u_j^n) / dt          ordre O(dt)
    u_xx ← (u_{j+1}^n - 2u_j^n + u_{j-1}^n) / dx²   ordre O(dx²)

Schéma explicite (r = dt/dx²) :
    u_j^{n+1} = r*u_{j-1}^n + (1-2r-alpha*dt)*u_j^n + r*u_{j+1}^n

Stabilité (Von Neumann) :
    g = 1 - 4r*sin²(xi/2) - alpha*dt
    Condition : r <= 1/2  =>  dt <= dx²/2

Forme matricielle :
    u^{n+1} = A * u^n
    A tridiagonale : diag = 1-2r-alpha*dt, hors-diag = r
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Paramètres ───────────────────────────────────────────────────────────────
alpha = 1.0       # coefficient de réaction (imposé par l'EDP E1)
L     = 1.0       # longueur du domaine
T     = 0.1       # temps final
N     = 100       # nombre d'intervalles en espace

dx = L / N
# Choix de dt : on impose r = 0.4 < 0.5 (condition de stabilité)
dt = 0.4 * dx**2
M  = int(round(T / dt))
dt = T / M        # ajustement pour tomber exactement en T

r = dt / dx**2
print(f"dx={dx:.4f}, dt={dt:.2e}, r=dt/dx²={r:.4f} ≤ 0.5 ✓ (stable)")

x_full = np.linspace(0, L, N + 1)          # x_0 ... x_N  (N+1 points)
x_int  = x_full[1:-1]                       # x_1 ... x_{N-1}  (inconnues)
t_arr  = np.linspace(0, T, M + 1)

# ─── Conditions initiales ─────────────────────────────────────────────────────
def f1(x): return np.sin(2 * np.pi * x)
def f2(x): return np.where(x <= 0.5, 2 * x, 2 * (1 - x))

# ─── Solutions analytiques ────────────────────────────────────────────────────
def sol_f1(x, t):
    """u(x,t) = sin(2pi*x) * exp(-(4pi²+alpha)*t)"""
    return np.sin(2 * np.pi * x) * np.exp(-(4 * np.pi**2 + alpha) * t)

def sol_f2(x, t, n_modes=201):
    """Série de Fourier : c_n = 8/n²pi² * sin(npi/2), n impairs"""
    u = np.zeros_like(x, dtype=float)
    for n in range(1, n_modes + 1, 2):
        cn = 8 * (-1)**((n - 1) // 2) / (n**2 * np.pi**2)
        u += cn * np.sin(n * np.pi * x) * np.exp(-(n**2 * np.pi**2 + alpha) * t)
    return u

# ─── Matrice A (tridiagonale, taille N-1) ─────────────────────────────────────
diag = (1 - 2*r - alpha*dt) * np.ones(N - 1)
off  = r * np.ones(N - 2)
A    = np.diag(diag) + np.diag(off, k=1) + np.diag(off, k=-1)

# ─── Résolution ───────────────────────────────────────────────────────────────
def solve_explicit(f_init):
    """Applique le schéma explicite et retourne la solution à tous les instants."""
    # Stockage uniquement des instants à afficher (économie mémoire)
    plot_steps = np.linspace(0, M, 6, dtype=int)
    snapshots  = {}

    u = f_init(x_int).copy()
    snapshots[0] = np.concatenate([[0], u, [0]])   # bords inclus

    for n in range(1, M + 1):
        u = A @ u
        if n in plot_steps:
            snapshots[n] = np.concatenate([[0], u, [0]])

    return snapshots

# ─── Visualisation ────────────────────────────────────────────────────────────
def plot_and_save(f_init, sol_exacte, label, filename):
    snapshots = solve_explicit(f_init)
    steps     = sorted(snapshots.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Schéma explicite — {label}  (α={alpha}, r={r:.3f})",
                 fontsize=13, fontweight='bold')

    # Gauche : évolution temporelle
    ax = axes[0]
    for s in steps:
        ax.plot(x_full, snapshots[s], label=f"t = {t_arr[s]:.3f}")
    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
    ax.set_title("Évolution temporelle")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 1)

    # Droite : comparaison avec solution exacte à T
    ax = axes[1]
    u_num = snapshots[M][1:-1]
    u_ex  = sol_exacte(x_int, T)
    err   = np.max(np.abs(u_num - u_ex))
    ax.plot(x_int, u_ex,  'k--', lw=2, label="Solution exacte")
    ax.plot(x_int, u_num, 'b-',  lw=2, label=f"Explicite (err={err:.2e})")
    ax.set_xlabel("x"); ax.set_ylabel(f"u(x, T={T})")
    ax.set_title("Comparaison numérique vs exact")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Erreur max {label}: {err:.3e}  →  {filename}")

plot_and_save(f1, sol_f1, "f₁(x) = sin(2πx)",
              "../figures/explicite_f1.png")
plot_and_save(f2, sol_f2, "f₂(x) chapeau",
              "../figures/explicite_f2.png")
