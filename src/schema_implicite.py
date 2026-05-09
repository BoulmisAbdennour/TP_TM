"""
Schéma implicite — EDP (E1)
=============================
    u_t = u_xx - alpha*u,   x in ]0,1[, t > 0
    u(0,t) = u(1,t) = 0
    u(x,0) = f(x)

Discrétisation :
    u_t  ← (u_j^{n+1} - u_j^n) / dt          ordre O(dt)
    u_xx ← évalué au temps n+1                 ordre O(dx²)

Schéma implicite (r = dt/dx²) :
    -r*u_{j-1}^{n+1} + (1+2r+alpha*dt)*u_j^{n+1} - r*u_{j+1}^{n+1} = u_j^n

Stabilité (Von Neumann) :
    g = 1 / (1 + 4r*sin²(xi/2) + alpha*dt)  =>  |g| <= 1  toujours
    => Inconditionnellement stable

Forme matricielle :
    A * u^{n+1} = u^n
    A tridiagonale : diag = 1+2r+alpha*dt, hors-diag = -r
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ─── Paramètres ───────────────────────────────────────────────────────────────
alpha = 1.0
L     = 1.0
T     = 0.1
N     = 100     # intervalles en espace
M     = 1000    # pas de temps (pas de contrainte de stabilité)

dx = L / N
dt = T / M
r  = dt / dx**2
print(f"dx={dx:.4f}, dt={dt:.2e}, r=dt/dx²={r:.4f}  (inconditionnellement stable)")

x_full = np.linspace(0, L, N + 1)
x_int  = x_full[1:-1]
t_arr  = np.linspace(0, T, M + 1)

# ─── Conditions initiales ─────────────────────────────────────────────────────
def f1(x): return np.sin(2 * np.pi * x)
def f2(x): return np.where(x <= 0.5, 2 * x, 2 * (1 - x))

# ─── Solutions analytiques ────────────────────────────────────────────────────
def sol_f1(x, t):
    return np.sin(2 * np.pi * x) * np.exp(-(4 * np.pi**2 + alpha) * t)

def sol_f2(x, t, n_modes=201):
    u = np.zeros_like(x, dtype=float)
    for n in range(1, n_modes + 1, 2):
        cn = 8 * (-1)**((n - 1) // 2) / (n**2 * np.pi**2)
        u += cn * np.sin(n * np.pi * x) * np.exp(-(n**2 * np.pi**2 + alpha) * t)
    return u

# ─── Matrice A au format banded (scipy) ───────────────────────────────────────
# Format : ab[0] = sur-diagonale, ab[1] = diagonale, ab[2] = sous-diagonale
size = N - 1
ab = np.zeros((3, size))
ab[0, 1:]  = -r                      # sur-diagonale
ab[1, :]   = 1 + 2*r + alpha*dt      # diagonale principale
ab[2, :-1] = -r                      # sous-diagonale

# ─── Résolution ───────────────────────────────────────────────────────────────
def solve_implicit(f_init):
    plot_steps = set(np.linspace(0, M, 6, dtype=int))
    snapshots  = {}

    u = f_init(x_int).copy()
    snapshots[0] = np.concatenate([[0], u, [0]])

    for n in range(1, M + 1):
        u = solve_banded((1, 1), ab, u)
        if n in plot_steps:
            snapshots[n] = np.concatenate([[0], u, [0]])

    return snapshots

# ─── Visualisation ────────────────────────────────────────────────────────────
def plot_and_save(f_init, sol_exacte, label, filename):
    snapshots = solve_implicit(f_init)
    steps     = sorted(snapshots.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Schéma implicite — {label}  (α={alpha}, r={r:.2f})",
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    for s in steps:
        ax.plot(x_full, snapshots[s], label=f"t = {t_arr[s]:.3f}")
    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
    ax.set_title("Évolution temporelle")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 1)

    ax = axes[1]
    u_num = snapshots[M][1:-1]
    u_ex  = sol_exacte(x_int, T)
    err   = np.max(np.abs(u_num - u_ex))
    ax.plot(x_int, u_ex,  'k--', lw=2, label="Solution exacte")
    ax.plot(x_int, u_num, 'g-',  lw=2, label=f"Implicite (err={err:.2e})")
    ax.set_xlabel("x"); ax.set_ylabel(f"u(x, T={T})")
    ax.set_title("Comparaison numérique vs exact")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Erreur max {label}: {err:.3e}  →  {filename}")

plot_and_save(f1, sol_f1, "f₁(x) = sin(2πx)",
              "../figures/implicite_f1.png")
plot_and_save(f2, sol_f2, "f₂(x) chapeau",
              "../figures/implicite_f2.png")
