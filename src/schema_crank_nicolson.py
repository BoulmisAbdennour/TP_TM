"""
Schéma de Crank-Nicolson — EDP (E1)
======================================
    u_t = u_xx - alpha*u,   x in ]0,1[, t > 0
    u(0,t) = u(1,t) = 0
    u(x,0) = f(x)

Schéma CN = moyenne de l'explicite (temps n) et l'implicite (temps n+1) :

    (u_j^{n+1} - u_j^n)/dt =
        (1/2)*[(u_{j+1}^n - 2u_j^n + u_{j-1}^n)/dx² - alpha*u_j^n]
      + (1/2)*[(u_{j+1}^{n+1} - 2u_j^{n+1} + u_{j-1}^{n+1})/dx² - alpha*u_j^{n+1}]

Système à résoudre (r = dt/dx²) :
    A * u^{n+1} = B * u^n

    A : diag = 1 + r + alpha*dt/2,  hors-diag = -r/2
    B : diag = 1 - r - alpha*dt/2,  hors-diag = +r/2

Stabilité (Von Neumann) :
    g = (1 - 4r*sin²(xi/2) - alpha*dt/2) / (1 + 4r*sin²(xi/2) + alpha*dt/2)
    |g| <= 1 toujours  =>  Inconditionnellement stable

Ordre : O(dt²) + O(dx²)  — meilleur ordre temporel parmi les 3 schémas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# ─── Paramètres ───────────────────────────────────────────────────────────────
alpha = 1.0
L     = 1.0
T     = 0.1
N     = 100
M     = 1000

dx = L / N
dt = T / M
r  = dt / dx**2
print(f"dx={dx:.4f}, dt={dt:.2e}, r=dt/dx²={r:.4f}  (inconditionnellement stable, ordre 2)")

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

# ─── Matrices A et B ──────────────────────────────────────────────────────────
size = N - 1

# Matrice A (côté gauche — implicite) au format banded
ab_A = np.zeros((3, size))
ab_A[0, 1:]  = -r / 2
ab_A[1, :]   = 1 + r + alpha*dt/2
ab_A[2, :-1] = -r / 2

# Coefficients de B (côté droit — explicite, appliqués manuellement)
c_B  = 1 - r - alpha*dt/2    # coefficient diagonal
o_B  = r / 2                 # coefficient hors-diagonal

# ─── Résolution ───────────────────────────────────────────────────────────────
def solve_cn(f_init):
    plot_steps = set(np.linspace(0, M, 6, dtype=int))
    snapshots  = {}

    u = f_init(x_int).copy()
    snapshots[0] = np.concatenate([[0], u, [0]])

    for n in range(1, M + 1):
        # Second membre : rhs = B * u^n
        rhs = np.empty(size)
        rhs[0]    = c_B*u[0]    + o_B*u[1]
        rhs[-1]   = o_B*u[-2]   + c_B*u[-1]
        rhs[1:-1] = o_B*u[:-2] + c_B*u[1:-1] + o_B*u[2:]

        # Résolution : A * u^{n+1} = rhs
        u = solve_banded((1, 1), ab_A, rhs)

        if n in plot_steps:
            snapshots[n] = np.concatenate([[0], u, [0]])

    return snapshots

# ─── Visualisation ────────────────────────────────────────────────────────────
def plot_and_save(f_init, sol_exacte, label, filename):
    snapshots = solve_cn(f_init)
    steps     = sorted(snapshots.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Crank-Nicolson — {label}  (α={alpha}, r={r:.2f})",
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
    ax.plot(x_int, u_num, color='darkorange', lw=2,
            label=f"Crank-Nicolson (err={err:.2e})")
    ax.set_xlabel("x"); ax.set_ylabel(f"u(x, T={T})")
    ax.set_title("Comparaison numérique vs exact")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Erreur max {label}: {err:.3e}  →  {filename}")

plot_and_save(f1, sol_f1, "f₁(x) = sin(2πx)",
              "../figures/crank_nicolson_f1.png")
plot_and_save(f2, sol_f2, "f₂(x) chapeau",
              "../figures/crank_nicolson_f2.png")
