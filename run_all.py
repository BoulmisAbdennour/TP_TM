"""
run_all.py — Script principal du TP TM
========================================
Lance les 3 schémas (explicite, implicite, Crank-Nicolson)
et génère toutes les figures + tableau de synthèse.

Usage :
    python run_all.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import os

os.makedirs("figures", exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# PARAMÈTRES COMMUNS
# ═══════════════════════════════════════════════════════════════
alpha = 1.0
L     = 1.0
T     = 0.1
N     = 100
M     = 1000

dx = L / N
dt = T / M
r  = dt / dx**2

# Pas de temps explicite (condition CFL : r_exp <= 0.5)
dt_exp = 0.4 * dx**2
M_exp  = int(round(T / dt_exp))
dt_exp = T / M_exp
r_exp  = dt_exp / dx**2

x_full = np.linspace(0, L, N + 1)
x_int  = x_full[1:-1]
t_arr  = np.linspace(0, T, M + 1)

print("=" * 60)
print("  TP TM — EDP (E1) : u_t = u_xx - alpha*u")
print("  M1 CHPS 2025-2026")
print("=" * 60)
print(f"  alpha={alpha}, N={N}, dx={dx:.4f}, T={T}")
print(f"  Explicite  : dt={dt_exp:.2e}, r={r_exp:.4f} ≤ 0.5 ✓")
print(f"  Implicite  : dt={dt:.2e},   r={r:.4f} (stable ∀r)")
print(f"  CN         : dt={dt:.2e},   r={r:.4f} (stable ∀r, ordre 2)")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# CONDITIONS INITIALES
# ═══════════════════════════════════════════════════════════════
def f1(x): return np.sin(2 * np.pi * x)
def f2(x): return np.where(x <= 0.5, 2 * x, 2 * (1 - x))

# ═══════════════════════════════════════════════════════════════
# SOLUTIONS ANALYTIQUES
# ═══════════════════════════════════════════════════════════════
def sol_f1(x, t):
    """
    f1 = sin(2pi*x) est le mode propre n=2.
    u(x,t) = sin(2pi*x) * exp(-(4pi²+alpha)*t)
    """
    return np.sin(2 * np.pi * x) * np.exp(-(4 * np.pi**2 + alpha) * t)

def sol_f2(x, t, n_modes=201):
    """
    f2 chapeau → série de Fourier :
    c_n = 8/n²pi² * sin(npi/2)  (n impairs seulement)
    """
    u = np.zeros_like(x, dtype=float)
    for n in range(1, n_modes + 1, 2):
        cn = 8 * (-1)**((n - 1) // 2) / (n**2 * np.pi**2)
        u += cn * np.sin(n * np.pi * x) * np.exp(-(n**2 * np.pi**2 + alpha) * t)
    return u

# ═══════════════════════════════════════════════════════════════
# SCHÉMA EXPLICITE
# ═══════════════════════════════════════════════════════════════
def run_explicit(f_init):
    c   = 1 - 2*r_exp - alpha*dt_exp
    u   = f_init(x_int).copy()
    t   = 0.0
    plot_steps = set(np.round(np.linspace(0, M_exp, 6)).astype(int))
    snaps = {0: np.r_[0, u, 0]}
    for n in range(1, M_exp + 1):
        u_new = np.empty(N - 1)
        u_new[0]    = c*u[0]    + r_exp*u[1]
        u_new[-1]   = r_exp*u[-2] + c*u[-1]
        u_new[1:-1] = r_exp*u[:-2] + c*u[1:-1] + r_exp*u[2:]
        u = u_new
        if n in plot_steps:
            snaps[n] = np.r_[0, u, 0]
    t_snaps = {s: s * dt_exp for s in snaps}
    return snaps, t_snaps

# ═══════════════════════════════════════════════════════════════
# SCHÉMA IMPLICITE
# ═══════════════════════════════════════════════════════════════
ab_imp = np.zeros((3, N - 1))
ab_imp[0, 1:]  = -r
ab_imp[1, :]   = 1 + 2*r + alpha*dt
ab_imp[2, :-1] = -r

def run_implicit(f_init):
    u = f_init(x_int).copy()
    plot_steps = set(np.round(np.linspace(0, M, 6)).astype(int))
    snaps = {0: np.r_[0, u, 0]}
    for n in range(1, M + 1):
        u = solve_banded((1, 1), ab_imp, u)
        if n in plot_steps:
            snaps[n] = np.r_[0, u, 0]
    t_snaps = {s: s * dt for s in snaps}
    return snaps, t_snaps

# ═══════════════════════════════════════════════════════════════
# SCHÉMA CRANK-NICOLSON
# ═══════════════════════════════════════════════════════════════
ab_cn = np.zeros((3, N - 1))
ab_cn[0, 1:]  = -r / 2
ab_cn[1, :]   = 1 + r + alpha*dt/2
ab_cn[2, :-1] = -r / 2
c_B = 1 - r - alpha*dt/2
o_B = r / 2

def run_cn(f_init):
    u = f_init(x_int).copy()
    plot_steps = set(np.round(np.linspace(0, M, 6)).astype(int))
    snaps = {0: np.r_[0, u, 0]}
    for n in range(1, M + 1):
        rhs = np.empty(N - 1)
        rhs[0]    = c_B*u[0]    + o_B*u[1]
        rhs[-1]   = o_B*u[-2]   + c_B*u[-1]
        rhs[1:-1] = o_B*u[:-2] + c_B*u[1:-1] + o_B*u[2:]
        u = solve_banded((1, 1), ab_cn, rhs)
        if n in plot_steps:
            snaps[n] = np.r_[0, u, 0]
    t_snaps = {s: s * dt for s in snaps}
    return snaps, t_snaps

# ═══════════════════════════════════════════════════════════════
# FIGURE 1 : Comparaison des 3 schémas vs exact à T
# ═══════════════════════════════════════════════════════════════
def fig_comparaison(f_init, sol_exacte, label, fname):
    sE, _  = run_explicit(f_init)
    sI, _  = run_implicit(f_init)
    sC, _  = run_cn(f_init)
    u_ex   = sol_exacte(x_int, T)

    schemas = [
        ("Explicite",      sE[max(sE)][1:-1], "steelblue"),
        ("Implicite",      sI[max(sI)][1:-1], "seagreen"),
        ("Crank-Nicolson", sC[max(sC)][1:-1], "darkorange"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Comparaison des schémas — {label}  (α={alpha}, T={T})",
                 fontsize=13, fontweight='bold')

    for ax, (nom, u_num, col) in zip(axes, schemas):
        err = np.max(np.abs(u_num - u_ex))
        ax.plot(x_int, u_ex,  'k--', lw=2, label="Exact")
        ax.plot(x_int, u_num, color=col, lw=2, label=f"{nom}\nerr={err:.2e}")
        ax.set_title(nom); ax.set_xlabel("x"); ax.set_ylabel("u(x,T)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: figures/{fname}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2 : Évolution temporelle (Crank-Nicolson)
# ═══════════════════════════════════════════════════════════════
def fig_evolution(f_init, sol_exacte, label, fname):
    snaps, t_snaps = run_cn(f_init)
    steps = sorted(snaps.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Évolution temporelle (Crank-Nicolson) — {label}",
                 fontsize=13, fontweight='bold')

    cmap = plt.cm.viridis
    ax = axes[0]
    for i, s in enumerate(steps):
        col = cmap(i / len(steps))
        ax.plot(x_full, snaps[s], color=col, label=f"t={t_snaps[s]:.3f}")
    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
    ax.set_title("Schéma Crank-Nicolson")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 1)

    ax = axes[1]
    for i, s in enumerate(steps):
        col = cmap(i / len(steps))
        t_val = t_snaps[s]
        u_ex  = sol_exacte(x_int, t_val)
        ax.plot(x_int, u_ex, '--', color=col, lw=1.5, alpha=0.7)
        ax.plot(x_int, snaps[s][1:-1], '-', color=col, lw=2,
                label=f"t={t_val:.3f}")
    ax.set_xlabel("x"); ax.set_ylabel("u(x,t)")
    ax.set_title("CN (plein) vs Exact (pointillé)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: figures/{fname}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3 : Étude de convergence
# ═══════════════════════════════════════════════════════════════
def fig_convergence(f_init, sol_exacte, label, fname):
    Ns   = [10, 20, 40, 80, 160, 320]
    errs = {"Explicite": [], "Implicite": [], "Crank-Nicolson": []}
    dxs  = []

    for n in Ns:
        dx_ = 1.0 / n
        dt_e = 0.4 * dx_**2;  M_e = int(round(T / dt_e));  dt_e = T / M_e
        dt_i = T / 1000
        r_e = dt_e / dx_**2
        r_i = dt_i / dx_**2
        xi  = np.linspace(0, 1, n + 1)[1:-1]
        sz  = n - 1
        dxs.append(dx_)

        # Explicite
        c = 1 - 2*r_e - alpha*dt_e
        u = f_init(xi).copy()
        for _ in range(M_e):
            un = np.empty(sz)
            un[0]    = c*u[0]    + r_e*u[1]
            un[-1]   = r_e*u[-2] + c*u[-1]
            if sz > 2:
                un[1:-1] = r_e*u[:-2] + c*u[1:-1] + r_e*u[2:]
            u = un
        errs["Explicite"].append(np.max(np.abs(u - sol_exacte(xi, T))))

        # Implicite
        ab = np.zeros((3, sz))
        ab[0, 1:]  = -r_i
        ab[1, :]   = 1 + 2*r_i + alpha*dt_i
        ab[2, :-1] = -r_i
        u = f_init(xi).copy()
        for _ in range(1000):
            u = solve_banded((1, 1), ab, u)
        errs["Implicite"].append(np.max(np.abs(u - sol_exacte(xi, T))))

        # CN
        abA = np.zeros((3, sz))
        abA[0, 1:]  = -r_i / 2
        abA[1, :]   = 1 + r_i + alpha*dt_i/2
        abA[2, :-1] = -r_i / 2
        cB_ = 1 - r_i - alpha*dt_i/2
        oB_ = r_i / 2
        u = f_init(xi).copy()
        for _ in range(1000):
            rhs = np.empty(sz)
            rhs[0]    = cB_*u[0]    + oB_*u[1]
            rhs[-1]   = oB_*u[-2]   + cB_*u[-1]
            if sz > 2:
                rhs[1:-1] = oB_*u[:-2] + cB_*u[1:-1] + oB_*u[2:]
            u = solve_banded((1, 1), abA, rhs)
        errs["Crank-Nicolson"].append(np.max(np.abs(u - sol_exacte(xi, T))))

    dxs_arr = np.array(dxs)
    fig, ax = plt.subplots(figsize=(8, 5))
    cols = {"Explicite":"steelblue","Implicite":"seagreen","Crank-Nicolson":"darkorange"}
    for nom, e in errs.items():
        ax.loglog(dxs_arr, e, 'o-', label=nom, color=cols[nom], lw=2)
    ref = np.array(errs["CN"] if "CN" in errs else errs["Crank-Nicolson"])
    ax.loglog(dxs_arr, errs["Explicite"][0]*(dxs_arr/dxs_arr[0])**2,
              'k--', alpha=0.4, label="Pente O(dx²)")
    ax.set_xlabel("Δx"); ax.set_ylabel("Erreur ||E||∞")
    ax.set_title(f"Convergence — {label}")
    ax.legend(); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: figures/{fname}")

# ═══════════════════════════════════════════════════════════════
# LANCEMENT
# ═══════════════════════════════════════════════════════════════
print("\n── Génération des figures ──")

print("\n[f1 = sin(2pi*x)]")
fig_comparaison(f1, sol_f1, "f₁(x) = sin(2πx)",  "comparaison_f1.png")
fig_evolution(f1,   sol_f1, "f₁(x) = sin(2πx)",  "evolution_f1.png")
fig_convergence(f1, sol_f1, "f₁(x) = sin(2πx)",  "convergence_f1.png")

print("\n[f2 = chapeau]")
fig_comparaison(f2, sol_f2, "f₂(x) chapeau",  "comparaison_f2.png")
fig_evolution(f2,   sol_f2, "f₂(x) chapeau",  "evolution_f2.png")
fig_convergence(f2, sol_f2, "f₂(x) chapeau",  "convergence_f2.png")

# ── Tableau de synthèse terminal ──
print()
print("=" * 60)
print("  TABLEAU DE SYNTHÈSE")
print("=" * 60)
print(f"  {'Schéma':<18} {'Ordre temps':<14} {'Ordre espace':<14} {'Stabilité'}")
print(f"  {'-'*58}")
print(f"  {'Explicite':<18} {'O(dt)':<14} {'O(dx²)':<14} {'dt ≤ dx²/2'}")
print(f"  {'Implicite':<18} {'O(dt)':<14} {'O(dx²)':<14} {'Inconditionnelle'}")
print(f"  {'Crank-Nicolson':<18} {'O(dt²)':<14} {'O(dx²)':<14} {'Inconditionnelle'}")
print()
print("  Erreurs à T=0.1 (N=100, schema vs exact) :")
sE, _ = run_explicit(f1);  eE1 = np.max(np.abs(sE[max(sE)][1:-1] - sol_f1(x_int,T)))
sI, _ = run_implicit(f1);  eI1 = np.max(np.abs(sI[max(sI)][1:-1] - sol_f1(x_int,T)))
sC, _ = run_cn(f1);        eC1 = np.max(np.abs(sC[max(sC)][1:-1] - sol_f1(x_int,T)))
sE, _ = run_explicit(f2);  eE2 = np.max(np.abs(sE[max(sE)][1:-1] - sol_f2(x_int,T)))
sI, _ = run_implicit(f2);  eI2 = np.max(np.abs(sI[max(sI)][1:-1] - sol_f2(x_int,T)))
sC, _ = run_cn(f2);        eC2 = np.max(np.abs(sC[max(sC)][1:-1] - sol_f2(x_int,T)))
print(f"  {'Schéma':<18} {'Erreur f1':<16} {'Erreur f2'}")
print(f"  {'-'*50}")
print(f"  {'Explicite':<18} {eE1:<16.3e} {eE2:.3e}")
print(f"  {'Implicite':<18} {eI1:<16.3e} {eI2:.3e}")
print(f"  {'Crank-Nicolson':<18} {eC1:<16.3e} {eC2:.3e}")
print("=" * 60)
print("\n✅ Toutes les figures générées dans ./figures/")
