# TP TM — Résolution numérique de l'EDP (E₁)
**M1 CHPS — 2025-2026**

---

## L'équation étudiée

$$\begin{cases} u_t(x,t) = u_{xx}(x,t) - \alpha\, u(x,t), & x \in ]0,1[,\ t > 0 \\ u(0,t) = u(1,t) = 0 & \text{(Dirichlet homogène)} \\ u(x,0) = f(x) & \text{(condition initiale)} \end{cases}$$

C'est une équation de **diffusion-réaction** : la chaleur diffuse dans la barre (terme $u_{xx}$) tout en s'amortissant exponentiellement (terme $-\alpha u$).

---

## Structure du dépôt

```
TP_TM/
├── run_all.py                  ← Script principal (génère toutes les figures)
├── src/
│   ├── schema_explicite.py     ← Schéma FTCS (ordre 1 temps, ordre 2 espace)
│   ├── schema_implicite.py     ← Schéma implicite (ordre 1 temps, ordre 2 espace)
│   └── schema_crank_nicolson.py← Schéma CN (ordre 2 temps ET espace)
├── figures/                    ← Figures générées automatiquement
└── README.md
```

---

## Solutions analytiques

### Cas f₁(x) = sin(2πx)

Le mode propre $n=2$ est un mode propre exact du problème. La solution est :

$$u(x,t) = \sin(2\pi x)\cdot e^{-(4\pi^2+\alpha)t}$$

Le profil sinusoïdal se conserve et s'amortit exponentiellement.

### Cas f₂(x) — fonction chapeau

$$f_2(x) = \begin{cases} 2x & 0 < x \leq \tfrac{1}{2} \\ 2(1-x) & \tfrac{1}{2} < x < 1 \end{cases}$$

Développement en série de Fourier :

$$u(x,t) = \sum_{\substack{n=1 \\ n \text{ impair}}}^{+\infty} \frac{8(-1)^{(n-1)/2}}{n^2\pi^2}\,\sin(n\pi x)\,e^{-(n^2\pi^2+\alpha)t}$$

---

## Schémas numériques

Tous les schémas utilisent la discrétisation :
- Espace : $x_j = j\,\Delta x$, $j=0,\ldots,N$, $\Delta x = 1/N$
- Temps  : $t_n = n\,\Delta t$, $u_j^n \approx u(x_j, t_n)$
- Nombre de Fourier : $r = \Delta t / \Delta x^2$

### 1. Schéma Explicite (FTCS)

$$u_j^{n+1} = r\,u_{j-1}^n + (1-2r-\alpha\Delta t)\,u_j^n + r\,u_{j+1}^n$$

| Propriété | Valeur |
|---|---|
| Ordre en temps | $O(\Delta t)$ |
| Ordre en espace | $O(\Delta x^2)$ |
| Stabilité | **Conditionnelle** : $r \leq \tfrac{1}{2}$ |

### 2. Schéma Implicite

$$-r\,u_{j-1}^{n+1} + (1+2r+\alpha\Delta t)\,u_j^{n+1} - r\,u_{j+1}^{n+1} = u_j^n$$

| Propriété | Valeur |
|---|---|
| Ordre en temps | $O(\Delta t)$ |
| Ordre en espace | $O(\Delta x^2)$ |
| Stabilité | **Inconditionnelle** |

### 3. Schéma de Crank-Nicolson

Système $A\,u^{n+1} = B\,u^n$ avec :
- $A$ : diagonale $= 1+r+\tfrac{\alpha\Delta t}{2}$, hors-diagonale $= -r/2$
- $B$ : diagonale $= 1-r-\tfrac{\alpha\Delta t}{2}$, hors-diagonale $= +r/2$

| Propriété | Valeur |
|---|---|
| Ordre en temps | $O(\Delta t^2)$ ✓ |
| Ordre en espace | $O(\Delta x^2)$ |
| Stabilité | **Inconditionnelle** |

---

## Installation et utilisation

```bash
# Prérequis
pip install numpy matplotlib scipy

# Lancer tous les schémas et générer les figures
python run_all.py

# Ou lancer chaque schéma individuellement (depuis le dossier src/)
cd src
python schema_explicite.py
python schema_implicite.py
python schema_crank_nicolson.py
```

---

## Résultats (N=100, α=1, T=0.1)

| Schéma | Δt utilisé | r | Erreur f₁ | Erreur f₂ |
|---|---|---|---|---|
| Explicite | 4×10⁻⁵ | 0.40 | ~3.5×10⁻⁵ | ~2.0×10⁻⁵ |
| Implicite | 1×10⁻⁴ | 1.00 | ~1.7×10⁻⁴ | ~2.1×10⁻⁴ |
| Crank-Nicolson | 1×10⁻⁴ | 1.00 | ~2.3×10⁻⁵ | ~4.5×10⁻⁵ |

> **Observation** : Crank-Nicolson utilise le même grand $\Delta t$ que l'implicite mais donne une précision proche de l'explicite grâce à son ordre 2 en temps.

---

## Théorème de Lax

Pour chaque schéma, la convergence est garantie par :

$$\text{Convergence} \iff \text{Consistance} + \text{Stabilité}$$

- **Consistance** : erreur de troncature $\to 0$ quand $\Delta t, \Delta x \to 0$ ✓
- **Stabilité** : perturbations contrôlées (vérifiée par analyse de Von Neumann) ✓
- **Conclusion** : les 3 schémas convergent vers la solution exacte ✓
