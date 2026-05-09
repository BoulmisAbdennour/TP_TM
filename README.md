# TP TM — Résolution de l'Équation de Diffusion-Réaction $(E_1)$

> **M1 CHPS — Techniques de Modélisation — 2025-2026**  
> Boulmis Abdennour

---

## Table des matières

1. [Présentation du problème](#1-présentation-du-problème)
2. [Structure du dépôt](#2-structure-du-dépôt)
3. [Installation et utilisation](#3-installation-et-utilisation)
4. [Solutions analytiques](#4-solutions-analytiques)
5. [Schémas numériques](#5-schémas-numériques)
6. [Résultats](#6-résultats)
7. [Théorème de Lax — Convergence](#7-théorème-de-lax--convergence)

---

## 1. Présentation du problème

On résout analytiquement et numériquement l'EDP $(E_1)$ suivante :

$$
(E_1) \;:\quad
\begin{cases}
  u_t(x,t) = u_{xx}(x,t) - \alpha\, u(x,t), & x \in ]0,1[,\; t > 0, \\
  u(0,t) = u(1,t) = 0, & \text{(conditions de Dirichlet)} \\
  u(x,0) = f(x). & \text{(condition initiale)}
\end{cases}
$$

C'est une équation de **diffusion-réaction** avec $\alpha = 1$ :
- le terme $u_{xx}$ modélise la **diffusion** de la chaleur dans la barre,
- le terme $-\alpha u$ modélise un **amortissement** de la solution.

Deux conditions initiales sont étudiées :

| Nom | Expression | Remarque |
|-----|-----------|----------|
| $f_1$ | $\sin(2\pi x)$ | Mode propre exact — solution analytique simple |
| $f_2$ | $2x$ si $x \leq \frac{1}{2}$, sinon $2(1-x)$ | Fonction chapeau — série de Fourier |

---

## 2. Structure du dépôt

```
TP_TM/
│
├── run_all.py                   ← Point d'entrée : génère toutes les figures
│
├── src/
│   ├── schema_explicite.py      ← Schéma FTCS        (ordre 1 temps, ordre 2 espace)
│   ├── schema_implicite.py      ← Schéma implicite   (ordre 1 temps, ordre 2 espace)
│   └── schema_crank_nicolson.py ← Schéma CN          (ordre 2 temps, ordre 2 espace)
│
├── figures/                     ← Figures PNG générées automatiquement
│   ├── comparaison_f1.png
│   ├── comparaison_f2.png
│   ├── evolution_f1.png
│   ├── evolution_f2.png
│   ├── convergence_f1.png
│   └── convergence_f2.png
│
├── rapport_TP_TM.tex            ← Rapport complet en LaTeX
└── README.md
```

---

## 3. Installation et utilisation

### Prérequis

```bash
pip install numpy matplotlib scipy
```

### Générer toutes les figures d'un coup

```bash
# Depuis le dossier TP_TM/
python run_all.py
```

Ce script lance les 3 schémas pour $f_1$ et $f_2$, affiche les erreurs dans le terminal et sauvegarde 6 figures dans `figures/`.

### Lancer un schéma individuellement

```bash
cd src/
python schema_explicite.py
python schema_implicite.py
python schema_crank_nicolson.py
```

### Compiler le rapport LaTeX

```bash
# Depuis le dossier TP_TM/ (obligatoire pour trouver figures/)
pdflatex rapport_TP_TM.tex
pdflatex rapport_TP_TM.tex   # 2e passe pour la table des matières
```

---

## 4. Solutions analytiques

### Méthode : séparation des variables

On pose $u(x,t) = X(x) \cdot T(t)$, ce qui conduit aux valeurs propres :

$$\lambda_n = n^2\pi^2 + \alpha, \quad n = 1, 2, 3, \ldots$$

avec les fonctions propres $X_n(x) = \sin(n\pi x)$.

---

### Cas $f_1(x) = \sin(2\pi x)$

$f_1$ est exactement le mode propre $n = 2$. La solution est :

$$\boxed{u(x,t) = \sin(2\pi x)\cdot e^{-(4\pi^2+\alpha)\,t}}$$

Le profil sinusoïdal se conserve en forme et s'amortit exponentiellement avec le taux $\lambda_2 = 4\pi^2 + 1 \approx 40.5$.

---

### Cas $f_2(x)$ — fonction chapeau

$f_2$ nécessite un développement en série de Fourier. Les coefficients sont :

$$c_n = \frac{8}{n^2\pi^2}\,\sin\!\left(\frac{n\pi}{2}\right) = \begin{cases} 0 & n \text{ pair} \\ \dfrac{8\,(-1)^{(n-1)/2}}{n^2\pi^2} & n \text{ impair} \end{cases}$$

La solution exacte est la série :

$$\boxed{u(x,t) = \sum_{\substack{n=1 \\ n \;\text{impair}}}^{+\infty} \frac{8\,(-1)^{(n-1)/2}}{n^2\pi^2}\,\sin(n\pi x)\; e^{-(n^2\pi^2+\alpha)\,t}}$$

> Les hautes fréquences s'amortissent très vite. Pour les grands $t$, seul le mode fondamental $n=1$ subsiste.

---

## 5. Schémas numériques

### Discrétisation commune

| Variable | Grille | Pas |
|----------|--------|-----|
| Espace | $x_j = j\,\Delta x$, $j = 0, \ldots, N$ | $\Delta x = 1/N$ |
| Temps | $t_n = n\,\Delta t$ | $\Delta t$ choisi selon le schéma |
| Notation | $u_j^n \approx u(x_j, t_n)$ | — |
| Nombre de Fourier | $r = \Delta t / \Delta x^2$ | critère de stabilité |

Conditions de Dirichlet : $u_0^n = u_N^n = 0$ pour tout $n$.

---

### Schéma 1 — Explicite (FTCS)

**Formule** :

$$u_j^{n+1} = r\,u_{j-1}^n + (1 - 2r - \alpha\Delta t)\,u_j^n + r\,u_{j+1}^n$$

**Forme matricielle** : $\mathbf{u}^{n+1} = A\,\mathbf{u}^n$ avec $A = \text{tridiag}(r,\; 1-2r-\alpha\Delta t,\; r)$

**Stabilité (Von Neumann)** : facteur d'amplification $g = 1 - 4r\sin^2(\xi/2) - \alpha\Delta t$

$$\boxed{r \leq \frac{1}{2} \quad \Longrightarrow \quad \Delta t \leq \frac{\Delta x^2}{2}} \quad \text{(conditionnellement stable)}$$

---

### Schéma 2 — Implicite

**Formule** :

$$-r\,u_{j-1}^{n+1} + (1+2r+\alpha\Delta t)\,u_j^{n+1} - r\,u_{j+1}^{n+1} = u_j^n$$

**Forme matricielle** : $A\,\mathbf{u}^{n+1} = \mathbf{u}^n$ avec $A = \text{tridiag}(-r,\; 1+2r+\alpha\Delta t,\; -r)$

**Stabilité (Von Neumann)** : $g = \dfrac{1}{1 + 4r\sin^2(\xi/2) + \alpha\Delta t} \leq 1$ toujours

$$\boxed{\text{Inconditionnellement stable — aucune contrainte sur } \Delta t}$$

---

### Schéma 3 — Crank-Nicolson

Moyenne du schéma explicite (temps $n$) et implicite (temps $n+1$).

**Système** : $A\,\mathbf{u}^{n+1} = B\,\mathbf{u}^n$ avec :

$$A = \text{tridiag}\!\left(-\tfrac{r}{2},\; 1+r+\tfrac{\alpha\Delta t}{2},\; -\tfrac{r}{2}\right) \qquad B = \text{tridiag}\!\left(\tfrac{r}{2},\; 1-r-\tfrac{\alpha\Delta t}{2},\; \tfrac{r}{2}\right)$$

**Stabilité (Von Neumann)** :

$$g = \frac{1 - 4r\sin^2(\xi/2) - \alpha\Delta t/2}{1 + 4r\sin^2(\xi/2) + \alpha\Delta t/2} \;\Rightarrow\; |g| \leq 1 \text{ toujours}$$

$$\boxed{\text{Inconditionnellement stable — ordre 2 en temps ET en espace}}$$

---

### Comparaison des 3 schémas

| Propriété | Explicite | Implicite | Crank-Nicolson |
|-----------|:---------:|:---------:|:--------------:|
| Ordre en temps | $\mathcal{O}(\Delta t)$ | $\mathcal{O}(\Delta t)$ | $\mathcal{O}(\Delta t^2)$ ✅ |
| Ordre en espace | $\mathcal{O}(\Delta x^2)$ | $\mathcal{O}(\Delta x^2)$ | $\mathcal{O}(\Delta x^2)$ |
| Stabilité | Conditionnelle $r \leq \frac{1}{2}$ | Inconditionnelle | Inconditionnelle |
| Coût par pas | Faible | Moyen | Moyen |
| **Recommandé si** | Petit $\Delta t$ imposé | Stabilité requise | **Meilleur compromis** |

---

## 6. Résultats

Paramètres communs : $N = 100$, $\alpha = 1$, $T = 0.1$.

| Schéma | $\Delta t$ | $r$ | Erreur $f_1$ | Erreur $f_2$ |
|--------|:----------:|:---:|:------------:|:------------:|
| Explicite | $4 \times 10^{-5}$ | 0.40 | $3.5 \times 10^{-5}$ | $2.0 \times 10^{-5}$ |
| Implicite | $1 \times 10^{-4}$ | 1.00 | $1.7 \times 10^{-4}$ | $2.1 \times 10^{-4}$ |
| **Crank-Nicolson** | $1 \times 10^{-4}$ | 1.00 | $\mathbf{2.3 \times 10^{-5}}$ | $\mathbf{4.5 \times 10^{-5}}$ |

> **Observation clé :** Crank-Nicolson utilise le même $\Delta t$ que l'implicite (pas de contrainte de stabilité) mais atteint une précision similaire à l'explicite grâce à son ordre 2 en temps.

---

## 7. Théorème de Lax — Convergence

$$\boxed{\text{Convergence} \iff \text{Consistance} + \text{Stabilité}}$$

| Propriété | Explicite | Implicite | Crank-Nicolson |
|-----------|:---------:|:---------:|:--------------:|
| Consistance ($\|T^h\| \to 0$) | ✅ | ✅ | ✅ |
| Stabilité ($\|g\| \leq 1$) | ✅ sous $r \leq \frac{1}{2}$ | ✅ | ✅ |
| **Convergence** | ✅ | ✅ | ✅ |

Les 3 schémas convergent vers la solution exacte quand $\Delta t, \Delta x \to 0$.  
Les courbes de convergence (voir `figures/convergence_*.png`) confirment une décroissance en $\mathcal{O}(\Delta x^2)$.
