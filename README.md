# Electrons-non-thermiques-dans-un-plasma-astrophysique

Projet de physique numérique (M1) — Université Paris Cité  
**Encadrant :** Matteo Cerruti  

## Description du projet
Ce projet vise à résoudre numériquement l'équation d'évolution de la distribution en énergie $N(\gamma, t)$ d'électrons relativistes dans un plasma astrophysique soumis à un refroidissement synchrotron et un échappement depuis zone d'émission.

## Équation maîtresse
$$\frac{\partial N(\gamma, t)}{\partial t} = - \frac{\partial}{\partial \gamma} \left[ \dot{\gamma}_{\text{cool}} N(\gamma, t) \right] + Q(\gamma) - \frac{N(\gamma, t)}{t_{\text{esc}}}$$

Sous l'hypothèse $u_{\text{soft}} = 0$, le terme de refroidissement est  donc seulement dû au rayonnement synchrotron :
$$\dot{\gamma}_{\text{cool}} = - \beta \gamma^2 = - \frac{4}{3} \frac{\sigma_T c}{m_e c^2} u_B \gamma^2$$

## Structure du dépôt
- `src/` : Modules Python contenant les constantes physiques, les termes de flux et les schémas numériques (explicite, implicite).
- `notebooks/` : Notebooks Jupyter de démonstration, de validation des régimes analytiques et de tracé des figures.
- `docs/` : Notes théoriques et préparation du rapport.

## Références bibliographiques
1. **G. Ghisellini (2013)**, *Radiative Processes in High Energy Astrophysics*, Springer.  
   [Article arXiv (abs/1202.5949)](https://arxiv.org/abs/1202.5949)
2. **S. Inoue & F. Takahara (1996)**, *ApJ 463, 555*.  
   [PDF ADS Harvard](https://articles.adsabs.harvard.edu/pdf/1996ApJ...463..555I)