# Hello. My name is Himanshu Bhisikar 👋

## 📖 My Journey
Driven by a passion for astrophysics that started at age 10, my quest to understand how nature works led me to IISER Pune, where I specialized in Physics. Following extensive academic research stints at the Inter-University Centre for Astronomy and Astrophysics (IUCAA), and the Indian Institute of Astrophysics (IIA), Bengaluru, two of India's premier astrophysics institutes, I experienced firsthand how transformative scientific computing and machine learning are for modeling, predicting, and understanding complex physical systems. This realization sparked my transition into a Machine Learning Scientist—combining physics-informed neural networks, autograd constraints, and uncertainty calibration to bridge fundamental physical principles with modern data-driven intelligence.

---

## 🚀 Featured Scientific ML Projects

### ☀️ [CHAMP-PV: Constrained Hybrid Approach for Modeling Photovoltaics](https://github.com/human1sh/champ-pv)
- **Tech Stack:** `PyTorch` | `Physics-Constrained Neural Networks` | `Autograd Penalties` | `HistGradientBoosting` | `Residual Learning`
- **Overview:** Developed a physics-constrained hybrid neural network framework predicting daily solar Photovoltaic (PV) Performance Ratio ($\text{PR}$) across 111k+ operational days to solve out-of-distribution extrapolation on older, unseen panels.
- **Physics & Formulation:** Embedded explicit algebraic formulas for system efficiency ($\eta_{\text{sys}}$), technology degradation ($d_{\text{tech}}$), cell temperature dynamics ($T_{\text{cell}}$), particulate soiling, and low-light attenuation with a zero-initialized, bounded neural residual multiplier ($1 + 0.25 \cdot \tanh(\text{NN}(x))$) and an autograd monotonic aging penalty ($\partial \text{PR} / \partial \text{age} \le 0$).
- **Benchmarks & Results:** Completely eliminated physical aging monotonicity violations ($18.7\% \rightarrow 0.2\%$) under age extrapolation (train $<8\text{y}$, test $\ge 10\text{y}$), outperforming unconstrained MLPs and gradient boosting while leveraging residual magnitude for 100% inverter/string outage detection (**ROC-AUC = 1.0000**).

---

### 🌌 [GUARD: Geomagnetic Upstream Alert & Risk Detector](https://github.com/human1sh/guard-dst-prediction)
- **Tech Stack:** `PyTorch` | `TCN-LSTM Networks` | `Physics-Informed ML` | `Asymmetric Loss` | `Cosine Annealing`
- **Overview:** Built an end-to-end SciML pipeline forecasting Earth's geomagnetic $Dst$ index 1-hour ahead to provide real-time early warnings for geomagnetic storm threats.
- **Physics & Benchmarks:** Pre-aligned L1 solar wind drivers by ballistic travel time ($\tau = X_{\text{L1}} / v_{\text{sw}}$), integrated Akasofu/Newell coupling terms and Burton kinetics, and trained with an asymmetric loss function penalizing under-predicted storm dips ($R^2 = 0.845$, $r = 0.920$, 100% severe storm detection rate).
- **Engineering:** Modular `src/` package architecture (`dataset`, `model`, `loss`, `evaluate`), CLI configuration (`main.py`), and publication-quality diagnostic plotting.

---

### 🛰️ [SPARTA: Spacecraft Physics & Astrodynamic Risk Trajectory Assessment](https://github.com/human1sh/SPARTA)
- **Tech Stack:** `PyTorch` | `Physics-Informed Neural Networks` | `Autograd Constraints` | `Temperature Scaling` | `PowerTransformer`
- **Overview:** Engineered a production-grade Scientific Machine Learning framework evaluating satellite collision avoidance maneuvers across 10,000+ LEO conjunction events.
- **Physics & Calibration:** Enforced soft autograd gradient monotonicity constraints ($\partial P_{\text{risk}} / \partial r_{\text{miss}} \le 0$) to physically penalize non-decreasing collision risk against miss distance, combined with L-BFGS temperature scaling calibration ($T = 1.8767$, **ROC-AUC = 0.9818**, **ECE = 0.011**).
- **Engineering:** Multi-variate telemetry feature ingestion (24+ signals including covariance ellipsoids and space weather indices), modular `src/` layout (`data`, `loss`, `models`, `plots`, `predict`, `train`, `utils`), automated `pytest` test suites, YAML CLI workflows, and GitHub Actions CI integration.

---

## 🛠️ Tech Stack & Tooling

### 🤖 Scientific Machine Learning & Deep Learning
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![DeepXDE](https://img.shields.io/badge/DeepXDE-000000?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

* **Core Paradigms:** Physics-Informed Neural Networks (PINNs), Differentiable Physics Constraints, Attention ResUNets, CNN Feature Extractors, Temperature Scaling & Post-Hoc Probability Calibration, Computer Vision & Image Segmentation.

---

### 🔬 Scientific Computing & Data Analysis
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)

* **Methods:** High-dimensional Tensor Manipulation, Non-linear Optimization (L-BFGS, Adam), Signal Processing, Time-series Feature Engineering, Bayesian Likelihoods & Statistical Modeling.

---

### 🌌 Astrophysics, Astronomy, Image Processing, Physical animations
![Astropy](https://img.shields.io/badge/Astropy-0C55A5?style=for-the-badge&logo=python&logoColor=white)
![Astroquery](https://img.shields.io/badge/Astroquery-FFD43B?style=for-the-badge&logo=python&logoColor=black)
![Photutils](https://img.shields.io/badge/Photutils-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Specutils](https://img.shields.io/badge/Specutils-8CAAE6?style=for-the-badge&logo=python&logoColor=white)
![Manim](https://img.shields.io/badge/Manim_Community-5C5C5C?style=for-the-badge&logo=python&logoColor=white)

* **Domain Expertise:** 3D Integral Field Spectroscopy (IFS/IFU Data Cubes), Multi-Wavelength Synthesis (AstroSat/UVIT, DECaLS, ALMA CO), Aperture Photometry, Astronomical Image Reduction, Programmatic Physics Animations (*fysyx* youtube channel).

---

### 💻 Developer Environment, OS & CI/CD
![Ubuntu](https://img.shields.io/badge/Ubuntu_26.04-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Linux Terminal](https://img.shields.io/badge/Zsh_&_Ptyxis-4EAA25?style=for-the-badge&logo=gnu-bash&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Zed](https://img.shields.io/badge/Zed_Editor-000000?style=for-the-badge&logo=zed&logoColor=white)

* **Engineering Practices:** Modular Python Package Architecture (`src/` layout), Automated Integration & Regression Testing (`pytest`), Configuration-Driven Training (`pyproject.toml`, YAML configs), Continuous Integration (CI) Actions.

---

## 📊 GitHub Highlights

![Himanshu's GitHub Stats](https://github-readme-stats.vercel.app/api?username=human1sh&show_icons=true&theme=tokyonight&cache_bust=1)
![Top Languages](https://github-readme-stats.vercel.app/api/top-langs/?username=human1sh&layout=compact&theme=tokyonight)
![Himanshu's Streak](https://github-readme-streak-stats.herokuapp.com/?user=human1sh&theme=tokyonight)

---

📫 **Connect with me:** [GitHub](https://github.com/human1sh)


<!--
**human1sh/human1sh** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
