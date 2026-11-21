# Hello. My name is Himanshu Bhisikar 👋

## 📖 My Journey
Driven by a passion for astrophysics that started at age 10, my quest to understand how nature works led me to IISER Pune, where I specialized in Physics. Following extensive academic research stints at the Inter-University Centre for Astronomy and Astrophysics (IUCAA), and the Indian Institute of Astrophysics (IIA), Bengaluru, two of India's premier astrophysics institutes, I experienced firsthand how transformative scientific computing and machine learning are for modeling, predicting, and understanding complex physical systems. This realization sparked my transition into a Machine Learning Scientist—combining physics-informed neural networks, autograd constraints, and uncertainty calibration to bridge fundamental physical principles with modern data-driven intelligence.

---

## 🚀 Featured Scientific ML Projects

### 🛰️ [RADAR-HSI: Radiometric Analysis, Destriping & Automated Recognition for Hyperspectral Imagery](https://github.com/human1sh/RADAR-HSI)
- **Tech Stack:** `PyTorch` | `3D Spatial-Spectral CNNs` | `SciML` | `Radiometric Restoration` | `SLIC Superpixels & OBIA` | `scikit-learn`
- **Overview:** Engineered an end-to-end scientific machine learning pipeline for optical airborne remote sensing telemetry on the Pavia University dataset ($610 \times 340 \times 103$ datacube, $9$ land-cover classes), solving critical spatial data leakage, simulating forward optoelectronic flight noise, and discovering fine-grained intra-class material sub-clusters.
- **Physics & Formulation:** Formulated a forward flight sensor degradation chain embedding Poisson photon shot noise ($10^4\,e^-$ full-well capacity), pushbroom column detector gain non-uniformity ($\sigma = 0.015$), electronic readout noise, and 12-bit ADC quantization ($26.76\text{ dB}$ baseline SNR). Built a localized moving-window ($W = 31$) column moment-matching destriper paired with Savitzky-Golay spectral filtering to suppress striping artifacts while preserving narrow mineral and vegetative absorption bands.
- **Benchmarks & Results:** Quantified and proved a **$\sim 45\%$ spatial autocorrelation data leakage inflation** between naive random splitting and strict spatial block cuts ($95.3\% \rightarrow 50.8\%$). Engineered a joint 3D Spatial-Spectral CNN with guard-band spatial isolation achieving a **$+7.21\%$ generalization boost** over 1D spectral models, and combined SLIC superpixels with connected-component stratification to uncover **23 physical intra-class material sub-clusters** with zero class dropout.
- **Engineering:** Modular `src/` layout (`dataset`, `phase1_baseline`, `phase2_simulator`, `phase2_restoration`, `phase3_spatial_nn`, `phase4a_unconstrained_slic`, `phase4b_constrained_discovery`, `utils`, `visualize`), comprehensive `pytest` test suites with synthetic hyperspectral generator, strict PEP 621 packaging (`pyproject.toml`), and publication-grade multi-panel diagnostic graphics.

---

### ☀️ [CHAMP-PV: Constrained Hybrid Approach for Modeling Photovoltaics](https://github.com/human1sh/champ-pv)
- **Tech Stack:** `PyTorch` | `Physics-Informed ML (SciML)` | `Custom Autograd Penalties` | `HistGradientBoosting` | `Residual Learning` | `scikit-learn`
- **Overview:** Developed an end-to-end physics-constrained hybrid neural network framework for predicting daily utility-scale solar Photovoltaic (PV) Performance Ratio ($\text{PR}$) across $111\text{k}+$ operational plant days ($300+$ geographically distributed global PV installations). Specifically engineered to solve catastrophic out-of-distribution (OOD) extrapolation failure modes on aging, legacy solar assets where unconstrained deep learning and pure tree-based models violate thermodynamic and semi-conductor degradation physics.
- **Physics & Formulation:** Formulated a rigorous semi-empirical physical baseline embedding explicit algebraic formulations for technology-specific degradation rates ($d_{\text{tech}}$ for mono-Si, poly-Si, and thin-film CdTe), Faiman-King cell temperature dynamics ($T_{\text{cell}} = T_{\text{amb}} + \frac{G}{u_0 + u_1 \cdot v_{\text{wind}}}$), empirical dust/particulate soiling accumulation and precipitation-clearing cycles, and non-linear low-light diode recombination attenuation. Coupled this deterministic prior with a zero-initialized, bounded neural residual multiplier ($\hat{\text{PR}} = \text{PR}_{\text{phys}} \cdot [1 + 0.25 \cdot \tanh(\text{NN}(x))]$), guaranteeing physical safety bounds at initialization. Implemented a custom PyTorch autograd regularization penalty enforcing strict non-positive first-order derivative aging constraints ($\mathcal{L}_{\text{mono}} = \lambda \cdot \text{ReLU}\left(\frac{\partial \hat{\text{PR}}}{\partial \text{age}}\right)^2$, $\frac{\partial \text{PR}}{\partial \text{age}} \le 0$) during backpropagation.
- **Benchmarks & Results:** Completely eliminated physical degradation monotonicity violations ($\mathbf{18.7\% \rightarrow 0.0\%}$) under severe out-of-distribution age extrapolation stress tests ($\text{Train} < 8\text{ years}$, $\text{Test} \ge 10\text{ years}$), where standard unconstrained MLPs, Random Forests, and HistGradientBoosting suffered massive hallucinated efficiency gains. Simultaneously outperformed all purely empirical baselines in test-set RMSE ($0.0384$ vs. $0.0512$) on out-of-sample inverter blocks. Furthermore, transformed the residual magnitude $|\hat{\text{PR}} - \text{PR}_{\text{observed}}|$ into an automated, zero-shot anomaly detector for hardware degradation, achieving an outstanding **$\text{ROC-AUC} = 1.0000$** for inverter clipping, string outages, and catastrophic ground-fault disconnects.
- **Engineering:** Production-grade modular Python package (`models/hybrid_network.py`, `physics/pv_equations.py`, `losses/autograd_penalties.py`, `evaluation/ood_stress_testing.py`), fully vectorised NumPy/PyTorch batch routines, automated synthetic solar array degradation simulator for `pytest` unit test verification, and complete reproducibility scripts generating publication-ready residual diagnostic plots and ablation tables.

---

### 🛰️ [GEOSAR: Cross-Modal SAR & Optical Satellite Imagery Registration](https://github.com/human1sh/GEOSAR)
- **Tech Stack:** `Python` | `Rasterio & GDAL` | `OpenCV` | `GeoPandas & Shapely` | `PyTorch` | `ONNX & ONNX Runtime` | `Streamlit`
- **Overview:** Architected a high-precision, cross-modal georeferencing and feature-alignment pipeline for co-registering commercial X-band SAR (Umbra) and high-resolution optical imagery (Maxar) on the public SpaceNet 9 dataset, coupled with real INT8 edge quantization latency profiling for downstream multimodal sensor fusion.
- **Physics & Formulation:** Formulated a zero-leakage geospatial patch-reading pipeline centering on physical map coordinates (UTM 37N projection, $0.6\text{ m}$ common grid) to eliminate coordinate-frame mismatch. Developed an 8-bin unsigned oriented-gradient channel representation to provide invariance against the severe radiometric contrast reversals characteristic of SAR specular backscatter versus optical surface reflectance. Designed Padfield-style masked normalized cross-correlation (NCC) with morphological margin erosion ($5\sigma$) to actively prevent zero-padded raster borders from generating false correlation peaks.
- **Benchmarks & Results:** Cut median residual tie-point misregistration by **$63\%$** ($30.1\text{ m} \rightarrow 11.2\text{ m}$) on out-of-sample evaluation scenes ($02\_01$ and $03\_01$) with an $82\%$ improvement rate across $308$ real ground tie-points. Proved that classical SIFT feature matching totally fails under cross-modal sensor geometry (rejecting $100\%$ of patches due to low inlier ratios). Demonstrated static INT8 post-training quantization on a 4-channel multimodal fusion U-Net, achieving a **$2.3\times$ CPU inference speedup** ($291\text{ ms} \rightarrow 129\text{ ms}$) and **$73\%$ model compression** ($1.91\text{ MB} \rightarrow 0.52\text{ MB}$) with under $0.24\%$ relative logit drift.
- **Engineering:** Production-grade modular architecture (`src/geo`, `src/registration`, `src/evaluate`, `src/edge_benchmark`, `src/gis_export`), automated GeoPackage (`.gpkg`) vector layer exports for GIS/QGIS integration, synthetic SAR/optical pair generator with speckle simulation for `pytest` unit suites, and an interactive multi-tab `Streamlit` web interface for misalignment diagnostics and error distribution visualization.

---

### 🌌 [GUARD: Geomagnetic Upstream Alert & Risk Detector](https://github.com/human1sh/guard-dst-prediction)
- **Tech Stack:** `PyTorch` | `Temporal Convolutional Networks (TCN)` | `Bidirectional LSTM` | `Physics-Informed ML (SciML)` | `Custom Asymmetric Loss` | `NOAA / NASA OMNI Telemetry`
- **Overview:** Engineered an end-to-end, physics-guided deep learning space weather forecasting pipeline predicting Earth's disturbance storm time ($Dst$) index 1 hour ahead from upstream solar wind and interplanetary magnetic field (IMF) telemetry, providing critical real-time early warnings to safeguard power grids, satellite constellations, and high-frequency communication links against catastrophic coronal mass ejection (CME) impacts.
- **Physics & Formulation:** Formulated a ballistic upstream propagation alignment correcting for the variable solar wind convective transit delay ($\tau = X_{\text{L1}} / v_{\text{sw}}$) from the Sun-Earth Lagrange point 1 (L1) to the magnetopause bow shock. Explicitly derived and fed key magnetospheric energy transfer coupling functions into the feature space—including the Akasofu epsilon parameter ($\epsilon = v_{\text{sw}} B^2 \sin^4(\theta/2) l_0^2$), Newell universal merging rate ($d\Phi_{\text{MP}}/dt = v_{\text{sw}}^{4/3} B_T^{2/3} \sin^{8/3}(\theta/2)$), and Burton ring current injection-decay kinetics ($d(Dst^*)/dt = Q(t) - Dst^*/\tau_R$). Designed a specialized non-symmetric loss function ($\mathcal{L}_{\text{asym}} = \frac{1}{N} \sum (\hat{y} - y)^2 \cdot [1 + \alpha \cdot \mathbb{I}(y < \hat{y}) \cdot \mathbb{I}(y < -50\text{ nT})]$) to heavily penalize dangerous storm under-prediction during severe negative excursions where standard symmetric MSE collapses toward quiet-time baselines.
- **Benchmarks & Results:** Evaluated over multiple solar cycles of NASA OMNI-2 telemetry across quiet, moderate, and extreme space weather intervals. Achieved state-of-the-art ring current tracking with an overall **$R^2 = 0.845$**, Pearson correlation **$r = 0.920$**, and an **RMSE of $11.4\text{ nT}$**, thoroughly outperforming standard GRU, unconstrained LSTM, and empirical Burton ODE models. Achieved a **$100\%$ detection rate on severe and extreme geomagnetic storms** ($Dst \le -100\text{ nT}$, G3-G5 class events) with zero false-negative missed alerts on major historical events like the St. Patrick's Day and Halloween storms, while preserving accurate recovery-phase decay dynamics without unphysical rapid oscillations.
- **Engineering:** Production-grade modular layout (`dataset/omni_loader.py`, `models/tcn_lstm_hybrid.py`, `loss/asymmetric_storm_loss.py`, `eval/lead_time_metrics.py`), automated telemetry data cleaning and multi-satellite imputation pipelines, Cosine Annealing learning rate scheduling with warm restarts, unit testing suite with synthetic solar wind shock profiles, and automated generation of multi-panel storm event tracking and error distribution figures.

---

### 🛰️ [SPARTA: Spacecraft Physics & Astrodynamic Risk Trajectory Assessment](https://github.com/human1sh/SPARTA)
- **Tech Stack:** `PyTorch` | `Physics-Informed Neural Networks (SciML)` | `Autograd Jacobians` | `L-BFGS Temperature Scaling` | `PowerTransformer` | `Astrodynamics & SSA`
- **Overview:** Engineered a mission-critical, physics-regularized Scientific Machine Learning framework assessing close-approach collision risk ($P_{\text{risk}}$) and maneuver decision thresholds across $10,000+$ Low Earth Orbit (LEO) conjunction data messages (CDMs). Built to replace computationally prohibitive Monte Carlo orbital propagations with real-time, well-calibrated epistemic probabilities for autonomous space situational awareness (SSA) and satellite fleet collision avoidance.
- **Physics & Formulation:** Ingested 24+ astrodynamic parameters encompassing 3D relative encounter kinematics ($r_{\text{miss}}$, relative velocity $\vec{v}_{\text{rel}}$), projected combined positional covariance ellipsoids in the B-plane encounter frame ($\sigma_x, \sigma_y, \sigma_z$), atmospheric drag variations, and solar radiation pressure indices ($F_{10.7}$, $Ap$). Embedded strict astrodynamic physical priors directly into network training via custom PyTorch autograd Jacobians enforcing negative gradient monotonicity ($\frac{\partial P_{\text{risk}}}{\partial r_{\text{miss}}} \le 0$) through a targeted one-sided hinge penalty ($\mathcal{L}_{\text{mono}} = \lambda \cdot \text{ReLU}\left(\frac{\partial \hat{P}_{\text{risk}}}{\partial r_{\text{miss}}}\right)^2$), ensuring that increasing physical separation distance strictly decreases collision risk and preventing unphysical risk surges caused by sparse covariance sampling.
- **Benchmarks & Results:** Achieved stellar conjunction discrimination performance with **$\text{ROC-AUC} = 0.9818$** and an **Average Precision ($\text{PR-AUC}$) of $0.942$** on severe close-encounter test distributions. Solved extreme neural confidence overconfidence via post-hoc L-BFGS optimization of temperature scaling logits ($T = 1.8767$), driving Expected Calibration Error down to a near-perfect **$\text{ECE} = 0.011$** ($1.1\%$) and Brier score to $0.034$. Monotonicity regularization completely eradicated physical violation artifacts across the operational miss-distance continuum ($0.0\text{ km} \le r_{\text{miss}} \le 10.0\text{ km}$) while running forward inference in under **$1.8\text{ ms}$ per conjunction**—delivering a $10,000\times$ acceleration over numerical Foster/Akella orbital integration schemes.
- **Engineering:** Production-grade modular repository architecture (`src/data`, `src/loss`, `src/models`, `src/plots`, `src/predict`, `src/train`, `src/utils`), configuration-driven pipelines via hierarchical YAML schemas, robust feature preprocessing leveraging Yeo-Johnson `PowerTransformer` for highly skewed orbital metrics, automated `pytest` test suites verifying autograd gradient bounds on synthetic encounters, and full GitHub Actions CI automation for seamless code quality verification.

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
