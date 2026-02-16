# DCModeling

Using the Z-Z-Z binary model to model diffusion coefficients in binary systems.

## Table of Contents

- [Equations](#equations)
  - [End Member Diffusion Coefficients](#end-member-diffusion-coefficients)
  - [Gibbs Energy](#gibbs-energy)
  - [Thermodynamic Factor](#thermodynamic-factor)
  - [Tracer Diffusion Coefficients](#tracer-diffusion-coefficients)
  - [Intrinsic Diffusion Coefficients](#intrinsic-diffusion-coefficients)
  - [Inter-diffusion Coefficient (Darken Equation)](#inter-diffusion-coefficient-darken-equation)
  - [Objective Function](#objective-function)
- [Results](#results)
- [Installation](#installation)

## Equations

For a binary system with A and B elements.

### End Member Diffusion Coefficients

**Arrhenius equation** (single-term):

$$D = D_0 \exp\!\left(-\frac{Q}{RT}\right)$$

Two-term form:

$$D = D_0 \exp\!\left(-\frac{Q_0}{RT}\right) + D_1 \exp\!\left(-\frac{Q_1}{RT}\right)$$

**Brown-Ashby correlation** (structure-dependent estimation):

$$D = D_0 \exp\!\left(-K \frac{T_m}{T}\right)$$

where $D_0$ and $K$ are constants that depend on crystal structure (FCC, BCC, HCP, etc.) and $T_m$ is the melting temperature.

### Gibbs Energy

$$G = G^{\text{mech}} + G^{\text{ideal}} + G^{\text{excess}} + G^{\text{mag}}$$

**Mechanical mixing:**

$$G^{\text{mech}} = x_A \, G_A + x_B \, G_B$$

**Ideal mixing:**

$$G^{\text{ideal}} = RT\!\left(x_A \ln x_A + x_B \ln x_B\right)$$

**Excess (Redlich-Kister):**

$$G^{\text{excess}} = x_A \, x_B \sum_k L_k \left(x_A - x_B\right)^k$$

**Magnetic contribution:**

$$G^{\text{mag}} = RT \ln(\beta + 1) \, f(\tau)$$

where $\tau = T / T_c$, $T_c$ is the Curie temperature, $\beta$ is the Bohr magneton number, and $f(\tau)$ is the Hillert-Jarl function with structure-dependent parameter $p$ (0.28 for FCC/HCP, 0.4 for BCC).

### Thermodynamic Factor

$$\Psi = \frac{x_A \, x_B}{RT} \frac{\partial^2 G}{\partial x^2}$$

### Tracer Diffusion Coefficients

$$D^{*}_{A} = \exp\!\left(x_A \ln D_{AA} + x_B \ln D_{AB} + \frac{\Phi_A \, x_A \, x_B}{RT}\right)$$

$$D^{*}_{B} = \exp\!\left(x_A \ln D_{BA} + x_B \ln D_{BB} + \frac{\Phi_B \, x_A \, x_B}{RT}\right)$$

where $D_{AA}$, $D_{AB}$, $D_{BA}$, $D_{BB}$ are end member diffusion coefficients (self- and impurity diffusion), and $\Phi$ is the interaction parameter. Supported model variants:

| Model | Parameters |
|-------|-----------|
| 1-para | $\Phi_A = \Phi_B = \Phi$ |
| 2-para | $\Phi_A$, $\Phi_B$ independent |
| 4-para | $\Phi_A = a_0 + a_1 T$, $\Phi_B = b_0 + b_1 T$ |

### Intrinsic Diffusion Coefficients

$$D^{I}_{A} = \Psi \cdot D^{*}_{A}, \qquad D^{I}_{B} = \Psi \cdot D^{*}_{B}$$

where $\Psi$ is the thermodynamic factor.

### Inter-diffusion Coefficient (Darken Equation)

$$\tilde{D} = x_B \, D^{I}_{A} + x_A \, D^{I}_{B}$$

### Objective Function

**Weighted mean squared error:**

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left[w_i \ln\!\left(\frac{D_i^{\text{pred}}}{D_i^{\text{exp}}}\right)\right]^2$$

## Results

### Thermodynamic factor calculation

![thermodynamic factor](https://github.com/Chuangye-Wang/DCModeling/blob/main/examples/FeNi/FeNi_Thermo-calc_vs_user-database.png)

## Installation

### Install required packages

```shell
pip install -r requirements.txt
```

### Add support for ThermoCalc

- Install ThermoCalc (with valid license connection)
   - Install `tc-python` into preferred python environment
      - For further instructions, see the help file at
        `C:/Program Files/Thermo-Calc/<version>/HTML5/content/installation/sdks/tc-python-install-advanced.htm`
      - In summary, start the ThermoCalc GUI once (with valid license), and a Python
        wheel will be created at a path such as `C:\Users\<YourUser>\Documents\Thermo-Calc\<version>\SDK\TC-Python\`
        (on Linux, this will be at `/home/YourUser/Thermo‑Calc/<version>/SDK/TC-Python`)
      - Install this wheel package into your environment using a command like the following:
        ```shell
        pip install C:\Users\<YourUser>\Documents\Thermo-Calc\<version>\SDK\TC-Python\TC_Python-<version>-py3-none-any.whl
        ```
