# DCModeling
### Using the Z-Z-Z binary model to model diffusion coefficients in binary systems.

## Installation

### Install required packages

pip install -r requirements.txt

### Add support for ThermoCalc

1. Install ThermoCalc (with valid license connection)
   2. Install `tc-python` into preferred python environment
      - For further instructions, see the help file at 
        `C:/Program Files/Thermo-Calc/<version>/HTML5/content/installation/sdks/tc-python-install-advanced.htm`
      - In summary, start the ThermoCalc GUI once (with valid license), and a Python
        wheel will be created at a path such as `C:\Users\<YourUser>\Documents\Thermo-Calc\<version>\SDK\TC-Python\`
        (on Linux, this will be at `/home/YourUser/Thermo‑Calc/<version>/SDK/TC-Python`)
      - Install this wheel package into your environment using a command like the following:
        ```Windows or Linux
        pip install C:\Users\<YourUser>\Documents\Thermo-Calc\<version>\SDK\TC-Python\TC_Python-<version>-py3-none-any.whl
        ```

## Diffusion illustration

![binary_diffusion](https://github.com/Chuangye-Wang/DCModeling/blob/main/figures/DCModeling_binary_diffusion.png)

![ternary_diffusion](https://github.com/Chuangye-Wang/DCModeling/blob/main/figures/DCModeling_logo.png)

## Thermodynamic factor calculation.

![thermodynamic factor](https://github.com/Chuangye-Wang/DCModeling/blob/main/examples/FeNi/FeNi_Thermo-calc_vs_user-database.png)

