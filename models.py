import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import *
from help_functions import *


class Model:
    def __init__(self):
        """
        To initialize the EndMemberModel class.
        """
        # DataFrame to keep the experimental data
        self.data = None
        self.interaction_parameters = None

    def load_interaction_parameters(self):
        pass

    def thermodynamic_factor(self, database_mode="json"):
        if database_mode.lower() == "json":
            thermodynamic_factor_user_defined(self.interaction_parameters)
        elif database_mode.lower() == "calphad":
            thermodynamic_factor_calphad_engine(self.data)


class EndMemberModel:
    """ The model for predicting self and impurity diffusion coefficients using Arrhenius equation.
    """

    def __init__(self):
        """ To initialize the EndMemberModel class.
        """
        # diffusion coefficients data
        self.diffusion_coef = None
        # temperature data
        self.temp_celsius = None
        self.temp_kelvin = None
        # the pre_factor in arrhenius equation
        self.pre_factor = None
        # the activation energy in arrhenius equation
        self.activ_energy = None

    def load_data(self, filename):
        """ To load the data for diffusion coefficients and temperature (in kelvin)

        Args:
            filename: A string for the path to the datafile.

        Returns:
            None
        """
        if file_path:
            data = pd.read_csv(file_path)
            self.diffusion_coef = data["D_exp"]
            if "T_K" in data.columns:
                self.temp_celsius = data["T_K"] - CELSIUS_KELVIN_OFFSET
                self.temp_kelvin = data["T_K"]
            elif "T_C" in data.columns:
                self.temp_celsius = data["T_C"]
                self.temp_kelvin = data["T_C"] + CELSIUS_KELVIN_OFFSET

    def fitting(self, mode=1):
        """ To fit the arrhenius equation D = D_0 * exp(-Q/R/T) or D = D_0 * exp(-Q_0/R/T) + D_1 * exp(-Q_1/R/T)

        Args:
            mode: 1 or 2; 1 means using 1 pre_factor and 1 activation energy in the equation;
                2 means using 2 pre_factors and 2 activation energy in the equation

        Returns:
            None
        """
        if mode == 1:
            temp_inv = 1 / self.temp_kelvin
            log_coefs = np.log(self.diffusion_coef)
            slope, intercept = np.polyfit(temp_inv, log_coefs, 1)
            self.pre_factor = np.exp(intercept)
            self.activ_energy = - slope * GAS_CONSTANT
            print(self.pre_factor, self.activ_energy)
        else:
            pass

    def plot(self, log_scale=True, base=10):
        """ To plot the diffusion coefficients vs temperature.

        Args:
            log_scale: A bool indicating whether using log scale for y-axis. True to use and False to not use.
            base: An int standing for the base of log function.

        Returns:
            None
        """
        temp_inv = 10000 / self.temp_kelvin
        diffusion_coef_pred = arrhenius(self.pre_factor, self.activ_energy, self.temp_kelvin)
        fig, ax = plt.subplots()
        ax.scatter(temp_inv, self.diffusion_coef, color="red", marker="o")
        ax.plot(temp_inv, diffusion_coef_pred, color="black")
        if log_scale:
            ax.set_yscale("log", basey=base)
            ax.set_ylabel("log D (m$^2$/s)")
        else:
            ax.set_ylabel("D (m$^2$/s)")
        ax.set_xlabel("10000/T ($K^{-1})$")
        plt.show()


if __name__ == "__main__":
    model = EndMemberModel()
    file_path = "./examples/impurity_data_Ag_in_Cu.csv"
    model.load_data(file_path)
    model.fitting()
    print(model.pre_factor, model.activ_energy)
    model.plot()
