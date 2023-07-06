import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from constants import *
from help_functions import *


class DiffusivityData:
    """
    A class include all the information needed to evaluate the diffusion model based on the experimental diffusion
    coefficients data.

    Attributes:
        data: A DataFrame to store the experimental data.
        elements: A list of elements in the system.
        system: A string for naming the system.
        phase: A string for the structure in the system.
        thermodynamic_interaction_parameters: A dict including thermodyanmic interaction parameters that will be used
            to calculate the thermodynamic factor.
        end_dc: A dict for the diffusion coefficients data of the four end members (impurity and self-diffusion).
        unit: A dict storing units for the data.
    """

    def __init__(self, elements: list, phase: str = "FCC_A1"):
        """
        To initialize the DiffusivityData class.
        Args:
            elements: A list of elements in the system.
            phase: A string for the structure in the system.
        """
        # DataFrame to store the experimental data
        self.data = None
        elements.sort()
        self.elements = elements
        self.system = "".join(elements)
        self.phase = phase
        self.thermodynamic_interaction_parameters = None
        self.end_dc = None

        # unit
        self.unit = {"diffusion coefficient": "m^2/s"}

    def load_data_from_excel(self, datafile):
        """ To load the diffusion coefficient data from Excel format.

        Args:
            datafile: A string for the path to the datafile.
            # process_data: A bool value for preprocessing loaded data before use it.

        Returns:
            None.

        Raises: TypeError when the format of datafile is not included.
        """
        if datafile.endswith('.xlsx'):
            self.data = pd.read_excel(datafile, sheet_name=self.system)
        elif datafile.endswith('.csv'):
            self.data = pd.read_csv(datafile)
        else:
            raise TypeError("The extension of the file should be .csv or .xlsx.")

        if "Weight" in self.data.columns():
            self.data.dropna(axis=0, subset=["Weight"], inplace=True)
        # if process_data:
        if "A_mp" in self.data.columns:
            self.data["comp_A_mf"] = self.data["A_mp"] / 100
            self.data["comp_B_mf"] = 1 - self.data["comp_A_mf"]

        if "T_C" in self.data.columns:
            self.data["temp_celsius"] = self.data["T_C"]
            self.data["temp_kelvin"] = self.data["temp_celsius"] + CELSIUS_KELVIN_OFFSET

    def load_interaction_parameters(self, datafile):
        """ To load the thermodynamic interaction parameters in the Gibbs energy function.

        Args:
            datafile: A string for the path to the datafile.

        Returns:
            None.
        """
        with open(datafile) as file:
            json_data = json.load(file)
        self.thermodynamic_interaction_parameters = json_data.get(self.system).get(self.phase)

    def end_member_calc(self, datafile):
        """
        To load the end member (self and impurity diffusion coefficients) data from json files.
        Args:
            datafile: A string for the directory path to the datafile.

        Returns:
            None
        """
        self.end_dc = end_member_diffusion_coefs(self.elements, datafile, self.data.temp_kelvin)

    def set_weight(self, weights=dict()):
        """
        To set the weights for literature.
        Args:
            weights: A dictionary with (literature, weight) pair information.

        Returns:
            None.
        """
        for literature, weight in weights.items():
            self.data.loc[self.data["Literature"] == literature, "Weight"] = weight

    def thermodynamic_factor_calc(self, database_mode="json", database="TCNI11", engine="Thermo-Calc"):
        """
        To calculate thermodynamic factor in two different ways.
        Args:
            database_mode: A string indicating the format of data file source. ("json", "calphad")
            database: A string indicating the database. It is a path to the database file when using self-defined
                database, and it is a database name according to the selected CALPHAD engine.

        Returns:
            None.
        """
        if database_mode.lower() == "json":
            if self.thermodynamic_interaction_parameters is None:
                raise ValueError("Please define thermodynamic_interaction_parameters first!")
            self.data["TF"] = thermodynamic_factor_user_defined(self.thermodynamic_interaction_parameters,
                                                                self.data["comp_A_mf"], self.data["comp_B_mf"],
                                                                self.data["temp_kelvin"])
        elif database_mode.lower() == "calphad":
            self.data["TF"] = thermodynamic_factor_calphad_engine(self.data, elements=self.elements, database=database,
                                                                  phase=self.phase, engine=engine)

    def diffusion_coefs_calc(self, coefs):
        """
        To calculate the diffusion coefficients using the diffusion model.
        Returns:
            None.
        """
        tracer_dc1, tracer_dc2 = tracer_diffusion_coefs(coefs, self.data.comp_A_mf,
                                                        self.data.temp_kelvin,
                                                        self.end_dc)
        intrinsic_dc1 = tracer_dc1 * self.data.TF
        intrinsic_dc2 = tracer_dc2 * self.data.TF
        inter_dc = \
            self.data.comp_A_mf * intrinsic_dc2 + \
            self.data.comp_B_mf * intrinsic_dc1
        diffusion_types = pd.get_dummies(self.data.Dtype)
        diffusion_elements = pd.get_dummies(self.data.Element)

        diffusion_coefs = \
            diffusion_types.get("DC", 0) * inter_dc + \
            diffusion_types.get("DT", 0) * diffusion_elements.get("A", 0) * tracer_dc1 + \
            diffusion_types.get("DT", 0) * diffusion_elements.get("B", 0) * tracer_dc2 + \
            diffusion_types.get("DI", 0) * diffusion_elements.get("A", 0) * intrinsic_dc1 + \
            diffusion_types.get("DI", 0) * diffusion_elements.get("B", 0) * intrinsic_dc2

        return diffusion_coefs


class EndMemberData:
    """ A class for predicting self and impurity diffusion coefficients using Arrhenius equation.

    Attributes:
        diffusion_coef: A DataFrame to store the experimental data.
        temp_celsius: An array or pd.Series for the temperature (in celsius).
        temp_kelvin: An array or pd.Series for the temperature (in kelvin).
        pre_factor: Float or list-based for pre factor in Arrhenius equation.
            float for D = D_0 * exp(-Q/R/T);
            list-based for D = D_0 * exp(-Q_0/R/T) + D_1 * exp(-Q_1/R/T).
        activ_energy: Float or list-based for activation energy in Arrhenius equation.
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

    def load_data(self, file_path):
        """ To load the data for diffusion coefficients and temperature (in kelvin)

        Args:
            file_path: A string for the path to the datafile.

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
            else:
                raise ValueError("No T_K or T_C column in the datafile. So the temperature data is not clear.")

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
    model = EndMemberData()
    file_p = "./examples/impurity_data_Ag_in_Cu.csv"
    model.load_data(file_p)
    model.fitting()
    print(model.pre_factor, model.activ_energy)
    model.plot()
