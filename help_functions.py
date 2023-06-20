import json
import os

import numpy as np
from tc_python import TCPython

from constants import *


def arrhenius(pre_factor, activ_energy, temp_kelvin):
    """ To calculate the diffusion coefficients using Arrhenius equation.

    Args:
        pre_factor: Float or list-based for pre factor in Arrhenius equation.
            float for D = D_0 * exp(-Q/R/T);
            list-based for D = D_0 * exp(-Q_0/R/T) + D_1 * exp(-Q_1/R/T).
        activ_energy: Float or list-based for activation energy in Arrhenius equation.
        temp_kelvin: An array or pd.Series for the temperature (in kelvin).

    Returns:
        An array or pd.Series for predicted diffusion coefficients.
    """
    # if isinstance(temp_kelvin, list):
    #     temp_kelvin = np.array(temp_kelvin)
    if isinstance(pre_factor, (float, int)):
        return pre_factor * np.exp(- activ_energy / GAS_CONSTANT / temp_kelvin)
    else:
        if len(pre_factor) != 2 or len(activ_energy) != 2:
            raise ValueError("Check the length of pre_factor and activ_energy.")
        pf1, pf2 = pre_factor
        ae1, ae2 = activ_energy
        return pf1 * np.exp(-ae1 / GAS_CONSTANT / temp_kelvin) + pf2 * np.exp(-ae2 / GAS_CONSTANT / temp_kelvin)


def end_member_diffusion_coefs(system: str, datafile: str, temp_kelvin):
    """
    To read the pre factor and activation energy for end members from json datafile.
    Args:
        system: A string for the system.
        datafile: A string for the path to the datafile.
        temp_kelvin: An array or pd.Series for temperature data.

    Returns:
        A dict containing the calculated diffusion coefficients of end members.
    """
    with open(datafile) as file:
        dict_data = json.load(file).get(system)
    end_dc = {}
    for member, coefs in dict_data.items():
        # define it as a function of temperature.
        end_dc[member] = arrhenius(coefs.get("D0"), coefs.get("Q"), temp_kelvin)

    return end_dc


def tracer_diffusion_coefs(model_params, comp1_mf, temp_kelvin, end_dc):
    """
    To calculate tracer diffusion coefficients.
    Args:
        model_params: A list of parameters in the diffusion model.
        comp1_mf: An array-like type with composition information in it.
        temp_kelvin: An array-like with temperature information in it.
        end_dc: A dict-like with four end members' diffusion coefficient data.

    Returns:
        An array-like containing calculated tracer diffusion coefficients.
    """
    comp2_mf = 1 - comp1_mf
    interaction_expr_1, interaction_expr_2 = 0, 0
    if len(model_params) == 0 or model_params is None:
        pass
    elif len(model_params) == 1:
        interaction_expr_1, interaction_expr_2 = model_params[0], model_params[0]
    elif len(model_params) == 2:
        interaction_expr_1, interaction_expr_2 = model_params[0], model_params[1]
    elif len(model_params) == 4:
        interaction_expr_1 = model_params[0] + model_params[1] * temp_kelvin
        interaction_expr_2 = model_params[2] + model_params[3] * temp_kelvin
    else:
        raise ValueError("The size of model_params is not correct.")

    dc_1 = np.exp(comp1_mf * np.log(end_dc.get("AA")) + comp2_mf * np.log(end_dc.get("AB"))
                  + interaction_expr_1 * comp1_mf * comp2_mf / GAS_CONSTANT / temp_kelvin)

    dc_2 = np.exp(comp1_mf * np.log(end_dc.get("BA")) + comp2_mf * np.log(end_dc.get("BB"))
                  + interaction_expr_2 * comp1_mf * comp2_mf / GAS_CONSTANT / temp_kelvin)

    return dc_1, dc_2


def intrinsic_diffusion_coefs(model, comp1_mf, temp_kelvin, thermodynamic_factor, end_dc):
    dc_1, dc_2 = tracer_diffusion_coefs(model, comp1_mf, temp_kelvin, end_dc)

    return thermodynamic_factor * dc_1, thermodynamic_factor * dc_2


def darken(model, comp1_mf, temp_kelvin, thermodynamic_factor, end_dc):
    intrinsic_d_1, intrinsic_d_2 = intrinsic_diffusion_coefs(model, comp1_mf, temp_kelvin, thermodynamic_factor, end_dc)

    return comp1_mf * intrinsic_d_1 + (1 - comp1_mf) * intrinsic_d_2


def mean_square_error(y, y_pred, weight):
    """
    To calculate the mean square error with weight.
    Args:
        y: An array-like for original data
        y_pred: An array-like for predicted data.
        weight: An array-like for the weight assigned on each data sample.

    Returns:
        A float of the mean square error.
    """
    return 0.5 * np.sum((np.log(y_pred / y) * weight)**2)


def thermodynamic_factor_calphad_engine(data, elements: list, database: str, phase="FCC_A1", engine="Thermo-Calc"):
    """ To calculate the thermodynamic factor using CALPHAD engine.

    Args:
        data: A DataFrame storing the data information.
        elements: A list including two elements in the binary system.
        database: A user-constructed or literature database or using Thermo-Calc owned databases.
        phase: A string indicating the phase of the diffusion system.
        engine: A string defining the engine used.

    Returns:
        Array or pd.Series, calculated thermodynamic factor.
    """
    if engine == "Thermo-Calc":
        poly_expression = 'enter-symbol function TF=x(' + elements[0] + ')/8.31451/T*mur(' + elements[0] + ').x(' + \
                          elements[0] + ');,,,,'
        list_of_conditions = [[('T', temp_kelvin), ('X(' + elements[0] + ')', comp_mole_frac)]
                              for temp_kelvin, comp_mole_frac in data[['T_K', 'A_mf']].values]
        with TCPython() as session:
            calculation = (
                session
                # .set_cache_folder(os.path.basename(__file__) + "_cache")
                .select_database_and_elements(database, elements)
                .without_default_phases()
                .select_phase(phase)
                .get_system()
                .with_batch_equilibrium_calculation()
                .run_poly_command(poly_expression)
                .set_condition(list_of_conditions[0][0][0], list_of_conditions[0][0][1])
                .set_condition(list_of_conditions[0][1][0], list_of_conditions[0][1][1])
                .disable_global_minimization())

        calculation.set_conditions_for_equilibria(list_of_conditions)
        # calculate the thermodynamic factor
        results = calculation.calculate(['TF'])

        return results.get_values_of('TF')


def thermodynamic_factor_user_defined(interaction_parameters: dict, comps_1, comps_2, temp_kelvin):
    """ To calculate the thermodynamic factor according to the definition of it.

    Args:
        interaction_parameters: A dict for interaction parameter.
        comps_1: An array or pd.Series for the composition of element A in mole fraction.
        comps_2: An array or pd.Series for the composition of element B in mole fraction.
        temp_kelvin: An array or pd.Series for the temperature data.

    Returns:
        An array or pd.Series, calculated thermodynamic factor.
    """
    item1 = 0
    item2 = 0
    for order, expression in interaction_parameters.items():
        # func_of_temp: a function of temperature
        func_of_temp = lambda T: eval(expression)
        interaction_param_values = func_of_temp(temp_kelvin)
        # k: order of interaction parameter
        k = int(order[1:])
        # first_term
        item1 += (2 * k + 1) * interaction_param_values * (comps_1 - comps_2) ** k
        # second_term
        if k >= 2:
            item2 += k * (k - 1) * interaction_param_values * (comps_1 - comps_2) ** (k - 2)

    return 1 - 2 * comps_1 * comps_2 / GAS_CONSTANT / temp_kelvin * (item1 - 2 * comps_1 * comps_2 * item2)
