from constants import GAS_CONSTANT


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
