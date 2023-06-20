import pandas as pd

from help_functions import *
from scipy.optimize import least_squares, minimize


class Optimizer:
    """
    An optimizer to optimize interaction parameters in the diffusion model.
    """

    def __init__(self, diffusivity_data, model='1-para', method="least_squares"):
        self.diffusivity_data = diffusivity_data
        self.model = model
        # self.model_structure = model_structure
        # initialize model parameters
        self.init_params = np.random.random(int(self.model.split("-")[0]))
        # initialize which method to use for optimization.
        if method not in ("least_squares", "minimize"):
            raise ValueError("The method should be either least_squares or minimize.")
        self.method = method
        self.optimized_results = {"OptimizedResult": None,
                                  "mse": None,
                                  "optimized_params": []}

    def diffusion_coefs_calc(self, coefs):
        """
        To calculate the diffusion coefficients using the diffusion model.
        Returns:
            None.
        """
        tracer_dc1, tracer_dc2 = tracer_diffusion_coefs(coefs, self.diffusivity_data.data.comp_A_mf,
                                                        self.diffusivity_data.data.temp_kelvin,
                                                        self.diffusivity_data.end_dc)
        intrinsic_dc1 = tracer_dc1 * self.diffusivity_data.data.TF
        intrinsic_dc2 = tracer_dc2 * self.diffusivity_data.data.TF
        inter_dc = \
            self.diffusivity_data.data.comp_A_mf * intrinsic_dc2 + \
            self.diffusivity_data.data.comp_B_mf * intrinsic_dc1
        diffusion_types = pd.get_dummies(self.diffusivity_data.data.Dtype)
        diffusion_elements = pd.get_dummies(self.diffusivity_data.data.Element)

        diffusion_coefs = \
            diffusion_types.DC * inter_dc + \
            diffusion_types.DT * diffusion_elements.A * tracer_dc1
    # diffusion_types.DT * diffusion_elements.B * tracer_dc2

    # diffusion_types.DI * diffusion_elements.A * intrinsic_dc1 + \
    # diffusion_types.DI * diffusion_elements.B * intrinsic_dc2 + \
        return diffusion_coefs

    def residual_error(self, coefs):  # T in K
        """
        To calculate the residual error between log D and log D_predicted, which is weighted.
        Returns:
            None.
        """
        return \
            np.log(self.diffusion_coefs_calc(coefs) / self.diffusivity_data.data.Dexp) \
            * self.diffusivity_data.data.Weight

    def residual_error_2(self, coefs):
        return 0.5 * np.sum(np.square(self.residual_error(coefs)))

    def optimize(self, **kwargs):
        mean_square_err = None
        predicted_diffusion_coefs = None

        if not self.init_params:
            predicted_diffusion_coefs = self.diffusion_coefs_calc(self.init_params)
            mean_square_err = mean_square_error(self.diffusivity_data.data.Dexp, predicted_diffusion_coefs,
                                                self.diffusivity_data.data.Weight)
            self.optimized_results["mse"] = mean_square_err
        else:
            if self.method == "least_squares":
                if "loss" not in kwargs:
                    kwargs["loss"] = "soft_l1"
                results = least_squares(self.residual_error, self.init_params, **kwargs)
                self.optimized_results["mse"] = results.cost
            else:
                if "method" not in kwargs:
                    kwargs["method"] = "BFGS"
                results = minimize(self.residual_error_2, self.init_params, **kwargs)
                self.optimized_results["mse"] = results.fun

            self.optimized_results["OptimizedResult"] = results
            self.optimized_results["optimized_params"] = results.x

        self.diffusivity_data.data["D_" + self.model] = self.diffusion_coefs_calc(self.optimized_results["optimized_params"])
