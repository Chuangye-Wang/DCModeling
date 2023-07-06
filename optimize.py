import pandas as pd

from help_functions import *
from scipy.optimize import least_squares, minimize


class Optimizer:
    """
    An optimizer to optimize interaction parameters in the diffusion model. Two methods are employed, and they are
    scipy.optimize.least_squares and scipy.optimize.minimize functions.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Attributes:
        diffusivity_data: A DiffusionData object that has all experimental info.
        model: A string for which diffusion model to use to describe the diffusion behavior in the system.
        method: A string for which optimization or minimization method to use.
        init_params: An array denoting the initialized values for parameters in the diffusion model.
        optimized_results: A dict to store the optimized results.
    """

    def __init__(self, diffusivity_data, model='1-para', method="least_squares"):
        """ Initialize.
        Args:
            diffusivity_data: A DiffusionData object that has all experimental info.
            model: A string for which diffusion model to use to describe the diffusion behavior in the system.
            method: A string for which optimization or minimization method to use.
        """
        self.diffusivity_data = diffusivity_data
        self.model = model
        self.method = method
        # initialize model parameters
        self.init_params = np.random.random(int(self.model.split("-")[0]))

        # initialize which method to use for optimization.
        if method not in ("least_squares", "minimize"):
            raise ValueError("The method should be either least_squares or minimize.")

        self.optimized_results = {"OptimizedResult": None,
                                  "mse": None,
                                  "optimized_params": []}

    def residual_error(self, coefs):  # T in K
        """
        To calculate the residual error between log D and log D_predicted, which is weighted.
        This is used for least_square optimization.
        Returns:
            None.
        """
        return \
            np.log(self.diffusivity_data.diffusion_coefs_calc(coefs) / self.diffusivity_data.data.Dexp) \
            * self.diffusivity_data.data.Weight

    def residual_error_for_minimize(self, coefs):
        """
        To calculate the residual error between log D and log D_predicted, which is already summed.
        This is specifically used for minimize optimization.
        Returns:
            None.
        """
        return 0.5 * np.sum(np.square(self.residual_error(coefs)))

    def optimize(self, **kwargs):
        """
        To optimize the object function.
        Args:
            **kwargs: Arbitrary keyword arguments for optimize functions. Optimization functions include
            least_squares and minimize methods.
            some keys for least_squares:
                {
                    method: A string for the algorithm used to perform minimization.
                    loss: A string for loss function.
                    f_scale: A float for value of soft margin between inlier and outlier residuals.
                }
        Returns:
            None
        """
        if not self.init_params:
            predicted_diffusion_coefs = self.diffusivity_data.diffusion_coefs_calc(self.init_params)
            total_square_err = total_square_error(self.diffusivity_data.data.Dexp, predicted_diffusion_coefs,
                                                  self.diffusivity_data.data.Weight)
            self.optimized_results["mse"] = total_square_err
        else:
            if self.method == "least_squares":
                if "loss" not in kwargs:
                    kwargs["loss"] = "soft_l1"
                results = least_squares(self.residual_error, self.init_params, **kwargs)
                self.optimized_results["mse"] = results.cost
            else:
                if "method" not in kwargs:
                    kwargs["method"] = "BFGS"
                results = minimize(self.residual_error_for_minimize, self.init_params, **kwargs)
                self.optimized_results["mse"] = results.fun

            self.optimized_results["OptimizedResult"] = results
            self.optimized_results["optimized_params"] = results.x

        self.diffusivity_data.data["D_" + self.model] = self.diffusivity_data.diffusion_coefs_calc(self.optimized_results["optimized_params"])
