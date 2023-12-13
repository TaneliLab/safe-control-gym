import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class KernelRecursiveLeastSquares:
    def __init__(self, num_taps, delta, lambda_, kernel='rbf', gamma=1.0, poly_c=1, poly_d=2):
        """
        Initialize the Parametric Kernel RLS algorithm.
        :param num_taps: The number of filter coefficients.
        :param delta: Small constant to initialize the P matrix.
        :param lambda_: Forgetting factor, typically close to 1.
        :param kernel: Type of kernel function ('rbf' for Radial Basis Function, 'poly' for Polynomial).
        :param gamma: Kernel width parameter for the RBF kernel.
        :param poly_c: Constant term in polynomial kernel.
        :param poly_d: Degree of polynomial kernel.
        """
        self.num_taps = num_taps
        self.lambda_ = lambda_
        self.gamma = gamma
        self.poly_c = poly_c
        self.poly_d = poly_d
        self.kernel_type = kernel
        self.P = (1 / delta) * np.identity(num_taps)
        self.alpha = np.zeros(num_taps)  # Dual coefficients
        self.X = np.zeros((num_taps, 1))  # Kernel matrix for acc_command

    def rbf_kernel(self, x, y):
        """ Radial Basis Function kernel. """
        return np.exp(-self.gamma * np.linalg.norm(x - y)**2)

    def polynomial_kernel(self, x, y):
        """ Polynomial kernel. """
        return (np.dot(x, y) + self.poly_c)**self.poly_d

    def kernel_function(self, x, y):
        """ Choose the appropriate kernel function. """
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(x, y)
        elif self.kernel_type == 'poly':
            return self.polynomial_kernel(x, y)
        else:
            raise ValueError("Invalid kernel type. Choose 'rbf' or 'poly'.")

    # def update(self, input_vector, desired_output):
    def update(self, acc_command, observation, desired_output):
        """
        Update the filter coefficients and estimate a new acceleration command.
        :param acc_command: Current acceleration command.
        :param observation: Observed signal.
        :param desired_output: Desired output (reference signal).
        :return: New estimated acceleration command.
        """
        # Compute the error between observation and desired output
        error = observation - desired_output

        # Use acc_command as the sole input to the kernel
        k = np.array([self.kernel_function(acc_command, self.X[i]) for i in range(self.num_taps)])

        # Reshape k to be a column vector if it's not already
        k = k.reshape(-1, 1)

        # Compute gain vector
        P_k = np.dot(self.P, k)
        k_P_k = np.dot(k.T, P_k)  # Transpose k to get a dot product
        gain = P_k / (self.lambda_ + k_P_k)

        # Update the dual coefficients
        self.alpha += gain.flatten() * error  # Flatten gain to match the shape of alpha

        # Update the inverse correlation matrix
        self.P = (self.P - np.outer(gain, P_k.T)) / self.lambda_

        # Update the kernel matrix with the new acc_command
        self.X = np.roll(self.X, -1, axis=0)
        self.X[-1] = acc_command

        # Estimate new acceleration command to minimize error in the next iteration
        new_acc_command = acc_command - np.dot(self.alpha, k)
        return new_acc_command[0]

    
# Example usage
# rls_kernel = KernelRecursiveLeastSquares(num_taps=10, delta=0.01, lambda_=0.99, kernel='rbf', gamma=0.5)
# acc_command = [your_acceleration_command]
# observation = [your_observed_signal]
# desired_output = [your_reference_signal]
# estimation = rls_kernel.update(acc_command, observation, desired_output)