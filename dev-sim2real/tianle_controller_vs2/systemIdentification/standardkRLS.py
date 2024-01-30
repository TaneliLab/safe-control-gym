import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class KernelRecursiveLeastSquares:
    def __init__(self, num_taps, delta, lambda_, kernel='rbf', gamma=1.0, poly_c=1, poly_d=2):
        """
        Initialize the Kernel RLS algorithm.
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
        self.X = np.zeros((num_taps, num_taps))  # Kernel matrix

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

    def update(self, input_vector, desired_output):
        """
        Update the filter coefficients.
        :param input_vector: Input vector (signal).
        :param desired_output: Desired output.
        :return: Filtered output.
        """
        # Compute kernel between input_vector and all previous inputs
        k = np.array([self.kernel_function(input_vector, self.X[i]) for i in range(self.num_taps)])

        # Compute the a priori error
        apriori_err = desired_output - np.dot(self.alpha, k)

        # Compute gain vector
        P_k = np.dot(self.P, k)
        k_P_k = np.dot(k, P_k)
        gain = P_k / (self.lambda_ + k_P_k)

        # Update the dual coefficients
        self.alpha += gain * apriori_err

        # Update the inverse correlation matrix
        self.P = (self.P - np.outer(gain, P_k)) / self.lambda_

        # Update the kernel matrix with the new input_vector
        self.X = np.roll(self.X, -1, axis=0)
        self.X[-1] = input_vector

        # Return filtered output
        return np.dot(self.alpha, k)

    
# Function to generate synthetic data
def generate_data(num_samples, noise_std):
    time = np.linspace(0, 20*np.pi, num_samples)
    desired_signal = np.sin(time)
    noise = np.random.normal(0, noise_std, num_samples)
    noisy_input = desired_signal + noise
    return time, noisy_input, desired_signal

# Example usage
num_samples = 1000
noise_std = 0.01
num_taps = 50  # You can change this to test different numbers of taps

# Generate synthetic data
time, noisy_input, desired_signal = generate_data(num_samples, noise_std)

# Example usage (you can change the kernel type to 'poly' for polynomial kernel)
# rls_kernel = KernelRecursiveLeastSquares(num_taps, delta=0.01, lambda_=0.99, kernel='rbf', gamma=0.5)
# For polynomial kernel, you might use something like:
rls_kernel = KernelRecursiveLeastSquares(num_taps, delta=0.01, lambda_=0.9, kernel='poly', poly_c=1, poly_d=3)


# Apply the Kernel RLS algorithm
estimated_signal = np.zeros(num_samples)
for i in range(num_samples):
    start_idx = max(0, i - num_taps + 1)
    input_vector = noisy_input[start_idx:i + 1]

    # Ensure input_vector is always of length num_taps
    if len(input_vector) < num_taps:
        input_vector = np.pad(input_vector, (num_taps - len(input_vector), 0), 'constant')

    # desired: obs, input
    estimated_signal[i] = rls_kernel.update(input_vector, desired_signal[i])

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(time, desired_signal, label='Desired Signal')
plt.plot(time, noisy_input, label='Noisy Input', alpha=0.5)
plt.plot(time, estimated_signal, label='Estimated Signal', color='red')
plt.legend()
plt.title('Kernel Recursive Least Squares Estimation')
plt.xlabel('Time')
plt.ylabel('Signal Amplitude')
plt.show()