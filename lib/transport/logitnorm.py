import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal


# Define the LogitNormal class with the pdf function
class LogitNormal:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normal = Normal(mean, std)

    def sample(self, sample_shape=torch.Size()):
        # Sample from the normal distribution
        samples = self.normal.sample(sample_shape)
        # Apply the sigmoid function to transform to the logit-normal distribution
        return torch.sigmoid(samples)

    def log_prob(self, value):
        # Convert the value to logit space
        logit_value = torch.log(value) - torch.log1p(-value)
        # Compute the log probability in the normal distribution
        normal_log_prob = self.normal.log_prob(logit_value)
        # Adjust the log probability to account for the sigmoid transformation
        return normal_log_prob - torch.log(value) - torch.log1p(-value)

    def pdf(self, value):
        # Ensure the value is in the interval (0, 1)
        if torch.any((value <= 0) | (value >= 1)):
            raise ValueError("Value must be in the interval (0, 1)")

        # Convert the value to logit space
        logit_value = torch.log(value) - torch.log1p(-value)
        # Compute the PDF in the normal distribution
        normal_pdf = torch.exp(self.normal.log_prob(logit_value))
        # Adjust the PDF to account for the sigmoid transformation
        sigmoid_derivative = value * (1 - value)
        return normal_pdf / sigmoid_derivative


if __name__ == "__main__":
    # Parameters for the LogitNormal distribution
    mean = torch.tensor(0.0)
    std = torch.tensor(1.0)

    # Instantiate the LogitNormal distribution
    logit_normal = LogitNormal(mean, std)

    # Values to evaluate the PDF at
    x = np.linspace(0.001, 0.999, 1000)
    x_tensor = torch.tensor(x, dtype=torch.float32)

    # Compute the PDF values
    pdf_values = logit_normal.pdf(x_tensor).detach().numpy()
    pdf_values *= x / (1 - x)

    # Plot the PDF
    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf_values, label="Logit-Normal PDF")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("PDF of Logit-Normal Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()
