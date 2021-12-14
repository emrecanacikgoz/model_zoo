"""
Configuration parameters that need to be used in model.
"""

# Model parameters in a structured order
model_configs = [
    # expand_ratio, channels, kernel_size, stride, repeats
    [1, 16, 3, 1, 1],
    [6, 24, 3, 2, 2],
    [6, 40, 5, 2, 2],
    [6, 80, 3, 2, 3],
    [6, 112, 5, 1, 3],
    [6, 192, 5, 2, 4],
    [6, 320, 3, 1, 1],
]

# Phi values for each version of EfficientNet[b0-b7]
phi_values = {
    # tuple of: (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}