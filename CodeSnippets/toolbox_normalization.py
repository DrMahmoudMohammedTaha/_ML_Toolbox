import numpy as np

# Sample data (numpy array)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Min-Max Scaling
def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data

# Standardization (Z-score)
def standardization(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    scaled_data = (data - mean) / std_dev
    return scaled_data

# Robust Scaling
def robust_scaling(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    scaled_data = (data - q1) / (q3 - q1)
    return scaled_data

# Log Transformation
def log_transformation(data):
    scaled_data = np.log(data)
    return scaled_data

# Applying normalization methods
scaled_min_max = min_max_scaling(data)
scaled_standardization = standardization(data)
scaled_robust = robust_scaling(data)
scaled_log = log_transformation(data)

# Displaying scaled data
print("Min-Max Scaling:")
print(scaled_min_max)
print("\nStandardization (Z-score):")
print(scaled_standardization)
print("\nRobust Scaling:")
print(scaled_robust)
print("\nLog Transformation:")
print(scaled_log)
