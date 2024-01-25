#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def generate_stock_path(N, p, initial_price=100):
    """
    Generates a synthetic stock price path.

    Parameters:
    N (int): Number of steps in the path.
    p (float): Probability of the stock price going up at each step.
    initial_price (float): Starting price of the stock.

    Returns:
    np.array: Array representing the stock price path.
    """
    # Input validation
    if not 0 <= p <= 1:
        raise ValueError("Probability p must be between 0 and 1.")
    if N <= 0:
        raise ValueError("Number of steps N must be a positive integer.")
    if initial_price <= 0:
        raise ValueError("Initial price must be a positive number.")

    # Generate random steps: 1 for up, -1 for down, starting from the second step
    steps = np.random.choice([1, -1], size=N-1, p=[p, 1-p])
    
    # Starting with the initial price and then applying the random steps
    path = np.concatenate([[initial_price], initial_price + np.cumsum(steps)])
    
    return path

# Example usage
path = generate_stock_path(1000, 0.5)
print(path[:10])  # Displaying the first 10 values of the path to check the function


# In[2]:


def generate_multiple_stock_paths(k, N, p, initial_price=100):
    """
    Generates multiple synthetic stock price paths.

    Parameters:
    k (int): Number of paths to generate.
    N (int): Number of steps in each path.
    p (float): Probability of the stock price going up at each step.
    initial_price (float): Starting price of the stock.

    Returns:
    np.ndarray: A 2D array where each row represents a stock price path.
    """
    # Input validation for the number of paths
    if k <= 0:
        raise ValueError("Number of paths k must be a positive integer.")

    # Generate k stock price paths
    paths = np.array([generate_stock_path(N, p, initial_price) for _ in range(k)])

    return paths

# Example usage
k = 50000  # Number of paths
N = 1000   # Number of steps in each path
p = 0.5    # Probability of price increase

paths = generate_multiple_stock_paths(k, N, p)
paths.shape  # Displaying the shape of the generated paths array to verify


# In[3]:


def calculate_max_drawdown_custom(paths):
    """
    Calculates the maximum drawdown for each synthetic stock price path using a custom method.

    Parameters:
    paths (np.ndarray): A 2D array where each row is a stock price path.

    Returns:
    np.array: Array containing the maximum drawdown for each path.
    """
    max_drawdowns = np.zeros(paths.shape[0])

    for i, path in enumerate(paths):
        peak = path[0]
        max_dd = 0

        for price in path:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            max_dd = max(max_dd, dd)

        max_drawdowns[i] = max_dd

    return max_drawdowns

# Calculating max drawdown for each path using the custom function
max_drawdowns_custom = calculate_max_drawdown_custom(paths)
max_drawdowns_custom[:10]  # Displaying the max drawdown for the first 10 paths for verification


# In[4]:


def calculate_corrected_max_drawdown_matrix_with_peaks_troughs(paths):
    num_paths, num_steps = paths.shape
    corrected_max_dd_matrix = np.zeros((num_paths, num_steps))
    peaks_matrix = np.zeros((num_paths, num_steps))
    troughs_matrix = np.zeros((num_paths, num_steps))

    for i, path in enumerate(paths):
        peak = path[0]
        max_dd_so_far = 0
        trough = path[0]

        for t in range(num_steps):
            peak = max(peak, path[t])
            current_dd = (peak - path[t]) / peak if peak != 0 else 0
            if current_dd > max_dd_so_far:
                max_dd_so_far = current_dd
                trough = path[t]

            corrected_max_dd_matrix[i, t] = max_dd_so_far
            peaks_matrix[i, t] = peak
            troughs_matrix[i, t] = trough

    return corrected_max_dd_matrix, peaks_matrix, troughs_matrix


# In[5]:


# def calculate_max_drawdown_matrix(paths):
#     """
#     Calculates the running maximum drawdown at each time step for each path.

#     Parameters:
#     paths (np.ndarray): A 2D array where each row is a stock price path.

#     Returns:
#     np.ndarray: A 2D array where each cell contains the max drawdown at that time step for each path.
#     """
#     num_paths, num_steps = paths.shape
#     max_drawdown_matrix = np.zeros((num_paths, num_steps))

#     for i, path in enumerate(paths):
#         peak = path[0]
#         for t in range(num_steps):
#             peak = max(peak, path[t])
#             max_drawdown_matrix[i, t] = (peak - path[t]) / peak if peak != 0 else 0

#     return max_drawdown_matrix

# Calculating the running max drawdown matrix along with peaks and troughs
max_dd_matrix, peaks_matrix, troughs_matrix = calculate_corrected_max_drawdown_matrix_with_peaks_troughs(paths)

# Displaying a portion of the matrix for verification
max_dd_matrix[:5, :10]  # First 5 paths, first 10 steps



# In[6]:


def calculate_average_max_drawdown_per_time_step(max_dd_matrix):
    """
    Calculates the average maximum drawdown for each time step across all paths.

    Parameters:
    max_dd_matrix (np.ndarray): A 2D array where each cell contains the max drawdown at that time step for each path.

    Returns:
    np.array: Array containing the average maximum drawdown for each time step.
    """
    # Calculating the average of max drawdowns for each time step
    average_max_dd_per_step = np.mean(max_dd_matrix, axis=0)

    return average_max_dd_per_step

# Calculating the average max drawdown for each time step
average_max_dd_per_step = calculate_average_max_drawdown_per_time_step(max_dd_matrix)

# Displaying the first 10 time steps for verification
average_max_dd_per_step[:10]


# In[7]:


import matplotlib.pyplot as plt

def plot_average_max_drawdown_per_time_step(average_max_dd_per_step):
    """
    Plots the distribution of average maximum drawdowns over time.

    Parameters:
    average_max_dd_per_step (np.array): Array containing the average maximum drawdown for each time step.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(average_max_dd_per_step, label='Average Max Drawdown')
    plt.xlabel('Time Step')
    plt.ylabel('Average Max Drawdown')
    plt.title('Average Maximum Drawdown Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting the distribution of average max drawdowns over time
plot_average_max_drawdown_per_time_step(average_max_dd_per_step)


# In[8]:


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assuming 'max_dd_matrix' is your original data matrix
average_max_dd_squared = np.mean(np.square(max_dd_matrix), axis=0)
time_steps = np.arange(max_dd_matrix.shape[1]).reshape(-1, 1)

ols_model = LinearRegression().fit(time_steps, average_max_dd_squared)
ols_predictions = ols_model.predict(time_steps)

r2_ols = r2_score(average_max_dd_squared, ols_predictions)
print("R-squared for OLS Model:", r2_ols)

# Number of parameters (1 for slope + 1 for intercept)
n_params = 2

# Calculating residuals
residuals = average_max_dd_squared - ols_predictions

# AIC calculation
n = len(average_max_dd_squared)
residual_sum_of_squares = np.sum(np.square(residuals))
aic_ols = n * np.log(residual_sum_of_squares / n) + 2 * n_params
print("AIC for OLS Model:", aic_ols)


# In[9]:


from pygam import LinearGAM, s
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

# Preparing data
average_max_dd_squared = np.mean(np.square(max_dd_matrix), axis=0)  # Replace max_dd_matrix with your data
time_steps = np.arange(max_dd_matrix.shape[1])

# Fitting GAM
gam_model = LinearGAM(s(0, spline_order=3, n_splines=6)).gridsearch(time_steps.reshape(-1, 1), average_max_dd_squared)

# Partial Residual Plot for GAM
for i, term in enumerate(gam_model.terms):
    if term.isintercept:
        continue

    XX = gam_model.generate_X_grid(term=i)
    pdep, confi = gam_model.partial_dependence(term=i, X=XX, width=0.95)
    
    plt.figure()
    plt.plot(XX[:, i], pdep)
    plt.plot(XX[:, i], confi, c='r', ls='--')
    plt.title(f'Partial dependence for feature {i}')
    plt.show()

# Calculate R-squared for GAM
gam_predictions = gam_model.predict(time_steps.reshape(-1, 1))
r2_gam = r2_score(average_max_dd_squared, gam_predictions)

# Calculate AIC for GAM
aic_gam = gam_model.statistics_['AIC']

# Fit OLS Model
ols_model = LinearRegression().fit(time_steps.reshape(-1, 1), average_max_dd_squared)
ols_predictions = ols_model.predict(time_steps.reshape(-1, 1))

# Calculate R-squared for OLS
r2_ols = r2_score(average_max_dd_squared, ols_predictions)

# Calculate AIC for OLS
n_params = 2  # Number of parameters in OLS (intercept and slope)
residuals = average_max_dd_squared - ols_predictions
n = len(average_max_dd_squared)
residual_sum_of_squares = np.sum(np.square(residuals))
aic_ols = n * np.log(residual_sum_of_squares / n) + 2 * n_params

print("GAM R-squared:", r2_gam, "GAM AIC:", aic_gam)
print("OLS R-squared:", r2_ols, "OLS AIC:", aic_ols)


# In[10]:


def predict_max_dd(N, ols_model, gam_model):
    """
    Predicts the max drawdown using both OLS and GAM models.

    Parameters:
    N (int): Time step for which the prediction is to be made.
    ols_model (LinearRegression): Trained OLS model.
    gam_model (LinearGAM): Trained GAM model.

    Returns:
    tuple: Predictions from OLS model and GAM model.
    """
    # Reshape N for model input
    N_reshaped = np.array([N]).reshape(-1, 1)

    # Predict using OLS model
    ols_prediction = ols_model.predict(N_reshaped)[0]

    # Predict using GAM model
    gam_prediction = gam_model.predict(N_reshaped)[0]

    return ols_prediction, gam_prediction

# Example usage (assuming ols_model and gam_model are already trained)
N = 3  # Replace with your desired time step
ols_pred, gam_pred = predict_max_dd(N, ols_model, gam_model)
print("OLS Prediction:", ols_pred, "GAM Prediction:", gam_pred)


# In[11]:


# For OLS model
ols_predictions = ols_model.predict(time_steps.reshape(-1, 1))
ols_residuals = average_max_dd_squared - ols_predictions

# For GAM model
gam_predictions = gam_model.predict(time_steps.reshape(-1, 1))
gam_residuals = average_max_dd_squared - gam_predictions

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(time_steps, ols_residuals, alpha=0.5, label='OLS Residuals')
plt.scatter(time_steps, gam_residuals, alpha=0.5, label='GAM Residuals', color='red')
plt.axhline(y=0, color='black', linestyle='--')
plt.xlabel('Time Step (N)')
plt.ylabel('Residuals')
plt.title('Residuals of OLS and GAM Models')
plt.legend()
plt.grid(True)
plt.show()


# In[12]:


import joblib
# Save the models
joblib.dump(gam_model, 'gam_model.pkl')


# In[13]:


joblib.dump(ols_model, 'ols_model.pkl')


# In[14]:


predict_max_dd(1, ols_model, gam_model)


# In[15]:


N_reshaped = np.array([3]).reshape(-1, 1)
gam_model.predict(N_reshaped)[0]


# In[16]:


N_reshaped = np.array([3]).reshape(-1, 1)
ols_model.predict(N_reshaped)[0]


# In[17]:


def get_empirical_average_max_dd(N, max_dd_matrix):
    """
    Gets the empirical average maximum drawdown for a given number of steps N.

    Parameters:
    N (int): The number of steps from the origin.
    max_dd_matrix (np.ndarray): A 2D array with max drawdowns for each path at each time step.

    Returns:
    float: The empirical average maximum drawdown for the given number of steps.
    """
    if N >= max_dd_matrix.shape[1]:
        raise ValueError("N exceeds the number of steps available in the data.")

    # Calculate the average max drawdown for each time step
    average_max_dd_per_step = calculate_average_max_drawdown_per_time_step(max_dd_matrix)

    # Return the average max drawdown for the specified number of steps
    return average_max_dd_per_step[N]

# Example usage
N = 3  # Replace with your desired number of steps
empirical_avg_max_dd = get_empirical_average_max_dd(N, max_dd_matrix)
print("Empirical Average Max Drawdown for N =", N, "is", empirical_avg_max_dd)


# In[18]:


# Import necessary libraries
import numpy as np
import pickle

# Your existing functions (like calculate_average_max_drawdown_per_time_step) and the max_dd_matrix should be defined above this

# Calculate the average maximum drawdown per time step
average_max_dd_per_step = calculate_average_max_drawdown_per_time_step(max_dd_matrix)

# Save the array to a .pkl file
with open('average_max_dd_per_step.pkl', 'wb') as file:
    pickle.dump(average_max_dd_per_step, file)


# In[19]:


from scipy import stats

def confidence_interval_max_dd(N, max_dd_matrix, confidence=0.99):
    """
    Calculates the confidence interval for the expected maximum drawdown at a given number of steps N.

    Parameters:
    N (int): The number of steps from the origin.
    max_dd_matrix (np.ndarray): A 2D array with max drawdowns for each path at each time step.
    confidence (float): The confidence level for the interval.

    Returns:
    tuple: The lower and upper bounds of the confidence interval.
    """
    if N >= max_dd_matrix.shape[1]:
        raise ValueError("N exceeds the number of steps available in the data.")

    # Extract max drawdown values at time step N
    dd_values_at_N = max_dd_matrix[:, N]

    # Calculate mean and standard error of the mean
    mean_dd = np.mean(dd_values_at_N)
    sem_dd = stats.sem(dd_values_at_N)  # Standard Error of the Mean

    # Determine the margin of error
    margin_of_error = sem_dd * stats.t.ppf((1 + confidence) / 2, len(dd_values_at_N) - 1)

    return mean_dd - margin_of_error, mean_dd + margin_of_error

# Example usage
N = 3  # Replace with your desired number of steps
lower_bound, upper_bound = confidence_interval_max_dd(N, max_dd_matrix, confidence=0.99)
print(f"99% Confidence Interval for Max Drawdown at N={N} is [{lower_bound:.2%}, {upper_bound:.2%}]")


# In[20]:


def test_max_drawdown_calculation(N_max, initial_price=100, step_size=1):
    results = {}
    for N in range(1, N_max + 1):
        all_combinations = np.array(np.meshgrid(*[[-1, 1]] * N)).T.reshape(-1, N)
        paths_max_dd = []
        for comb in all_combinations:
            path = np.concatenate(([initial_price], initial_price + np.cumsum(comb * step_size)))
            path_array = np.array([path])
            
            # Original calculation
            max_dd_orig = calculate_max_drawdown_custom(path_array)[0]
            
            # Experimental calculation
            max_dd_exp, _, _ = calculate_max_drawdown_experimental(path_array)

            # Calculate absolute difference (ensuring it's a scalar)
            abs_diff = np.abs(max_dd_orig - max_dd_exp).item()  # Using .item() to convert to scalar

            paths_max_dd.append((path, max_dd_orig, max_dd_exp, f"{abs_diff * 100:.3f}%"))

        # Sort by largest absolute difference
        results[N] = sorted(paths_max_dd, key=lambda x: float(x[3].rstrip('%')), reverse=True)

    return results

# Testing max drawdown calculation
N_max = 4
test_results = test_max_drawdown_calculation(N_max)

# Printing the test results in a human-readable format
for N in range(1, N_max + 1):
    print(f"\nTest Results for N = {N}:")
    print("-" * 60)
    print(f"{'Path':<30} {'Original DD':<15} {'Experimental DD':<20} {'Abs. Difference':<15}")
    for path, orig_dd, exp_dd, diff in test_results[N]:
        path_str = ', '.join(map(str, path))  # Convert the path array to a comma-separated string

        # Ensure orig_dd and exp_dd are scalars before formatting
        orig_dd_scalar = orig_dd.item() if isinstance(orig_dd, np.ndarray) else orig_dd
        exp_dd_scalar = exp_dd.item() if isinstance(exp_dd, np.ndarray) else exp_dd

        print(f"{path_str:<30} {orig_dd_scalar:<15.3f} {exp_dd_scalar:<20.3f} {diff:<15}")


# In[ ]:


def calculate_max_drawdown_experimental(paths):
    max_drawdowns = np.zeros(paths.shape[0])
    peak_values = np.zeros(paths.shape[0])
    trough_values = np.zeros(paths.shape[0])

    for i, path in enumerate(paths):
        peak = path[0]
        trough = path[0]
        max_dd = 0

        for price in path:
            if price > peak:
                peak = price
            dd = (peak - price) / peak
            if dd > max_dd:
                max_dd = dd
                trough = price

        max_drawdowns[i] = max_dd
        peak_values[i] = peak
        trough_values[i] = trough

    return max_drawdowns, peak_values, trough_values


# In[ ]:


# Extracting max drawdowns for N=1, N=2, and N=3
max_dd_at_N1 = max_dd_matrix[:, 1]  # All rows, column corresponding to N=1
max_dd_at_N2 = max_dd_matrix[:, 2]  # All rows, column corresponding to N=2
max_dd_at_N3 = max_dd_matrix[:, 3]  # All rows, column corresponding to N=3

# Calculating average max drawdown manually
avg_max_dd_manual_N1 = np.mean(max_dd_at_N1)
avg_max_dd_manual_N2 = np.mean(max_dd_at_N2)
avg_max_dd_manual_N3 = np.mean(max_dd_at_N3)

# Comparing with the function output
print("Manually Calculated Average Max Drawdown:")
print("N=1:", avg_max_dd_manual_N1)
print("N=2:", avg_max_dd_manual_N2)
print("N=3:", avg_max_dd_manual_N3)
print("\nFunction Output for Average Max Drawdown:")
print("N=1:", average_max_dd_per_step[1])
print("N=2:", average_max_dd_per_step[2])
print("N=3:", average_max_dd_per_step[3])


# In[ ]:


def calculate_corrected_max_drawdown_matrix(paths):
    num_paths, num_steps = paths.shape
    corrected_max_dd_matrix = np.zeros((num_paths, num_steps))

    for i, path in enumerate(paths):
        peak = path[0]
        max_dd_so_far = 0  # Initialize the maximum drawdown observed so far

        for t in range(num_steps):
            peak = max(peak, path[t])
            current_dd = (peak - path[t]) / peak if peak != 0 else 0
            max_dd_so_far = max(max_dd_so_far, current_dd)
            corrected_max_dd_matrix[i, t] = max_dd_so_far

    return corrected_max_dd_matrix


# In[ ]:


def compare_max_dd_matrices_sample(paths, sample_size=50, max_output_per_path=5):
    sampled_indices = np.random.choice(paths.shape[0], size=sample_size, replace=False)
    sampled_paths = paths[sampled_indices, :]

    original_max_dd_matrix = calculate_max_drawdown_matrix(sampled_paths)
    corrected_max_dd_matrix = calculate_corrected_max_drawdown_matrix(sampled_paths)

    for i, (original_row, corrected_row) in enumerate(zip(original_max_dd_matrix, corrected_max_dd_matrix)):
        differences = [(t, orig_dd, corr_dd) for t, (orig_dd, corr_dd) in enumerate(zip(original_row, corrected_row)) if orig_dd != corr_dd]
        if differences:
            print(f"Differences found in path {sampled_indices[i]} (showing up to {max_output_per_path} differences):")
            for diff in differences[:max_output_per_path]:
                t, orig_dd, corr_dd = diff
                print(f"  Time step {t}: Original DD = {orig_dd:.3f}, Corrected DD = {corr_dd:.3f}")
            if len(differences) > max_output_per_path:
                print(f"  ... and {len(differences) - max_output_per_path} more differences.")
            print()

# Example usage on a sample of the simulated dataset
compare_max_dd_matrices_sample(paths, sample_size=50, max_output_per_path=5)


# In[ ]:


def display_significant_differences(paths, threshold=0.01, num_paths_to_show=20, steps_to_show=10):
    # Calculating max drawdown matrices
    original_max_dd_matrix = calculate_max_drawdown_matrix(paths)
    corrected_max_dd_matrix = calculate_corrected_max_drawdown_matrix(paths)

    # Identifying paths with significant differences
    significant_diff_indices = []
    for i in range(paths.shape[0]):
        if np.any(np.abs(original_max_dd_matrix[i] - corrected_max_dd_matrix[i]) > threshold):
            significant_diff_indices.append(i)
            if len(significant_diff_indices) == num_paths_to_show:
                break

    # Displaying the first few steps for the chosen paths
    for index in significant_diff_indices:
        print(f"Path {index} Price Series: {paths[index][:steps_to_show]}")
        print("Original Max DD:  ", ["{:.2%}".format(val) for val in original_max_dd_matrix[index][:steps_to_show]])
        print("Corrected Max DD: ", ["{:.2%}".format(val) for val in corrected_max_dd_matrix[index][:steps_to_show]])
        print()

# Example usage on the simulated dataset
display_significant_differences(paths, threshold=0.01, num_paths_to_show=20, steps_to_show=10)



# In[ ]:


def calculate_corrected_max_drawdown_matrix_with_peaks_troughs(paths):
    num_paths, num_steps = paths.shape
    corrected_max_dd_matrix = np.zeros((num_paths, num_steps))
    peaks_matrix = np.zeros((num_paths, num_steps))
    troughs_matrix = np.zeros((num_paths, num_steps))

    for i, path in enumerate(paths):
        peak = path[0]
        max_dd_so_far = 0  # Initialize the maximum drawdown observed so far
        trough = path[0]

        for t in range(num_steps):
            peak = max(peak, path[t])
            current_dd = (peak - path[t]) / peak if peak != 0 else 0
            if current_dd > max_dd_so_far:
                max_dd_so_far = current_dd
                trough = path[t]

            corrected_max_dd_matrix[i, t] = max_dd_so_far
            peaks_matrix[i, t] = peak
            troughs_matrix[i, t] = trough

    return corrected_max_dd_matrix, peaks_matrix, troughs_matrix

# Example usage
corrected_max_dd_matrix, peaks_matrix, troughs_matrix = calculate_corrected_max_drawdown_matrix_with_peaks_troughs(paths)


# In[ ]:


# Checking the length of the price series vs the drawdown series for a sample path
sample_path_index = 0  # Example index
price_series_length = len(paths[sample_path_index])
drawdown_series_length = len(max_dd_matrix[sample_path_index])

print("Length of Price Series:", price_series_length)
print("Length of Drawdown Series:", drawdown_series_length)


# In[ ]:


def display_significant_differences_with_peaks_troughs(paths, threshold=0.01, num_paths_to_show=5, steps_to_show=10):
    # Calculating the corrected max drawdown matrix along with peaks and troughs
    corrected_max_dd_matrix, peaks_matrix, troughs_matrix = calculate_corrected_max_drawdown_matrix_with_peaks_troughs(paths)

    # Identifying paths with significant differences
    for i in range(paths.shape[0]):
        if np.any(corrected_max_dd_matrix[i] > threshold):
            print(f"Path {i} Price Series: {paths[i][:steps_to_show]}")
            print("Max DD:            ", ["{:.2%}".format(dd) for dd in corrected_max_dd_matrix[i][:steps_to_show]])
            print("Peaks:             ", peaks_matrix[i][:steps_to_show])
            print("Troughs:           ", troughs_matrix[i][:steps_to_show])
            print("Peak-Trough Diff:  ", peaks_matrix[i][:steps_to_show] - troughs_matrix[i][:steps_to_show])
            print()
            num_paths_to_show -= 1
            if num_paths_to_show == 0:
                break

# Example usage on the simulated dataset
display_significant_differences_with_peaks_troughs(paths, threshold=0.01, num_paths_to_show=5, steps_to_show=10)


# In[ ]:


get_ipython().system('pip install nbimporter')


# In[ ]:




