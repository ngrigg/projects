#!/usr/bin/env python
# coding: utf-8

# In[79]:


def calculate_total_paths(t, T):
    """
    Calculate the total number of paths the stock can take between time t and T.

    Parameters:
    t (int): The starting time.
    T (int): The ending time.

    Returns:
    int: Total number of paths between t and T.
    """

    # Error handling for invalid t and T values
    if not isinstance(t, int) or not isinstance(T, int) or t < 0 or T < t:
        print("Error: t and T must be non-negative integers with t <= T.")
        return None

    # Number of steps from t to T
    steps = T - t

    # Calculate the total number of paths using the binomial coefficient (2^steps)
    total_paths = 2 ** steps

    # Print statement for debugging
    print(f"For time interval t = {t} to T = {T}, total number of paths: {total_paths}")

    return total_paths

# Example use of the function
calculate_total_paths(0, 5)  # Test with t = 0 and T = 5


# In[80]:


def check_path(price_path, T):
    """
    Check the validity of a given price path.

    Parameters:
    price_path (list): The list representing a price path.
    T (int): The time period for which the path is being checked.

    Returns:
    bool: True if the path is valid, False otherwise.
    """
    
    if price_path is None:
        return True

    # Check if the path starts at 0 and has the correct length
    if price_path[0] != 0 or len(price_path) != T + 1:
        return False

    # Check if each element in the path is either +1 or -1 away from its predecessor
    for i in range(1, len(price_path)):
        if abs(price_path[i] - price_path[i - 1]) != 1:
            return False

    return True

# Example use of the function
example_path = [0, 1, 2, 1, 2, 3]
check_path(example_path, 5)  # Example input path and T = 5



# In[92]:


def max_dd_smart(cumulative_values):
    peak = cumulative_values[0]
    max_drawdown = 0
    for value in cumulative_values:
        if value > peak:
            peak = value
        else:
            drawdown = peak - value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown

# Example use of the functions
example_values = [0, 1, 2, 1, 0, -1, -2]
print("Smart Maximum Drawdown:", max_dd_smart(example_values))


# In[93]:


def check_path_validity(current_value, time):
    """
    Validates the legality of a path during its construction.
    """
    return -time <= current_value <= time


# In[94]:


def get_empirical_average_max_dd(N, average_max_dd_per_step):
    """
    Gets the empirical average maximum drawdown for a given number of steps N.

    Parameters:
    N (int): The number of steps from the origin.
    average_max_dd_per_step (np.ndarray): A 1D array with average max drawdowns for each time step.

    Returns:
    float: The empirical average maximum drawdown for the given number of steps.
    """

    # Check if N is within the range of the array
    if N >= len(average_max_dd_per_step):
        raise ValueError("N exceeds the number of steps available in the data.")

    # Return the average max drawdown for the specified number of steps
    return average_max_dd_per_step[N]


# In[105]:


from functools import lru_cache

@lru_cache(maxsize=1000)
def calculate_max_dd_paths_memoized(T, current_path_tuple):
    current_path = list(current_path_tuple)  # Convert tuple back to list for processing

    if len(current_path) == T + 1:
        actual_max_dd = max_dd_smart(current_path)
        return {actual_max_dd: 1}

    path_counts = {}
    for next_value in [current_path[-1] - 1, current_path[-1] + 1]:
        if -T <= next_value <= T:
            next_path = current_path + [next_value]
            # Convert next_path to a tuple for the recursive call
            sub_path_counts = calculate_max_dd_paths_memoized(T, tuple(next_path))
            
            for max_dd, count in sub_path_counts.items():
                path_counts[max_dd] = path_counts.get(max_dd, 0) + count

    return path_counts


# In[106]:


import pickle
import numpy as np

def basic_checks_for_memoized(T, existing_paths, average_max_dd_per_step):
    # Validate existing_paths structure
    for key in existing_paths.keys():
        if not isinstance(key, int):
            print(f"Invalid key detected: {key}")
            raise ValueError("Invalid key format in existing_paths")
    
    total_paths = calculate_total_paths(0, T)

    # Adjusted expected max_dd calculation
    expected_max_dd = sum(max_dd * count for max_dd, count in existing_paths.items()) / total_paths
    expected_percent_max_dd = (expected_max_dd / 100) * 100

    # Empirical average max drawdown for T (as a percentage)
    empirical_avg_max_dd = get_empirical_average_max_dd(T, average_max_dd_per_step) * 100

    print(f"For T={T}: Expected % Max Drawdown: {expected_percent_max_dd:.2f}%, Empirical Average Max DD: {empirical_avg_max_dd:.2f}%")

    # Initialize check counters
    total_checks = 4
    passed_checks = 0
    failed_check_details = []

    # Check 1: Total number of paths matches the result of calculate_total_paths
    total_paths_calculated = sum(existing_paths.values())
    if total_paths_calculated == total_paths:
        passed_checks += 1
    else:
        failed_check_details.append(f"Total paths mismatch for T={T}")
    
    # Check 2: There is only 1 path where max_dd = T
    max_dd_equals_T_paths = existing_paths.get(T, 0)
    if max_dd_equals_T_paths == 1:
        passed_checks += 1
    else:
        failed_check_details.append(f"More than one path where max_dd equals T for T={T}")

    # Check 3: The number of cases where max_dd = 1 is greater than or equal to T
    max_dd_equals_1_paths = existing_paths.get(1, 0)
    if max_dd_equals_1_paths >= T:
        passed_checks += 1
    else:
        failed_check_details.append(f"Number of paths with max_dd = 1 less than T for T={T}")

    # Check 4: The number of cases where max_dd = 0 is always 1
    max_dd_equals_0_paths = existing_paths.get(0, 0)
    if max_dd_equals_0_paths == 1:
        passed_checks += 1
    else:
        failed_check_details.append(f"Incorrect number of paths with max_dd = 0 for T={T}")

    # Summary of checks
    if passed_checks == total_checks:
        print(f"All {total_checks} checks passed for T={T}.")
    else:
        print(f"{passed_checks} of {total_checks} checks passed for T={T}.")
        if failed_check_details:
            print("Failed checks details:")
            for detail in failed_check_details:
                print(detail)
        else:
            print("No details available for failed checks.")

for T in range(1, 21):
    max_dd_paths = calculate_max_dd_paths_memoized(T, (0,))
    basic_checks_for_memoized(T, max_dd_paths, average_max_dd_per_step)



# In[109]:


get_ipython().run_line_magic('prun', '-s cumulative calculate_max_dd_paths_memoized(10, (0,))')


# In[110]:


get_ipython().system('pip install memory_profiler')
get_ipython().run_line_magic('load_ext', 'memory_profiler')
get_ipython().run_line_magic('memit', 'calculate_max_dd_paths_memoized(10, (0,))')


# In[114]:


import time

for T in range(1, 35):  # Adjust the range as needed
    start_time = time.time()
    mem_usage_before = get_ipython().run_line_magic('memit', '-o -q calculate_max_dd_paths_memoized(T, (0,))')
    duration = time.time() - start_time

    print(f"T={T}: Duration: {duration:.2f} seconds, Memory Usage: {mem_usage_before.mem_usage[0]:.2f} MiB")


# In[ ]:




