# Cognitive Computing – Assignment 6
# Name: Madhav Narang
# Roll Number: 1024170287



import numpy as np

# Initial sensor readings
temperature = np.array([25, 28, 31, 35, 38, 27, 33, 40])

# a. Add 2°C to every reading using vectorization
corrected_temp = temperature + 2
print("a. Corrected temperatures (+2°C):", corrected_temp)

# b. Convert all temperatures from Celsius to Fahrenheit
fahrenheit_temp = (9 / 5) * corrected_temp + 32
print("b. Temperatures in Fahrenheit:", fahrenheit_temp)

# c. Identify all readings greater than 32°C using Boolean indexing
# Note: Applying this to the corrected temperature array
greater_than_32 = corrected_temp[corrected_temp > 32]
print("c. Readings greater than 32°C:", greater_than_32)

# d. Count how many readings exceed 32°C
count_exceed_32 = np.sum(corrected_temp > 32)
print("d. Number of readings exceeding 32°C:", count_exceed_32)

# e. Explanation of Efficiency:
"""
e. Explanation:
Explicitly iterating through an array using a 'for' loop in Python introduces
significant overhead because Python is an interpreted language. The loop executes
at the Python layer, checking types and handling objects dynamically at each step.

Vectorization and Boolean indexing offload these computations to highly optimized,
compiled C code under the hood of NumPy. This allows operations to run at near-hardware
speeds using SIMD (Single Instruction, Multiple Data) processor instructions.
Additionally, NumPy processes continuous memory blocks, maximizing CPU cache utilization
and drastically speeding up data processing.
"""

########################################################################

import numpy as np

# Step tracking matrix (4 users x 3 days)
steps = np.array([
    [7000, 7500, 8000],
    [10000, 11000, 10500],
    [5000, 5500, 6000],
    [9000, 8500, 9500]
])

# a. Calculate the total steps recorded
total_steps = np.sum(steps)
print("a. Total steps recorded:", total_steps)

# b. Calculate the mean number of steps
mean_steps = np.mean(steps)
print("b. Mean number of steps:", mean_steps)

# c. Find the maximum and minimum values
max_steps = np.max(steps)
min_steps = np.min(steps)
print(f"c. Maximum steps: {max_steps}, Minimum steps: {min_steps}")

# d. Calculate the total steps for each day using axis=0
daily_total_steps = np.sum(steps, axis=0)
print("d. Total steps for each day (Day 1, 2, 3):", daily_total_steps)

# e. Calculate the total steps for each user using axis=1
user_total_steps = np.sum(steps, axis=1)
print("e. Total steps for each user (User 1, 2, 3, 4):", user_total_steps)

# f. Find the user/day position containing the maximum number of steps using argmax()
flat_max_idx = np.argmax(steps)
# Convert flat index back to 2D coordinates (row, col)
user_idx, day_idx = np.unravel_index(flat_max_idx, steps.shape)
print(f"f. Maximum steps position: Row {user_idx} (User {user_idx+1}), Column {day_idx} (Day {day_idx+1})")

########################################################################

import numpy as np

# a. Create the original array
original = np.array([1, 2, 3, 4, 5, 6])
print("a. Original array:", original)

# b. Create a slice from index 1 to 4 (exclusive of 4)
subset = original[1:4]
print("b. Subset slice:", subset)

# c. Modify the first element of subset to 999 and display both
subset[0] = 999
print("c. After modifying subset:")
print("   Subset:", subset)
print("   Original (affected because slicing creates a view):", original)

# Reset original array for clean demonstration
original = np.array([1, 2, 3, 4, 5, 6])

# d. Create another slice using .copy() and modify it
copied_array = original[1:4].copy()
copied_array[0] = 500
print("d. After modifying explicit copy:")
print("   Copied array:", copied_array)
print("   Original (unaffected because copy creates a new allocation):", original)

# e. Create 1 to 12 array and reshape it into a 3x4 matrix
matrix_3x4 = np.arange(1, 13).reshape(3, 4)
print("\ne. Reshaped 3x4 Matrix:\n", matrix_3x4)

# f. NumPy indexing and slicing extractions
print("\nf. Extractions:")
print("   First row:", matrix_3x4[0, :])
print("   Last row:", matrix_3x4[-1, :])
print("   Second column:", matrix_3x4[:, 1])
# Rows 1-2 (indices 1,2) and Columns 2-3 (indices 2,3) -> Python ranges are exclusive at the stop boundary
print("   Elements from rows 1-2 and columns 2-3:\n", matrix_3x4[1:3, 2:4])

# g. Create flattened versions using flatten() and ravel()
flattened = matrix_3x4.flatten()
raveled = matrix_3x4.ravel()
print(f"\ng. Flattened: {flattened}")
print(f"   Raveled: {raveled}")

# h. Modify an element in ravel() and observe matrix_3x4
raveled[0] = 99
print("\nh. Modifying ravel()[0] to 99...")
print("   Original 3x4 Matrix (affected because ravel returns a view whenever possible):\n", matrix_3x4)

# i. Modify an element in flatten() and observe matrix_3x4
flattened[1] = 88
print("\ni. Modifying flatten()[1] to 88...")
print("   Original 3x4 Matrix (unaffected because flatten always returns a deep copy):\n", matrix_3x4)

# j. Display metadata of the 3x4 matrix
print("\nj. Matrix Properties:")
print("   Shape:", matrix_3x4.shape)
print("   Number of dimensions (ndim):", matrix_3x4.ndim)
print("   Total size:", matrix_3x4.size)
print("   Data type (dtype):", matrix_3x4.dtype)

########################################################################

import numpy as np

# Inputs
y = np.array([40, 65, 30, 85])
X = np.array([
    [6, 50, 9],
    [7, 60, 7],
    [5, 45, 6],
    [4, 30, 8]
])

# a. Display shape and dimensions of X
print("a. X Shape:", X.shape)
print("   X Dimensions:", X.ndim)

# b. Calculate X.T and explain
X_T = X.T
print("\nb. Transpose X.T:\n", X_T)
"""
Explanation: The transpose swaps the rows and columns of X.
In the context of the data, while each row of X represented a specific user's features,
each row of X.T now isolates a specific feature (Row 0 = Sleep hours, Row 1 = Activity level,
Row 2 = Stress level) across all users.
"""

# c. Calculate the matrix product: X.T @ X
XTX = X_T @ X
print("\nc. Matrix Product (X.T @ X):\n", XTX)

# d. Calculate the matrix inverse using np.linalg.inv()
XTX_inv = np.linalg.inv(XTX)
print("\nd. Inverse Matrix (X.T @ X)^-1:\n", XTX_inv)

# e. Implement the Ordinary Least Squares equation: beta = (X.T @ X)^-1 @ X.T @ y
beta = XTX_inv @ X_T @ y
print("\ne. Calculated OLS Coefficients (beta):", beta)

# f. Explain what the resulting coefficients represent
"""
f. Coefficient Explanation:
The array contains 3 values [beta_0, beta_1, beta_2] mapping directly to the features:
1. beta_0 (Sleep hours): Weight impact of sleep duration on the assistance score.
2. beta_1 (Activity level): Weight impact of daily physical activity on the score.
3. beta_2 (Stress level): Weight impact of psychological stress on the score.

Each coefficient details how much the predicted assistance score changes when that
specific feature increases by 1 unit, keeping all other features constant.
"""

# g. Predict assistance score for a new user
new_user = np.array([5, 40, 7])
predicted_score = new_user @ beta
print(f"\ng. Predicted assistance score for new user {new_user}: {predicted_score:.2f}")
