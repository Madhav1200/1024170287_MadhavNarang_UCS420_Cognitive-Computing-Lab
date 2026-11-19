# Cognitive Computing – Assignment 5
# Name: Madhav Narang
# Roll Number: 1024170287

import numpy as np

# Q1. NumPy Array Operations
arr = np.array([10, 20, 30, 40, 50])
print("Q1. Original Array:", arr)
print("a) After adding 2:", arr + 2)
print("b) After multiplying by 3:", arr * 3)
print("c) After dividing by 2:", arr / 2)

# Q2. Reverse an Array and Find the Most Frequent Value
arr = np.array([1, 2, 3, 6, 4, 5])
print("\nQ2. a) Reversed array:", arr[::-1])

def find_most_frequent(arr_input):
    counts = np.bincount(arr_input)
    most_frequent_val = np.argmax(counts)
    indices = np.where(arr_input == most_frequent_val)[0]
    return most_frequent_val, indices

x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
val_x, idx_x = find_most_frequent(x)
print("b) i. For array x: Most frequent value is", val_x, "at indices", idx_x)

y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
counts_y = np.bincount(y)
max_count = np.max(counts_y)
modes = np.where(counts_y == max_count)[0]
for mode in modes:
    idx_y = np.where(y == mode)[0]
    print("b) ii. For array y: Most frequent value is", mode, "at indices", idx_y)

# Q3. Access Elements in a 2D Array
arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("\nQ3. a) Element at 1st row, 2nd column:", arr[0, 1])
print("b) Element at 3rd row, 1st column:", arr[2, 0])

# Q4. Array Properties and Transpose
madhavnarang = np.linspace(10, 100, 25)
print("\nQ4. Array details:")
print("Dimensions (ndim):", madhavnarang.ndim)
print("Shape:", madhavnarang.shape)
print("Total elements (size):", madhavnarang.size)
print("Data type (dtype):", madhavnarang.dtype)
print("Total bytes consumed:", madhavnarang.nbytes)

reshaped_transpose = madhavnarang.reshape(-1, 1)
print("\nTranspose using reshape(-1, 1):\n", reshaped_transpose)
print("\nCan we use .T on 1D directly?")
print("madhavnarang.T shape:", madhavnarang.T.shape)
print("A 1D array's .T remains 1D; reshape(-1, 1) makes it a column.")

# Q5. Array Statistics, Reshape and Resize
ucs420_madhavnarang = np.array([[10, 20, 30, 40],
                                [50, 60, 70, 80],
                                [90, 15, 20, 35]])
print("\nQ5. Original 3x4 Array:\n", ucs420_madhavnarang)
print("\n--- Statistics ---")
print("Mean:", np.mean(ucs420_madhavnarang))
print("Median:", np.median(ucs420_madhavnarang))
print("Max:", np.max(ucs420_madhavnarang))
print("Min:", np.min(ucs420_madhavnarang))
print("Unique elements:", np.unique(ucs420_madhavnarang))

reshaped_ucs420_madhavnarang = ucs420_madhavnarang.reshape(4, 3)
print("\n--- Reshaped Array (4x3) ---\n", reshaped_ucs420_madhavnarang)

resized_ucs420_madhavnarang = np.resize(ucs420_madhavnarang, (2, 3))
print("\n--- Resized Array (2x3) ---\n", resized_ucs420_madhavnarang)
