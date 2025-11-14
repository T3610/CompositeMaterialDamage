import numpy as np

# Example array
arr = np.array([
    [5, 12, 25],
    [19, 8, 15],
    [10, 22, 17]
])
# Replace values between 10 and 20 with 15
arr[(arr >= 10) & (arr <= 20)] = 15

print(arr)