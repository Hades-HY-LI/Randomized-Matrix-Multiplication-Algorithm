import numpy as np
import pandas as pd
from randmatmulti.function import rmm
from tabulate import tabulate
import matplotlib.pyplot as plt
import timeit

# Read two matrices
A = pd.read_csv("STA243_homework_1_matrix_A.csv", header = None)
B = pd.read_csv("STA243_homework_1_matrix_B.csv", header = None)

# Get the dimension of each matrix
m, n = A.shape
n, p = B.shape

# Calculate pk
A_2norm = np.linalg.norm(A, axis=0)  # Calculate L2 norm of each column vector
B_2norm = np.linalg.norm(B, axis=1)  # Calculate L2 norm of each row vector
pk =  A_2norm * B_2norm / np.dot(A_2norm, B_2norm)  # Calculate pk vector

r = np.array([20, 50, 100, 200])
# Initializing the set of M matrix
M_set = np.empty((r.shape[0], m, p))
for i in np.arange(r.shape[0]):
    M_set[i, :, :] = rmm(r[i], m, n, p, A, B, pk)

error_Fnorm = []
for i in np.arange(r.shape[0]):
    error_Fnorm = np.append(error_Fnorm, np.linalg.norm(M_set[i, :, :] - np.dot(A, B)))
# Denominator of the error
A_Fnorm = np.linalg.norm(A)
B_Fnorm = np.linalg.norm(B)

# Calculate the relative approximation error
apprerr = error_Fnorm/(A_Fnorm * B_Fnorm)

# Show the result in a table
mydata = [['r = 20', 'r = 50', 'r = 100', 'r = 200'],
          apprerr]
# display table
print(tabulate(mydata))

# Question (d)
figure, axis = plt.subplots(2, 2)
axis[0, 0].matshow(M_set[0, :, :])
axis[0, 0].set_title('Plot when r = 20')

axis[0, 1].matshow(M_set[1, :, :])
axis[0, 1].set_title('Plot when r = 50')

axis[1, 0].matshow(M_set[2, :, :])
axis[1, 0].set_title('Plot when r = 100')

axis[1, 1].matshow(M_set[3, :, :])
axis[1, 1].set_title('Plot when r = 200')

# Save the plot of Question 4.(d)
plt.savefig('Q4_d.png')

# Combine all the operations and display
plt.show()
