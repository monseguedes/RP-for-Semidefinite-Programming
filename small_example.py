"""
This solves a small example for debugging purposes.
"""

from scipy.optimize import minimize
import monomials

###-------------------###
### Original PO problem ###
###-------------------###

# Objective function: f(x) = x1^2 + x2^2 + 2*x1*x2
def objective(x):
    return x[0]**2 + x[1]**2 + 2 * x[0] * x[1]

# Constraint: inside the unit sphere
def constraint(x):
    return x[0]**2 + x[1]**2 - 1

# Initial guess
initial_guess = [0, 1]

# Define the optimization problem
problem = {
    'fun': objective,
    'x0': initial_guess,
    'constraints': [{'type': 'eq', 'fun': constraint}]
}

# Solve the optimization problem using the SLSQP algorithm (Sequential Least Squares Quadratic Programming)
result = minimize(**problem, method='SLSQP')

# Print the optimal solution
if result.success:
    print("Optimal solution found:")
    print(f"x1 = {result.x[0]}")
    print(f"x2 = {result.x[1]}")
    print(f"f(x) = {result.fun}")
else:
    print("Optimal solution not found")


###-------------------###
### Dual of the SDP relaxation ###
###-------------------###
import cvxpy as cp
import numpy as np

# Define the variables
n = 6  # number of variables
m = 2  # number of constraints

# Variables
y = cp.Variable(6)

# Given data
c = np.array([0, 0, 0, 1, 2, 1])  # coefficients for the objective function
s = np.array([0, 0, 0, 1, 0, 1])  # coefficients for the constraint
# Get list of all the matrices for picking coefficients.
monomial_matrix = monomials.generate_monomials_matrix(2, 2)
distinct_monomials = monomials.get_list_of_distinct_monomials(monomial_matrix)
A = []
for monomial in distinct_monomials:
    A.append(monomials.pick_specific_monomial(monomial_matrix, monomial))
A = np.array(A)
for i in range(len(A)):
    print("---" * 10)
    print("monomial_" + str(i) + ": " + str(distinct_monomials[i]))
    print("A_" + str(i) + ":")
    monomials.print_readable_matrix(A[i])
    print("coefficient_" + str(i) + ": " + str(c[i]))
    print("sphere coefficient_" + str(i) + ": " + str(s[i]))
print("---" * 10)

# Objective function
objective = cp.Maximize(c.T @ y)

# Constraints
constraints = [
    cp.sum(s.T @ y) == 1,  # sum(y_i * s_i) = 1
    - sum(A[i] * y[i] for i in range(n)) >> 0  # - sum(A_i *  y_i) is positive semidefinite
]


# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Print the results
if problem.status == cp.OPTIMAL:
    print("Optimal value =", problem.value)
    print("Optimal y:")
    print(y.value)
else:
    print("Optimization problem not solved successfully.")
    print("Status:", problem.status)

