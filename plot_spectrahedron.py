import numpy as np
import matplotlib.pyplot as plt
import mosek.fusion as mf

# Define SDP problem
n = 2

with mf.Model("SDP") as M:
    # PSD variable X
    X = M.variable(mf.Domain.inPSDCone(n))
    M.objective(mf.ObjectiveSense.Maximize, mf.Expr.dot([[1, 0], [0, 1]], X))
    M.constraint(X.index(0, 1), mf.Domain.equalsTo(1.0))
    M.solve()

    X_sol = X.level()
    X_sol = np.array(X_sol)
    X_sol = X_sol.reshape((n, n))

# Visualize the spectrahedron
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the optimal matrix as a point
eigvals, eigvecs = np.linalg.eigh(X_sol)
ax.scatter(eigvals[0], eigvals[1], X_sol, color="r", label="Optimal Point")

# Add labels and legend
ax.set_xlabel("Eigenvalue 1")
ax.set_ylabel("Eigenvalue 2")
ax.set_zlabel("Optimal Value")
ax.legend()

plt.show()
