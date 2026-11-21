import time
import numpy as np


def BCO(X, objective_function, VRmin, VRmax, max_iter):

    num_collies = X.shape[0]
    num_variables = X.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    fitness = np.array([objective_function(ind) for ind in X])
    best_index = np.argmin(fitness)
    X_best = X[best_index].copy()
    best_score = fitness[best_index]

    Convergence_curve = np.zeros((max_iter, 1))
    start_time = time.time()
    for t in range(max_iter):
        print(f"Iteration {t + 1}/{max_iter}")

        alpha = 2 * np.exp(-((4 * t / max_iter) ** 2))
        beta = np.exp(-t / (max_iter / 10))
        gamma = np.random.uniform(0.5, 1.5)
        sorted_indices = np.argsort(fitness)
        best = X[sorted_indices[0]]
        middle = X[sorted_indices[num_collies // 2]]
        worst = X[sorted_indices[-1]]

        for i in range(num_collies):
            r1 = np.random.rand(num_variables)
            r2 = np.random.rand(num_variables)
            r3 = np.random.rand(num_variables)

            D_best = abs(r1 * best - X[i])
            D_mid = abs(r2 * middle - X[i])
            D_worst = abs(r3 * worst - X[i])

            X_new = (
                X[i]
                + alpha * (best - D_best)
                + beta * (middle - D_mid)
                - gamma * (worst - D_worst)
            )

            X_new = np.clip(X_new, lb, ub)
            new_fitness = objective_function(X_new)


            if new_fitness < fitness[i]:
                X[i] = X_new.copy()
                fitness[i] = new_fitness

            if new_fitness < best_score:
                best_score = new_fitness
                X_best = X_new.copy()

        Convergence_curve[t] = best_score

    total_time = time.time() - start_time
    Leader_score = best_score

    return Leader_score, Convergence_curve, X_best, total_time
