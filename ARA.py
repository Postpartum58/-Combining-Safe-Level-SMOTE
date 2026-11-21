import time
import numpy as np


def ARA(X, objective_function, VRmin, VRmax, max_iter):

    num_addaxes = X.shape[0]
    num_variables = X.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    fitness = np.array([objective_function(ind) for ind in X])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    best_score = fitness[best_idx]
    Convergence_curve = np.zeros((max_iter, 1))
    start_time = time.time()
    for t in range(max_iter):
        alpha = 2 - 2 * (t / max_iter)
        beta = 1 - (t / max_iter)
        rho = np.random.uniform(0, 1)

        for i in range(num_addaxes):
            if np.random.rand() < 0.5:
                rand_idx = np.random.randint(0, num_addaxes)
                rand_addax = X[rand_idx]
                r1 = np.random.rand()
                r2 = np.random.rand()

                step = alpha * r1 * (rand_addax - r2 * X[i])
                X_new = X[i] + step
            else:
                step = beta * np.random.randn(num_variables) * (X_best - X[i])
                X_new = X[i] + rho * step
            X_new = np.clip(X_new, lb, ub)
            new_fit = objective_function(X_new)
            if new_fit < fitness[i]:
                fitness[i] = new_fit
                X[i] = X_new.copy()

                if new_fit < best_score:
                    best_score = new_fit
                    X_best = X_new.copy()

        Convergence_curve[t] = best_score
    total_time = time.time() - start_time
    Leader_score = best_score

    return Leader_score, Convergence_curve, X_best, total_time
