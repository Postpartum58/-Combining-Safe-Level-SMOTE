import time
import numpy as np

# Improved Addax Optimization Algorithm formula updation done in line 25
def Proposed(X, objective_function, VRmin, VRmax, max_iter):

    num_rabbits = X.shape[0]
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

        a = 2 * (1 - (t / max_iter))
        b = 1.5 * np.exp(-t / max_iter)

        for i in range(num_rabbits):
            c = fitness[i] / (fitness[i] + np.max(fitness))
            if np.random.rand() < 0.5:
                rand_rabbit = X[np.random.randint(0, num_rabbits)]
                step = a * np.random.randn(num_variables) * (rand_rabbit - X[i])
                X_new = X[i] + step
            else:
                jump_strength = b * np.random.randn(num_variables)
                X_new = X[i] + c * jump_strength * (X_best - X[i])
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
