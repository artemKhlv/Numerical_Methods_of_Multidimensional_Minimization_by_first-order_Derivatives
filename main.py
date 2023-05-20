import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize._linesearch import line_search

call_count_f = 0
call_count_grad = 0


def f(x):
    global call_count_f
    call_count_f += 1
    return 4*pow(x[0], 4) - 6*x[0]*x[1] - 34*x[0] + 5*pow(x[1], 4) + 42*x[1] + 7


def grad_f(x):
    global call_count_grad
    call_count_grad += 1
    return np.array([16*pow(x[0], 3) - 6*x[1] - 34, -6*x[0] + 20*pow(x[1], 3) + 42])


#поиск оптимального шага
def line_search(f, grad_f, x, delta_x):
    alpha = 1.0
    rho = 0.5
    c = 0.1
    phi_0 = f(x)
    dphi_0 = np.dot(grad_f(x), delta_x)
    while True:
        phi_alpha = f(x + alpha * delta_x)
        if phi_alpha > phi_0 + c * alpha * dphi_0:
            alpha *= rho
        else:
            dphi_alpha = np.dot(grad_f(x + alpha * delta_x), delta_x)
            if dphi_alpha < c * dphi_0:
                alpha /= rho
            else:
                break
    return alpha


def fletcher_reeves_method():
    iter_count = 0
    # Шаг 1
    x = x0.copy()
    x_traj = [x.copy()]
    # Шаг 2
    k = 0
    # Шаг 3
    grad_k = grad_f(x)
    delta_k = -grad_k
    while True:
        iter_count += 1
        # Шаг 4
        if np.linalg.norm(grad_k) < eps1:
            break
        # Шаг 5
        if k >= M:
            break
        # Шаг 6
        n = k % 1
        beta_k = 0 if n == 0 else np.square(np.linalg.norm(grad_f(x))) / np.square(np.linalg.norm(grad_f(x_traj[k-1])))
        # Шаг 7
        delta_k = -grad_f(x) + beta_k * delta_k
        # Шаг 8
        alpha_k = line_search(f, grad_f, x, delta_k)
        # Шаг 9
        x = x + alpha_k * delta_k
        # Шаг 10
        if k > 0:
            if np.abs(f(x_traj[k-1]) - f(x)) < eps2 and np.linalg.norm(x_traj[k-1] - x) < eps2:
                break
        # Шаг 11
        k += 1
        x_traj.append(x.copy())
    return x, iter_count, x_traj


x0 = np.array([2, 2])
eps1 = 0.0005
eps2 = 0.0005
M = 1000
min, iters, x_traj = fletcher_reeves_method()
print("Число итераций: ", iters)
print("Количество вычислений функции: ", call_count_f)
print("Количество вычислений градиента функции: ", call_count_grad)
print("Найденное решение (min): ", min)
print("Значение функции: ", f(min))

# print("Траектория движения к экстремуму", x_traj)
# x_traj = np.array(x_traj)
# plt.figure(figsize=(6, 6))
# plt.plot(x_traj[:, 0], x_traj[:, 1], '-o')
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title('Fletcher-Reeves Method')
# plt.show()
# Визуализация
x = np.linspace(-2, 4, 100)
y = np.linspace(-3, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f([X, Y])
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-10, 10, 100), cmap='jet')
plt.plot(*zip(*x_traj), '-o', color='black')
plt.title('Fletcher-Reeves Method', fontsize=14)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.show()