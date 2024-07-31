import numpy as np
from scipy.linalg import solve

def build_transition_matrix(lambda_rate, mu_fast, mu_slow, c_fast, c_slow, max_queue_length, alpha):
    size = (max_queue_length + 1) ** 2
    Q = np.zeros((size, size))

    for n_fast in range(max_queue_length + 1):
        for n_slow in range(max_queue_length + 1):
            i = n_fast * (max_queue_length + 1) + n_slow
            P_fast = 1 / (1 + np.exp(alpha * (n_fast - n_slow)))
            P_slow = 1 - P_fast
            
            if n_fast > 0:
                j = (n_fast - 1) * (max_queue_length + 1) + n_slow
                Q[i, j] = lambda_rate * P_fast
            if n_slow > 0:
                j = n_fast * (max_queue_length + 1) + (n_slow - 1)
                Q[i, j] = lambda_rate * P_slow
            if n_fast < max_queue_length:
                j = (n_fast + 1) * (max_queue_length + 1) + n_slow
                Q[i, j] = mu_fast * min(n_fast + 1, c_fast)
            if n_slow < max_queue_length:
                j = n_fast * (max_queue_length + 1) + (n_slow + 1)
                Q[i, j] = mu_slow * min(n_slow + 1, c_slow)
            Q[i, i] = -(lambda_rate + min(n_fast, c_fast) * mu_fast + min(n_slow, c_slow) * mu_slow)

    return Q

lambda_rate = 10
mu_fast = 5
mu_slow = 2
c_fast = 3
c_slow = 5
alpha = 0.1
max_queue_length = 10

Q = build_transition_matrix(lambda_rate, mu_fast, mu_slow, c_fast, c_slow, max_queue_length, alpha)
size = Q.shape[0]
Q[-1, :] = 1  # 保证 Q 可解
b = np.zeros(size)
b[-1] = 1

P = solve(Q.T, b)
P = P.reshape((max_queue_length + 1, max_queue_length + 1))

# 计算平均队列长度
L_fast = sum(n_fast * P[n_fast, n_slow] for n_fast in range(max_queue_length + 1) for n_slow in range(max_queue_length + 1))
L_slow = sum(n_slow * P[n_fast, n_slow] for n_fast in range(max_queue_length + 1) for n_slow in range(max_queue_length + 1))

# 计算平均等待时间
W_fast = L_fast / (lambda_rate * (1 / (1 + np.exp(alpha * (L_fast - L_slow)))))
W_slow = L_slow / (lambda_rate * (1 - 1 / (1 + np.exp(alpha * (L_fast - L_slow)))))

print(f"快充桩平均等待时间: {W_fast:.2f} 时间单位")
print(f"慢充桩平均等待时间: {W_slow:.2f} 时间单位")

