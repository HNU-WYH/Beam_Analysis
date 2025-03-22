import numpy as np
import matplotlib.pyplot as plt

# 为方便演示，将 x_{i-1} = 0, x_i = 1, x_{i+1} = 2
# 1) 定义标准的 hat 函数（线性有限元常用）
def hat_function(x):
    """
    A simple piecewise-linear 'hat' centered at x=1 on [0,2].
    hat(0)=hat(2)=0, hat(1)=1.
    """
    if x < 0 or x > 2:
        return 0.0
    elif 0 <= x <= 1:
        return x  # 线性上升
    else:  # 1 <= x <= 2
        return 2 - x  # 线性下降

# 2) 定义分段三次的 \phi_{2i-1}(x) 示例
#   这里给出一个在 [0,2] 上的“驼峰”形多项式，仅做演示
def phi_1(xi):
    return 1.0 - 3.0*(xi**2) + 2.0*(xi**3)

def phi_2(xi):
    return xi*(xi-1)**2

def phi_3(xi):
    return 3.0*(xi**2) - 2.0*(xi**3)

def phi_4(xi):
    return xi**2*(xi-1)

def phi_2i_minus_1(x):
    """
    Piecewise cubic shape with slope ~1 at x=1 (demonstration).
    For x in [0,2], let xi = x/2,
    phi_{2i}(x) = xi*(1 - xi)^2, zero outside [0,2].
    """
    if x < 0 or x > 2:
        return 0.0
    elif x < 1:
        return phi_3(x)
    else:
        return phi_1(x-1)

def phi_2i(x):
    """
    Piecewise cubic shape with slope ~1 at x=1 (demonstration).
    For x in [0,2], let xi = x/2,
    phi_{2i}(x) = xi*(1 - xi)^2, zero outside [0,2].
    """
    if x < 0 or x > 2:
        return 0.0
    elif x < 1:
        return 1 * phi_4(x)
    else:
        return 1 * phi_2(x - 1)

# 生成绘图点
xx = np.linspace(0, 2, 300)
hat_vals = [hat_function(x) for x in xx]
phi_minus_vals = [phi_2i_minus_1(x) for x in xx]
phi_vals = [phi_2i(x) for x in xx]

# 绘图
plt.figure(figsize=(5,2.5))
# plt.plot(xx, hat_vals, label="Hat function", lw=2)
plt.plot(xx, phi_minus_vals, label=r"$\phi_{2i-1}$", lw=2)
plt.plot(xx, phi_vals, label=r"$\phi_{2i}$", lw=2)

# 设置横坐标刻度与标签
plt.xticks([0, 1, 2], [r'$x_{i-1}$', r'$x_i$', r'$x_{i+1}$'])

# plt.title("Basis Functions For Longitudinal Deformation")
# plt.xlabel("x")
plt.ylabel("Function Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
