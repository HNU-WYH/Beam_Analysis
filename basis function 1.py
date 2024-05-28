#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import matplotlib.pyplot as plt

def phi1(xi):
    return 1 - 3*xi**2 + 2*xi**3

def phi2(xi):
    return xi*(xi - 1)**2

def phi3(xi):
    return 3*xi**2 - 2*xi**3

def phi4(xi):
    return xi**2*(xi - 1)

class BasisFunction:
    def __init__(self, i, n, h):
        self.i = i
        self.n = n
        self.h = h

    def __call__(self, x):
        if x < 0 or x > (self.n - 1) * self.h:
            return 0  # out of the boundary

        xi = x / self.h  # convert x to standard coordinate

        # the left side basis
        if self.i == 0:  
            if 0 <= x < self.h:
                return phi1(xi)
            else:
                return 0

        # the left side differential basis
        elif self.i == 1:  
            if 0 <= x < self.h:
                return self.h * phi2(xi)
            else:
                return 0

        # the right side basis
        elif self.i == 2*self.n - 2:  
            if (self.n - 1) * self.h <= x <= self.n * self.h:
                return phi3((x - (self.n - 1) * self.h) / self.h)
            else:
                return 0

        # the right side differential basis
        elif self.i == 2*self.n - 1:  
            if (self.n - 1) * self.h <= x <= self.n * self.h:
                return self.h * phi4((x - (self.n - 1) * self.h) / self.h)
            else:
                return 0

        # inner basis
        else:
            node = (self.i + 1) // 2
            if self.i % 2 == 1:  # basis function
                if (node - 1) * self.h <= x < node * self.h:
                    return phi3((x - (node - 1) * self.h) / self.h)
                elif node * self.h <= x < (node + 1) * self.h:
                    return phi1((x - node * self.h) / self.h)
                else:
                    return 0
            else:  # differential basis function
                if (node - 1) * self.h <= x < node * self.h:
                    return self.h * phi4((x - (node - 1) * self.h) / self.h)
                elif node * self.h <= x < (node + 1) * self.h:
                    return self.h * phi2((x - node * self.h) / self.h)
                else:
                    return 0

def plot_beam(u, n, h):
    # Increase the number of x coordinates for higher resolution
    x = np.linspace(0, n * h, 5000)
    y = np.zeros_like(x)

    for i in range(len(u)):
        basis_func = BasisFunction(i, n, h)
        y += u[i] * np.array([basis_func(xi) for xi in x])
    
    plt.figure(figsize=(10, 6))  # Set figure size
    plt.plot(x, y, label='Piecewise Polynomial')
    plt.xlabel('x')
    plt.ylabel('Displacement')
    plt.title('Beam Displacement')
    plt.legend()
    plt.grid(True)  # Add grid for better visualization
    
    # Increase the number of ticks on x and y axes
    plt.xticks(np.linspace(0, n * h, num=n+1))
    plt.yticks(np.linspace(np.min(y), np.max(y), num=10))
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)
    
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()

def main():
    n = 10  # number of elements
    h = 1.0  # length of each element
    xi = np.linspace(0, n * h, 2 * n)
    u = np.concatenate((np.sin(xi), np.cos(xi) * h))  # values and derivatives of sin function

    plot_beam(u, n, h)

if __name__ == "__main__":
    main()


# In[ ]:




