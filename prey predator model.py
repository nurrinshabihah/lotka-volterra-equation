#!/usr/bin/env python
# coding: utf-8

# # Solution

# In[33]:


from solvers import solver
import numpy as np
import matplotlib.pyplot as plt
import time
from tabulate import tabulate
import pandas as pd


# ## Implementation
# 
# The Lotka-Volterra equations for a prey-predator system are given by:
# 
# \begin{align*}
# \frac{dx}{dt} &= \alpha x - \beta xy + f(t) \\
# \frac{dy}{dt} &= \delta xy - \gamma y + g(t)
# \end{align*}
# 
# where:
# 
#   $x$ is the prey population, 
#   
#   $y$ is the predator population, 
#   
#   $\alpha, \beta, \gamma$ and $\delta$ are positive real numbers representing various rates,
#   
#   and $f$ and $g$ are given functions of $t$ which relate to external migration into or out of the system.
# 
# This system of differential equation will be solved by three methods for arbitrary initial conditions $x(t=0), y(t=0)$. The chosen methods are the midpoint, Van der Houwen and Ralston-4 method where all of them are the family of the Runge-Kutta iterative methods. 
# 

# Two test cases are considered when solving this problem. The test cases are defined as follows:

# In[34]:


# Define test case 1
def rhs1(t: np.double, y: np.ndarray) -> np.ndarray:
    alpha = 1.0
    beta = 1.0
    gamma = 1.0
    delta = 1.0
    
    f_t = - np.sin(t) - np.cos(t)**2 - np.cos(t)
    g_t = np.sin(t) + np.cos(t)**2 - np.cos(t)
    
    x_prime = alpha*y[0] - beta*y[0]*y[1] + f_t
    y_prime = delta*y[0]*y[1] - gamma*y[1] + g_t
    
    return np.array([x_prime, y_prime])

# Define test case 2
def rhs2(t: np.double, y: np.ndarray) -> np.ndarray:
    alpha = 2/3
    beta = 4/3
    gamma = 1.0
    delta = 1.0
    
    f_t = 0.0
    g_t = 0.0
    
    x_prime = alpha*y[0] - beta*y[0]*y[1] + f_t
    y_prime = delta*y[0]*y[1] - gamma*y[1] + g_t
    
    return np.array([x_prime, y_prime])

# Solving differential equation using midpoint method
def midpoint_method(rhs, t0, y0, dt, T):
    num_steps = int((T - t0) / dt)
    t_values = np.linspace(t0, T, num_steps + 1)
    y_values = np.zeros((num_steps + 1, len(y0)))
    y_values[0] = y0

    for i in range(1, num_steps + 1):
        t_half = t_values[i - 1] + 0.5 * dt
        y_half = y_values[i - 1] + 0.5 * dt * rhs(t_values[i - 1], y_values[i - 1])
        y_values[i] = y_values[i - 1] + dt * rhs(t_half, y_half)

    return t_values, y_values


# Plot prey and predator graph
def plot_comparison(dt_values, results, title, position):
    plt.subplot(2, 1, position)
    
    for i, dt in enumerate(dt_values):
        result = results[i]
        t = result[0]
        y = result[1]
        plt.plot(t, np.array(y)[:, 0], label=f"Prey (x), dt = {dt}")
        plt.plot(t, np.array(y)[:, 1], label=f"Predator (y), dt = {dt}")

    plt.xlabel('Time (t)')
    plt.ylabel('Population')
    plt.title(title)
    plt.legend()


# ## Results
# 
# To evaluate the solutions of the given differential equations, we conducted a comprehensive analysis by plotting Population against Time, $t$  for various time steps $\mathrm{d}t$. This experiments involved a range of time step values: 
# $\mathrm{d}t = T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400$. 
# 
# For test case A, we are using $T = 2.5\pi$ and for test case B, $T = 30$. 
# 
# In each test case, the solutions for both prey and predator populations were simultaneously graphed. This approach allows for a nuanced exploration of the dynamic relationship between these two subjects over time. The visual representation not only captures the evolution of each population individually but also provides insights into their mutual interactions. 

# In[35]:


# Midpoint method
plt.figure(figsize=(15, 10))
plt.suptitle('Midpoint Method for Comparison of Predator Population with Different Time Step Sizes')

# Test case 1 
T = 2.5 * np.pi
dt_values = [T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400]
results1 = [midpoint_method(rhs1, 0.0, np.array([2.0, 0.0]), dt, T) for dt in dt_values]
plot_comparison(dt_values, results1, 'Test Case A', 1)

# Test case 2 
T = 30
dt_values = [T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400]
results2 = [midpoint_method(rhs2, 0.0, np.array([0.9, 0.9]), dt, T) for dt in dt_values]
plot_comparison(dt_values, results2, 'Test Case B', 2)

plt.tight_layout()
plt.show()

# Van der Houwen method
plt.figure(figsize=(15, 10))
plt.suptitle('Van der Houwen Method for Comparison of Predator Population with Different Time Step Sizes')

# Test case 1
T = 2.5 * np.pi
dt_values = [T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400]
results3 = [solver(rhs1, np.array([2.0, 0.0]), 0.0, dt, T, "Van der Houwen") for dt in dt_values]
plot_comparison(dt_values, results3, 'Test Case A', 1)

# Test case 2 
T = 30
dt_values = [T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400]
results4 = [solver(rhs2, np.array([0.9, 0.9]), 0.0, dt, T, "Van der Houwen") for dt in dt_values]
plot_comparison(dt_values, results4, 'Test Case B', 2)

plt.tight_layout()
plt.show()

# Ralston-4 method
plt.figure(figsize=(15, 10))
plt.suptitle('Ralston-4 Method for Comparison of Predator Population with Different Time Step Sizes')

# Test case 1 
T = 2.5 * np.pi
dt_values = [T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400]
results5 = [solver(rhs1, np.array([2.0, 0.0]), 0.0, dt, T, "Ralston-4") for dt in dt_values]
plot_comparison(dt_values, results5, 'Test Case A', 1)

# Test case 2 
T = 30
dt_values = [T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400]
results6 = [solver(rhs2, np.array([0.9, 0.9]), 0.0, dt, T, "Ralston-4") for dt in dt_values]
plot_comparison(dt_values, results6, 'Test Case B', 2)

plt.tight_layout()
plt.show()


# Analysis of the prey-predator graph reveals distinctive patterns for each test case and provides insights into the stability and performance of different numerical methods.
# 
# ### Test Case A
# When the prey population is at its maximum, the predator population is at its minimum, resulting in oscillatory behaviour. The solutions obtained using different methods exhibit similar smoothness in the graph, suggesting that the choice of method does not significantly impact the overall solution quality. The time step $T/50$ yields comparable results to other smaller time steps implying a high stability of this solution. The consistency in solution regardless of time step indicates a stable numerical methods for Test Case 1.
# 
# ### Test Case B
# Both prey and predator populations follow similar patterns, with the prey population generally exceeding the predator population, except during specific intervals. Notably, the midpoint method produces a visibly stiffer graph, particularly evident with larger time steps, $T/50$. When comparing the gap between the graphs at $T/50$ and $T/100$ the midpoint method exhibits a larger gap than the Van der Houwen method, while the Ralston-4 method shows the smallest gap, resulting in a smoother graph.
# 
# It can be observed that starting from $T/100$ and smaller, the graphs become smoother and solutions overlap could suggest convergence. This implies that as the time step decreases, the solutions from different methods converge towards a consistent solution. Overall, the Ralston-4 method consistently produces the smoothest graphs, suggesting a more accurate solution. The Van der Houwen method also performs well, while the midpoint method exhibits stiffness, especially with larger time steps.

# As test case B has no exact solution, we cannot find the absolute error. Hence, we find the error of both test cases by comparing the maximum differences between methods for each $dt$.

# In[36]:


dt_values = [T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400]

# results [time[[prey][predator]]]

# Compare solutions for predator at every dt
def predator_differences(result1, result2, title):  
    errors = []
    for i, dt in enumerate(dt_values):
        res1 = result1[i]
        max_value1 = -float('inf')  # Initialize with very low value (negative infinity)
        for array in res1[1]:
            max_value1 = max(max_value1, array[1])  # Update max_value if a higher value is found
        res2 = result2[i]
        max_value2 = -float('inf')  
        for array in res2[1]:
            max_value2 = max(max_value2, array[1])  
        error = abs(max_value1 - max_value2)
        errors.append(error)

    plt.xlabel('Time step (dt)')
    plt.ylabel('Error')
    plt.loglog(dt_values, errors, label=title)
    plt.grid(True)
    

# Compare solutions for prey at every dt
def prey_differences(result1, result2, title):  
    errors = []
    for i, dt in enumerate(dt_values):
        res1 = result1[i]
        max_value1 = -float('inf')  
        for array in res1[1]:
            max_value1 = max(max_value1, array[0])  
        res2 = result2[i]
        max_value2 = -float('inf') 
        for array in res2[1]:
            max_value2 = max(max_value2, array[0])  
        error = abs(max_value1 - max_value2)
        errors.append(error)

    plt.xlabel('Time step (dt)')
    plt.ylabel('Error')
    plt.loglog(dt_values, errors, label=title)
    plt.grid(True)


# In[37]:


df_time = pd.DataFrame (
    {
        "dt": dt_values,  
    }
)

def compare_max_1(results1, results3, result5):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    plt.suptitle("Maximum Difference between Different Methods for Test Case A")
    
    plt.subplot(2, 1, 1)
    plt.title("Prey")
    prey_differences(results1, results3, "diff(midpoint, van der houwen)")
    prey_differences(results1, results5, "diff(midpoint, ralston-4)")
    prey_differences(results5, results3, "diff(van der houwen, ralston-4)")
    plt.loglog(df_time['dt'], df_time['dt']**3, linestyle='--', label="$dt^3$", color='purple')
    plt.loglog(df_time['dt'], df_time['dt']**4, linestyle='--', label="$dt^4$", color='black')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.title("Predator")
    predator_differences(results1, results3, "diff(midpoint, van der houwen)")
    predator_differences(results1, results5, "diff(midpoint, ralston-4)")
    predator_differences(results5, results3, "diff(van der houwen, ralston-4)")
    plt.loglog(df_time['dt'], df_time['dt']**3, linestyle='--', label="$dt^3$", color='purple')
    plt.loglog(df_time['dt'], df_time['dt']**4, linestyle='--', label="$dt^4$", color='black') 
    plt.legend()
    
compare_max_1(results1, results3, results5)


# In[38]:


def compare_max_2(results2, results4, result6):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    plt.suptitle("Maximum Difference between Different Methods for Test Case B")
    
    plt.subplot(2, 1, 1)
    plt.title("Prey")
    prey_differences(results2, results4, "diff(midpoint, van der houwen)")
    prey_differences(results2, results6, "diff(midpoint, ralston-4)")
    prey_differences(results6, results4, "diff(van der houwen, ralston-4)")
    plt.loglog(df_time['dt'], df_time['dt']**4, linestyle='--', label="$dt^4$", color='black')
    plt.loglog(df_time['dt'], df_time['dt']**3, linestyle='--', label="$dt^3$", color='purple')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Predator")
    predator_differences(results2, results4, "diff(midpoint, van der houwen)")
    predator_differences(results2, results6, "diff(midpoint, ralston-4)")
    predator_differences(results6, results4, "diff(van der houwen, ralston-4)")
    plt.loglog(df_time['dt'], df_time['dt']**4, linestyle='--', label="$dt^4$", color='black')
    plt.loglog(df_time['dt'], df_time['dt']**3, linestyle='--', label="$dt^3$", color='purple')
    plt.legend()
    

compare_max_2(results2, results4, results6)


# The observed trend of larger time steps resulting in increased errors aligns with expectations in numerical analysis. In the context of solving differential equations, larger $dt$ often lead to more significant errors. This is evident in the comparison of different numerical methods. The maximum differences between the midpoint and Van der Houwen method, as well as between the midpoint and Ralston-4, exhibit behavior consistent with an error approximately proportional to $dt^3$. Usually, the error for the midpoint method is approximately proportional to $dt^2$, meaning that each time $dt$ is halved, the error quarters. This shows the improvement of using both Van der Houwen and Ralston-4 methods. Furthermore, the consistent discrepancy between the Van der Houwen and Ralston-4 methods, resembling a $dt^4$ behavior, underscores the higher accuracy of the fourth-order method compared to the third-order one. 

# In practical terms, this trade-off between time step size and accuracy emphasizes the need for careful consideration in selecting an optimal $dt$ to balance computational efficiency with solution precision in differential equation simulations. Here, we compare the runtime of each methods in both test cases with $O(N)$ by averaging the runtime of each method for ten times.

# In[39]:


def time_midpoint1(dt):
    start = time.time()
    T = 2.5 * np.pi
    midpoint_method(rhs1, 0.0, np.array([2.0, 0.0]), dt, T)
    end = time.time() 
    return end - start

def time_midpoint2(dt):
    start = time.time()
    T = 30
    midpoint_method(rhs1, 0.0, np.array([0.9, 0.9]), dt, T)
    end = time.time()
    return end - start

def time_vdh1(dt):
    T = 2.5 * np.pi
    start = time.time()
    solver(rhs1, np.array([2.0, 0.0]), 0.0, dt, T, "Van der Houwen")
    end = time.time() 
    return end - start

def time_vdh2(dt):
    T = 30
    start = time.time()
    solver(rhs1, np.array([0.9, 0.9]), 0.0, dt, T, "Van der Houwen")
    end = time.time()
    return end - start

def time_ralston1(dt):
    T = 2.5 * np.pi
    start = time.time()
    solver(rhs1, np.array([2.0, 0.0]), 0.0, dt, T, "Ralston-4")
    end = time.time() 
    return end - start

def time_ralston2(dt):
    T = 30
    start = time.time()
    solver(rhs1, np.array([0.9, 0.9]), 0.0, dt, T, "Ralston-4")
    end = time.time() 
    return end - start


# In[40]:


dt_values = [T/50, T/100, T/200, T/400, T/800, T/1600, T/3200, T/6400]

n = [int(T/dt) for dt in dt_values]

def time_solve(method):
    results = []
    tests = 10
    
    for dt in dt_values:
        time = 0.0
        for _ in range(tests):
            time += method(dt)
        results.append(time/tests)
    
    return results 

midpoint1 = time_solve(time_midpoint1)
midpoint2 = time_solve(time_midpoint2)
vdh1 = time_solve(time_vdh1)
vdh2 = time_solve(time_vdh2)
ralston1 = time_solve(time_ralston1)
ralston2 = time_solve(time_ralston2)


# In[41]:


# Store data in dataframe for processing
df_time = pd.DataFrame (
    {
        "n": n,
        "dt": dt_values,
        "Midpoint" : midpoint1,
        "van der houwen" : vdh1,
        "Ralston-4": ralston1,   
    }
)

plt.figure(figsize=(15, 7))
plt.title("Comparison of Runtimes of Different Methods for Test Case A")

# Add extra columns for ratios
for col in df_time.columns:
    if "dt" not in col:
        df_time[f"ratio ({col})"] = df_time[col] / df_time[col].shift()
        
plt.loglog(df_time['n'], midpoint1, label="Midpoint")
plt.loglog(df_time['n'], vdh1, label="Van der Houwen")
plt.loglog(df_time['n'], ralston1, label="Ralston")

plt.loglog(df_time['n'], df_time['n'], linestyle='--', label="O(N)", color='black')

plt.xlabel("$N$")
plt.ylabel("run time (s)")
plt.grid(True)
plt.legend()

df_time


# In[42]:


# Store data in dataframe for processing
df_time = pd.DataFrame (
    {
        "n": n,
        "dt": dt_values,
        "Midpoint" : midpoint2,
        "van der houwen" : vdh2,
        "Ralston-4": ralston2,   
    }
)

plt.figure(figsize=(15, 7))
plt.title("Comparison of Runtimes of Different Methods for Test Case B")

# Add extra columns for ratios
for col in df_time.columns:
    if "dt" not in col:
        df_time[f"ratio ({col})"] = df_time[col] / df_time[col].shift()
        
plt.loglog(df_time['n'], midpoint2, label="Midpoint")
plt.loglog(df_time['n'], vdh2, label="Van der Houwen")
plt.loglog(df_time['n'], ralston2, label="Ralston")

plt.loglog(df_time['n'], df_time['n'], linestyle='--', label="O(n)", color='black')

plt.xlabel("$N$")
plt.ylabel("run time (s)")
plt.grid(True)
plt.legend()

df_time


# ## Analysis
# 

# #### Midpoint Method
# Midpoint method, an improvement of the Euler's method with second-order accuracy proves reliable for smaller $dt$. The graph consistently converges to the exact solution for each $dt$, starting from $dt = T/100$ and further decreasing $dt$ yields negligible differences in the solution. Theoratically, the error is proportional to $(dt)^2$ indicating less significant error. While it presents a promising solution, the midpoint method is acknowledged for having lower accuracy compared to other methods.
# 
# The runtime remains exceptionally low compared to $O(N)$ and exhibits stable linear increase as $N$ becomes larger. Theoritically, midpoint methods requires twice the computational work per step. However it can be seen that the computational cost remains low, even for smaller $dt$. In terms of efficiency, midpoint method is the most effective as it displays the lowest average runtime for all $dt$.

# #### Van der Houwen Method
# Van der Houwen method is the third-order Runge-Kutta scheme which offers higher accuracy than midpoint method. The difference in the solution only can be seen when $dt = T/50$ but is less significant than the difference observed in midpoint method. Smaller $dt$ values converge to both exact and approximate solutions more effectively. Error analysis through maximum difference reveals that the error of this method is approximately proportional to $(dt)^3$ which is an improvement from midpoint method. 
# 
# The runtime for Van der Houwen method is slightly greater than midpoint method but remains below $O(N)$. The graph demostrates a linear increase, suggesting  stable and efficient computational performance.

# #### Ralston-4 Method
# The Ralston-4 method, a fourth-order Runge-Kutta scheme, exhibits the highest accuracy among the methods utilised in this investigation. This is evident in the solution graph, illustrating the smoothest results even with larger $dt$ values. Hence, this method provides the most stable solution to this model. Error analysis proves the accuracy of this method, with error proportional to $(dt)^4$ for test case B and even smaller errors in test case A, where an exact solution exists.
# 
# Despite having the highest runtime among all methods for both test cases, it still remains below $O(N)$. This underscores that the Ralston-4 method gives a good balance between accuracy and computational cost. 

# ## Conclusion
# 
# In conclusion, the comparison of numerical methods for solving the prey-predator model reveals distinct characteristics and trade-offs. All three methods employed are advanced versions of Euler's basic first-order approach. While Euler's method offers a quick approximation, it sacrifices solution accuracy, making it suitable for gaining behavioral insights into the prey and predator. However, for quantitative analysis and long-term predictions, higher accuracy becomes essential. Considering both accuracy and computational efficiency, we recommend the Van der Houwen method. It demonstrates comparable stability and accuracy to Ralston-4 while maintaining efficiency in terms of running time.
# 

# ## References
# 
# - [Euler's Method in Python](https://www.youtube.com/watch?v=R9V8NhUof6c&lc=Ugy2MriazEMhp_aQAUd4AaABAg.9xue1g14Kgd9y7fTJhztj0)
# 
# - [Wikipedia: List of Runge–Kutta Methods](https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods)
# 
# - [Runge-Kutta Methods](https://onlinehw.math.ksu.edu/math340book/chap1/xc1.php)
# 
# - [Wikipedia: Runge-Kutta Methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
# 
# - [Wikipedia: Euler method](https://en.wikipedia.org/wiki/Euler_method)
# 
# - [Wikipedia: Numerical methods for ordinary differential equations](https://en.wikipedia.org/wiki/Numerical_methods_for_ordinary_differential_equations)
# 
# - [Wikipedia: Midpoint method](https://en.wikipedia.org/wiki/Midpoint_method)
# 
# - [Wikipedia: Lotka-Volterra equations](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations)
# 
# 
