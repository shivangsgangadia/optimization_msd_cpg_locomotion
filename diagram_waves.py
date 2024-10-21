#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as pyplot
import numpy
import random


# Modified Kuramoto model: $\dot{\theta_{i}} = \omega_{i} + \sum_{j=1}^{K}K_{ij} \times \sin\left(\theta_{j} - \theta_{i}\right) \times \frac{\texttt{current_spring_length}}{\texttt{rest_length}}$

# In[3]:


n = 2
calc_times = 100
theta = numpy.zeros((n, calc_times))
omega = numpy.array([random.randint(1, 9) for _ in range(n)])
# omega = numpy.array([5 for _ in range(n)])


# In[4]:


k = 10
len_ratio = numpy.array([10, 10])
def d_theta(omega, k, theta_i, theta, l_ratio):
    s = (k / n) * (numpy.sin(theta - theta_i + l_ratio )).sum()
    return (omega * 0.1 + s)

# print(numpy.append(theta[0, :], 1))
# No disturbance: constant length of springs
for i in range(1, 25):
    for j in range(n):
        _d_theta = d_theta(omega[j], k, theta[j, i-1], theta[:, i-1], 0)
        theta[j, i] = _d_theta

# Disturbance: changing spring lengths
for i in range(25, 75):
    for j in range(n):
        _d_theta = d_theta(omega[j], k, theta[j, i-1], theta[:, i-1], random.randint(-5, 5))
        theta[j, i] = _d_theta

# No disturbance: constant length of springs
for i in range(75, 100):
    for j in range(n):
        _d_theta = d_theta(omega[j], k, theta[j, i-1], theta[:, i-1], 0)
        theta[j, i] = _d_theta

        
y_1 = []
y_2 = []
for i in range(100):
    y_1.append(numpy.sin(i))
    y_2.append(numpy.sin(i + numpy.pi/4))


# In[7]:


# pyplot.plot(theta[0], theta[1])
# pyplot.plot(numpy.sin(theta[0]))
# pyplot.plot(numpy.sin(theta[1]))
fig, axes = pyplot.subplots(2, figsize=(15, 10))
axes[1].set_xticks(numpy.linspace(0, 100, 25))
axes[1].set_title("CPG waveform")
for th in theta:
    axes[1].plot(2000 * numpy.sin(th / k))

axes[0].plot(2000 * numpy.array(y_1))
axes[0].plot(2000 * numpy.array(y_2))
axes[0].set_title("Sine waveform")
axes[0].set_xticks(numpy.linspace(0, 100, 25))

for ax in axes.flat:
    ax.set(xlabel="Time", ylabel="Amplitude")
# pyplot.plot(theta[2])
pyplot.show()


# In[ ]:





# In[ ]:




