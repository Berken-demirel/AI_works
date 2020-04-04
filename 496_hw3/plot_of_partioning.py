import matplotlib.pyplot as plt
import numpy as np


def output_controller_good(x):
    if 0 >= x >= -0.15/2:
        return_value = (x * 2 / 0.15) + 1
    elif x <= -0.15/2 or x >= 0.15/2:
        return_value = 0
    else:
        return_value = -(x * 2 / 0.15) + 1
    return return_value


def output_controller_high(x):
    if x >= 0 or x < -0.15:
        return_value = 0
    elif -0.15/2 >= x >= -0.15:
        return_value = 2 * x / 0.15 + 2
    else:
        return_value = - (2 * x / 0.15)
    return return_value


def output_controller_small(x):
    if x <= 0 or x > 0.15:
        return_value = 0
    elif 0.15/2 >= x >= 0:
        return_value = 2 * x / 0.15
    else:
        return_value = - (2 * x / 0.15) + 2
    return return_value


t = np.linspace(-0.2, 0.2, 200)

counter = 0
good_triangle = np.zeros(shape=t.shape)
high_triangle = np.zeros(shape=t.shape)
small_triangle = np.zeros(shape=t.shape)
for i in t:
    good_triangle[counter] = output_controller_good(i)
    high_triangle[counter] = output_controller_high(i)
    small_triangle[counter] = output_controller_small(i)
    counter += 1

plt.ylabel("Infection rate")
plt.xlabel("Steps")
plt.plot(t, small_triangle, label='Small Triangle')
plt.plot(t, good_triangle, label='Good Triangle')
plt.plot(t, high_triangle, label='High Triangle')
plt.legend(loc="upper right")


plt.show()