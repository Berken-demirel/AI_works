from plague import Plague
import numpy as np
import random
import matplotlib.pyplot as plt


def fuzzy_controller(infected_percentage):
    memberships = [calculate_membership_low_infected(infected_percentage),
                   calculate_membership_good_infected(infected_percentage),
                   calculate_membership_high_infected(infected_percentage)]

    best_member = memberships.index(max(memberships))
    if best_member == 0:
        return output_controller_low(memberships[0])
    elif best_member == 1:
        return output_controller_good(memberships[1])
    else:
        return output_controller_high(memberships[2])


def calculate_membership_low_infected(x):
    if x <= 0.4:
        return 1
    elif x >= 0.55:
        return 0
    else:
        slope = -(1 / 0.15)
        value = 1 - min(1, slope * (0.4 - x))
        return value


def output_controller_low(x):
    first_point = (0.15 * x / 2)
    second_point = -((x - 2) * 0.15 / 2)
    return_value = random.uniform(first_point, second_point)
    return return_value


def calculate_membership_good_infected(x):
    if x <= 0.5 or x >= 0.7:
        return 0
    else:
        slope = (1 / 0.1)
        value = 1 - min(1, slope * abs(0.6 - x))
        return value


def output_controller_good(x):
    first_point = (x - 1) * 0.15 / 2
    second_point = -((x - 1) * 0.15 / 2)
    return_value = random.uniform(first_point, second_point)
    return return_value


def calculate_membership_high_infected(x):
    if x >= 0.8:
        return 1
    elif x <= 0.65:
        return 0
    else:
        slope = -(1 / 0.15)
        value = 1 - min(1, slope * (0.8 - x))
        return value


def output_controller_high(x):
    first_point = ((x - 2) * 0.15 / 2)
    second_point = -(0.15 * x / 2)
    return_value = random.uniform(first_point, second_point)
    return return_value


plague = Plague()

number_iterations = 200
counter = 0
infected_bots_plot = np.empty(shape=(200,))
while counter != number_iterations:
    infected_bots, _ = plague.checkInfectionStatus()
    infected_bots_plot[counter] = infected_bots
    control_variable = fuzzy_controller(infected_bots)
    plague.spreadPlague(control_variable)
    counter += 1


plt.ylabel("Infection rate")
plt.xlabel("Steps")
plt.plot(plague.infected_percentage_curve_)
plt.show()

plt.ylabel("Infection cost")
plt.xlabel("Steps")
plt.plot(plague.infection_rate_curve_)
plt.show()

cost_sum = sum(plague.infected_percentage_curve_[1:100])

plague.viewPlague(100, cost_sum)