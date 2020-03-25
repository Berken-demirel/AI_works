import cv2
import numpy as np
import random
import math
from operator import itemgetter


class Individual:

    def __init__(self, num_genes):
        self.Genes = self.create_genes(num_genes)
        self.Gene = self.Gene()
    class Gene:
        def __init__(self):
            random.seed(random.random())
            while True:
                self.x = random.randint(1, 180)
                self.y = random.randint(1, 180)
                self.radius = random.randint(1, 40)
                if abs(self.x - self.radius) < 180 or abs(self.y - self.radius < 180):
                    break
            self.R = random.randint(0, 255)
            self.G = random.randint(0, 255)
            self.B = random.randint(0, 255)
            self.A = random.random()

        def guided_mutate(self):
            self.x = random.randint(0, self.x + 180 / 4) if self.x - 180/4 <= 0 else random.randint(self.x - 180 / 4, self.x + 180 / 4)
            self.y = random.randint(0, self.y + 180 / 4) if self.y - 180/4 <= 0 else random.randint(self.y - 180 / 4, self.y + 180 / 4)
            self.radius = random.randint(1, self.radius + 10) if self.radius - 10 <= 0 else random.randint(self.radius - 10, self.radius + 10)
            self.R = random.randint(0, self.R + 64) if self.R - 64 <= 0 else random.randint(self.R - 64, self.R + 64)
            self.G = random.randint(0, self.G + 64) if self.G - 64 <= 0 else random.randint(self.G - 64, self.G + 64)
            self.B = random.randint(0, self.B + 64) if self.B - 64 <= 0 else random.randint(self.B - 64, self.B + 64)
            self.A = random.uniform (0, self.A + 0.25) if self.A - 0.25 <= 0 else random.uniform(self.A - 0.25, self.A + 0.25)

    def create_genes(self, num_genes):
        genes = {}
        output = []
        while num_genes != 0:
            d = self.Gene()
            inserted_element = {d: d.radius}
            genes.update(inserted_element)
            num_genes -= 1
        # Sort the genes according to their radius
        a = sorted(genes.items(), key=lambda x: x[1], reverse=True)
        for a_tuple in a:
            output.append(a_tuple[0])

        return output

    def order_my_genes(self):
        genes = {}
        counter = 0
        while len(self.Genes) != counter:
            d = self.Genes[counter]
            inserted_element = {d: d.radius}
            genes.update(inserted_element)
            counter += 1
        # Sort the genes according to their radius
        a = sorted(genes.items(), key=lambda x: x[1], reverse=True)
        counter = 0
        for a_tuple in a:
            self.Genes[counter] = (a_tuple[0])
            counter += 1

    def draw_image(self, source_image):
        self.order_my_genes()
        shape_of_image = source_image.shape[0]
        image = np.zeros((shape_of_image, shape_of_image, 3), np.uint8)
        image[:] = (255, 255, 255)
        output = image.copy()
        for k in self.Genes:
            overlay = output
            cv2.circle(overlay, (k.x, k.y), k.radius, (k.B, k.G, k.R), -1)
            output = cv2.addWeighted(overlay, k.A, output, 1 - k.A, 0)

        return output

    def do_crossover(self, parent2):
        num_gene = len(self.Genes)
        child1 = Individual(num_gene)
        child2 = Individual(num_gene)
        counter = 0
        while counter != num_gene:
            prob = random.randint(0, 1)
            if prob == 0:
                child1.Genes[counter] = self.Genes[counter]
                child2.Genes[counter] = parent2.Genes[counter]
            else:
                child1.Genes[counter] = parent2.Genes[counter]
                child2.Genes[counter] = self.Genes[counter]
            counter += 1
        return child1, child2

    def do_mutation(self, mutation_type):
        selected_gene = random.randint(0, len(self.Genes) - 1)
        if mutation_type == 0:
            new_gene = self.Gene
            self.Genes[selected_gene] = new_gene
        else:
            self.Genes[selected_gene].guided_mutate()


def initialize_population(num_inds=20, num_genes=50):
    individuals = []
    while num_inds != 0:
        individuals.append(Individual(num_genes))
        num_inds -= 1

    return individuals


def calculate_fitness(individual, source_image):
    image_of_individual = np.int64(individual.draw_image(source_image))
    source_image = np.int64(source_image)
    f = 0
    for i in range(0, 3):
        individual_single_channel = image_of_individual[:, :, i].flatten()
        source_single_channel = source_image[:, :, i].flatten()
        f += -1 * sum((source_single_channel - individual_single_channel) ** 2)
        i += 1
    return f


def selection(Fitness_function, individuals, frac_elites=0.2, tm_size=5):
    number_of_next_generation = np.ceil(len(individuals) * frac_elites)
    output_individuals = []
    output_fitness = []
    while number_of_next_generation != 0:
        index = [i for i, x in enumerate(Fitness_function) if x == max(Fitness_function)]
        index = int(index[0])
        output_individuals.append(individuals[index])
        output_fitness.append(Fitness_function[index])
        del individuals[index]
        Fitness_function = np.delete(Fitness_function, [index, index])
        number_of_next_generation -= 1

    return output_fitness, output_individuals, Fitness_function, individuals


def tournament(individuals, Fitness_function, tm_size=5):
    output_individuals = []
    output_fitness = []
    counter = 0
    while counter != len(individuals):
        participants = random.sample(range(0, len(individuals)), tm_size)
        tournament_fitness = list(Fitness_function[participants])
        tournament_individuals = itemgetter(*participants)(individuals)
        best_of_tournament = np.argmax(tournament_fitness)
        output_individuals.append(tournament_individuals[best_of_tournament])
        output_fitness.append(tournament_fitness[best_of_tournament])
        counter += 1

        # deleted_index = np.where((Fitness_function == tournament_fitness[best_of_tournament]))
        # deleted_index = int(deleted_index[0])
        # del individuals[deleted_index]
        # Fitness_function = np.delete(Fitness_function, [deleted_index, deleted_index])

    return output_fitness, output_individuals


def crossover(fitness, population, frac_parents=0.6):
    number_of_parents = np.floor(len(population) * frac_parents)
    number_of_parents = number_of_parents if number_of_parents % 2 == 0 else number_of_parents - 1
    childs = []
    while number_of_parents != 0:
        best_of_parent_index = np.argmax(fitness)
        parent1 = population[best_of_parent_index]
        del population[best_of_parent_index]
        fitness = np.delete(fitness, [best_of_parent_index, best_of_parent_index])

        best_of_parent_index = np.argmax(fitness)
        parent2 = population[best_of_parent_index]
        del population[best_of_parent_index]
        fitness = np.delete(fitness, [best_of_parent_index, best_of_parent_index])

        child1, child2 = parent1.do_crossover(parent2)
        childs.append(child1)
        childs.append(child2)
        number_of_parents -= 2

    return childs, population


def mutation(individuals, mutation_prob, mutation_type):
    number_individuals = len(individuals)
    counter = 0
    while counter != number_individuals:
        processed_individual = individuals[counter]
        if random.random() < mutation_prob:
            processed_individual.do_mutation(mutation_type)
            individuals[counter] = processed_individual
        counter += 1


source_image = cv2.imread("painting.png")

population = initialize_population(num_inds=20, num_genes=50)

num_generations = 100001
i = 0
fitness_plot = np.zeros(100)
while i != num_generations:
    Fitness_of_individuals = np.zeros(len(population))
    counter = 0
    # Evaluate all the individuals
    for x in population:
        Fitness_of_individuals[counter] = calculate_fitness(x, source_image)
        counter += 1

    if i % 1000 == 0:
        best_individual_index = np.argmax(Fitness_of_individuals)
        best_individual = population[best_individual_index]
        fitness_plot[int(i / 1000)] = Fitness_of_individuals[best_individual_index]
        best_individual_image = best_individual.draw_image(source_image)
        cv2.imwrite('img' + str(i) + '.png', best_individual_image)

    # Select individuals
    best_fitness, best_population, to_tournament_fitness, to_tournament_individuals = selection(Fitness_of_individuals,
                                                                                                population,
                                                                                                frac_elites=0.2,
                                                                                                tm_size=5)

    # Go to Tournament
    other_fitness, other_population = tournament(to_tournament_individuals, to_tournament_fitness, tm_size=5)
    # Do Crossovers
    childs, old_population = crossover(other_fitness, other_population, frac_parents=0.6)
    # Mutation
    mutation_population = childs + old_population
    mutation(mutation_population, mutation_prob=0.2, mutation_type=0)
    # New Population
    new_population = best_population + mutation_population
    population = new_population
    i += 1
