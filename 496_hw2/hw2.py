import cv2
import numpy as np
import random
from operator import itemgetter

# Class for each Individual
class Individual:

    def __init__(self, num_genes):
        self.Genes = self.create_genes(num_genes)
        self.Gene = self.Gene()

    # Each individual is composed of Gene which has 7 features
    class Gene:
        def __init__(self):
            random.seed(random.random())
            # Until the circle is within the image, change the x,y and radius values.
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

    # Guided mutation, change the gene values according to its previous values
    # This code is quite long since we have to change the features if they are not valid.
    def guided_mutate(self, guided_gene):
        output = self.Gene
        if guided_gene.x + 45 > 180:
            output.x = random.randint(guided_gene.x - 45, 180)
        elif guided_gene.x - 45 < 0:
            output.x = random.randint(0, guided_gene.x + 45)
        else:
            output.x = random.randint(guided_gene.x - 45, guided_gene.x + 45)

        if guided_gene.y + 45 > 180:
            output.y = random.randint(guided_gene.y - 45, 180) if guided_gene.y - 45 < 180 else random.randint(135,180)
        elif guided_gene.y - 45 < 0:
            output.y = random.randint(0, guided_gene.y + 45)
        else:
            output.y = random.randint(guided_gene.y - 45, guided_gene.y + 45)

        if guided_gene.radius + 10 > 40:
            output.radius = random.randint(guided_gene.radius - 10, 40)
        elif guided_gene.radius - 10 < 0:
            output.radius = random.randint(0, guided_gene.radius + 10)
        else:
            output.radius = random.randint(guided_gene.radius - 10, guided_gene.radius + 10)

        if guided_gene.R + 64 > 255:
            output.R = random.randint(guided_gene.R - 64, 255)
        elif guided_gene.R - 64 < 0:
            output.R = random.randint(0, guided_gene.R + 64)
        else:
            output.R = random.randint(guided_gene.R - 64, guided_gene.R + 64)

        if guided_gene.G + 64 > 255:
            output.G = random.randint(guided_gene.G - 64, 255)
        elif guided_gene.G - 64 < 0:
            output.G = random.randint(0, guided_gene.G + 64)
        else:
            output.G = random.randint(guided_gene.G - 64, guided_gene.G + 64)

        if guided_gene.B + 64 > 255:
            output.B = random.randint(guided_gene.B - 64, 255)
        elif guided_gene.B - 64 < 0:
            output.B = random.randint(0, guided_gene.B + 64)
        else:
            output.B = random.randint(guided_gene.B - 64, guided_gene.B + 64)

        if guided_gene.A + 0.25 > 1:
            output.A = random.uniform(guided_gene.A - 0.25, 1)
        elif guided_gene.A - 0.25 < 0:
            output.A = random.uniform(0, guided_gene.A + 0.25)
        else:
            output.A = random.uniform(guided_gene.A - 0.25, guided_gene.A + 0.25)

        return output

    def create_genes(self, num_genes):
        genes = {}
        output = []
        # Create Gene according to parameter num_genes
        while num_genes != 0:
            d = self.Gene()
            # Create a dictionary, add the created gene to dict with its radius
            inserted_element = {d: d.radius}
            genes.update(inserted_element)
            num_genes -= 1
        # Sort the genes according to their radius. (This sorting is used for the first creation of an individual)
        a = sorted(genes.items(), key=lambda x: x[1], reverse=True)
        # Add the genes an individual in the sorted order
        for a_tuple in a:
            output.append(a_tuple[0])

        return output

    # This function is used for ordering genes after creating an individual i.e mutation,crossover
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
        # Order the genes according to radius
        self.order_my_genes()
        # Initialize <image> completely white with the same shape as the <source_image>.
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
        # Create two child with the equal number of gene with parents
        child1 = Individual(num_gene)
        child2 = Individual(num_gene)
        counter = 0
        while counter != num_gene:
            # Exchange of each gene is calculated individually with equal probability
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
            new_gene = self.guided_mutate(self.Genes[selected_gene])
            self.Genes[selected_gene] = new_gene


def initialize_population(num_inds=20, num_genes=50):
    individuals = []
    while num_inds != 0:
        individuals.append(Individual(num_genes))
        num_inds -= 1

    return individuals


def calculate_fitness(individual, source_image):
    # Initialize <image> completely white with the same shape as the <source_image>
    # In order to avoid the overflows the calculation is done with np.int64
    image_of_individual = np.int64(individual.draw_image(source_image))
    source_image = np.int64(source_image)
    f = 0
    # For each RGB channel calculate the pixel difference between two images
    for i in range(0, 3):
        individual_single_channel = image_of_individual[:, :, i].flatten()
        source_single_channel = source_image[:, :, i].flatten()
        f += -1 * sum((source_single_channel - individual_single_channel) ** 2)
        i += 1
    return f


def selection(Fitness_function, individuals, frac_elites=0.2, tm_size=5):
    # Calculate the number of individuals which go to next generation directly
    number_of_next_generation = np.ceil(len(individuals) * frac_elites)
    output_individuals = []
    output_fitness = []
    while number_of_next_generation != 0:
        # Find the best in the population
        index = [i for i, x in enumerate(Fitness_function) if x == max(Fitness_function)]
        index = int(index[0])
        # Add the best individual for the next generation with their fitness values
        output_individuals.append(individuals[index])
        output_fitness.append(Fitness_function[index])
        # Delete the added individuals from the population with their fitness values
        del individuals[index]
        Fitness_function = np.delete(Fitness_function, [index, index])
        number_of_next_generation -= 1

    return output_fitness, output_individuals, Fitness_function, individuals


def tournament(individuals, Fitness_function, tm_size=5):
    output_individuals = []
    output_fitness = []
    counter = 0
    # Until the number of output is equal to the number of individuals, continue to tournament
    while counter != len(individuals):
        # Choose the individuals according the tm_size.
        # If the number of individuals are less than tm_size, add all individuals to tournament.
        if len(individuals) <= 5:
            participants = [0, 1, 2, 3]
        elif len(individuals) <= tm_size:
            participants = np.arange(0, len(individuals), 1)
        # Otherwise take the random samples from individuals according to tm_size
        else:
            participants = random.sample(range(0, len(individuals)), tm_size)

        tournament_fitness = list(Fitness_function[participants])
        tournament_individuals = itemgetter(*participants)(individuals)
        # Choose the best of the tournament
        best_of_tournament = np.argmax(tournament_fitness)
        # Add the champion to the output of the tournament
        output_individuals.append(tournament_individuals[best_of_tournament])
        output_fitness.append(tournament_fitness[best_of_tournament])
        counter += 1

    return output_fitness, output_individuals


def crossover(fitness, population, frac_parents=0.6):
    # Decide the number of parents which involve the crossover
    number_of_parents = np.floor(len(population) * frac_parents)
    number_of_parents = number_of_parents if number_of_parents % 2 == 0 else number_of_parents - 1
    childs = []
    while number_of_parents != 0:
        # Amongst the parents choose the best two of them for crossover
        best_of_parent_index = np.argmax(fitness)
        parent1 = population[best_of_parent_index]
        # Since the chosen parents go to crossover, delete them from the population with their fitness values
        del population[best_of_parent_index]
        fitness = np.delete(fitness, [best_of_parent_index, best_of_parent_index])

        best_of_parent_index = np.argmax(fitness)
        parent2 = population[best_of_parent_index]
        del population[best_of_parent_index]
        fitness = np.delete(fitness, [best_of_parent_index, best_of_parent_index])

        # Create two children from two parents and add them to childs
        child1, child2 = parent1.do_crossover(parent2)
        childs.append(child1)
        childs.append(child2)
        number_of_parents -= 2

    return childs, population


def mutation(individuals, mutation_prob, mutation_type):
    # Mutate the all individuals except the elites
    number_individuals = len(individuals)
    counter = 0
    output = []
    while counter != number_individuals:
        processed_individual = individuals[counter]
        #  While the generated random number is smaller than
        # <mutation prob> a random gene is selected to be mutated
        if random.random() < mutation_prob:
            processed_individual.do_mutation(mutation_type)

        output.append(processed_individual)
        counter += 1
    return output

# Load image
source_image = cv2.imread("painting.png")

# Initialize population with <num_inds> individuals each having <num_genes> genes
population = initialize_population(num_inds=20, num_genes=50)

num_generations = 10001
i = 0
fitness_plot = np.zeros(11)
while i != num_generations:
    Fitness_of_individuals = np.zeros(len(population))
    counter = 0
    # Evaluate all the individuals
    for x in population:
        Fitness_of_individuals[counter] = calculate_fitness(x, source_image)
        counter += 1

    # Record the best individual and the fitness for every 1000 epoch
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
    childs, old_population = crossover(other_fitness, other_population, frac_parents=0.8)
    # Mutation ( For the sake of easiness, I've used "1" for guided mutation and "0" for the unguided mutation)
    mutation_population = childs + old_population
    mutated_population = mutation(mutation_population, mutation_prob=0.2, mutation_type=1)
    # New Population
    population = best_population + mutated_population
    i += 1

import matplotlib.pyplot as plt

plt.ylabel("Fitness score")
plt.xlabel("Steps")
plt.plot(fitness_plot)
