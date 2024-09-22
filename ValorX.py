import random
import numpy as np

# Função alvo f(x)
def f(x):
    return x**3 - 6*x + 14

# Função para decodificar o valor de x a partir do vetor binário
def decode_chromosome(chromosome, bounds, bits):
    # Converte o binário para decimal
    normalized_value = sum([bit * (2 ** i) for i, bit in enumerate(reversed(chromosome))])
    min_bound, max_bound = bounds
    precision = (max_bound - min_bound) / (2**bits - 1)
    decoded_value = min_bound + normalized_value * precision
    return decoded_value

# Função de fitness (menor valor de f(x))
def fitness(chromosome, bounds, bits):
    x = decode_chromosome(chromosome, bounds, bits)
    return -f(x)  # Negativo pois estamos minimizando

# Seleção por torneio
def tournament_selection(population, fitness_values, tournament_size=3):
    selected = random.sample(range(len(population)), tournament_size)
    best = min(selected, key=lambda ind: fitness_values[ind])
    return population[best]

# Crossover de um ponto
def one_point_crossover(parent1, parent2):
    cut = random.randint(1, len(parent1) - 1)
    return parent1[:cut] + parent2[cut:], parent2[:cut] + parent1[cut:]

# Crossover de dois pontos
def two_point_crossover(parent1, parent2):
    cut1, cut2 = sorted(random.sample(range(1, len(parent1)), 2))
    return parent1[:cut1] + parent2[cut1:cut2] + parent1[cut2:], parent2[:cut1] + parent1[cut1:cut2] + parent2[cut2:]

# Mutação (flip de bits)
def mutate(chromosome, mutation_rate):
    return [bit if random.random() > mutation_rate else 1 - bit for bit in chromosome]

# Elitismo: mantém uma proporção dos melhores indivíduos
def elitism(population, fitness_values, elitism_rate):
    elite_size = int(elitism_rate * len(population))
    elite_indices = sorted(range(len(fitness_values)), key=lambda ind: fitness_values[ind], reverse=True)[:elite_size]
    elite_individuals = [population[i] for i in elite_indices]
    return elite_individuals

# Algoritmo Genético
def genetic_algorithm(bounds, bits=16, population_size=10, generations=100, mutation_rate=0.01, crossover='one-point', elitism_enabled=True, elitism_rate=0.1, selection_method='tournament'):
    # Inicialização da população (vetores binários aleatórios)
    population = [[random.randint(0, 1) for _ in range(bits)] for _ in range(population_size)]
    
    for generation in range(generations):
        # Avaliação da população (fitness)
        fitness_values = [fitness(chromosome, bounds, bits) for chromosome in population]
        
        # Elitismo
        new_population = []
        if elitism_enabled:
            elite = elitism(population, fitness_values, elitism_rate)
            new_population.extend(elite)
        
        # Seleção, Crossover e Mutação
        while len(new_population) < population_size:
            if selection_method == 'tournament':
                parent1 = tournament_selection(population, fitness_values)
                parent2 = tournament_selection(population, fitness_values)
            
            # Crossover
            if crossover == 'one-point':
                offspring1, offspring2 = one_point_crossover(parent1, parent2)
            elif crossover == 'two-point':
                offspring1, offspring2 = two_point_crossover(parent1, parent2)
            
            # Mutação
            offspring1 = mutate(offspring1, mutation_rate)
            offspring2 = mutate(offspring2, mutation_rate)
            
            new_population.append(offspring1)
            if len(new_population) < population_size:
                new_population.append(offspring2)
        
        population = new_population
    
    # Após o término das gerações, seleciona o melhor indivíduo
    fitness_values = [fitness(chromosome, bounds, bits) for chromosome in population]
    best_individual = population[np.argmax(fitness_values)]
    best_x = decode_chromosome(best_individual, bounds, bits)
    return best_x, f(best_x)

# Parâmetros
bounds = (-10, 10)  # Limite de x [-10, 10]
bits = 16  # Precisão binária
population_size = 10  # Tamanho da população
generations = 100  # Número de gerações
mutation_rate = 0.01  # Taxa de mutação
crossover_type = 'two-point'  # Tipo de crossover
elitism_enabled = True  # Usar elitismo
elitism_rate = 0.1  # Proporção de indivíduos elite
selection_method = 'tournament'  # Método de seleção

# Executando o algoritmo
best_x, best_fitness = genetic_algorithm(bounds, bits, population_size, generations, mutation_rate, crossover_type, elitism_enabled, elitism_rate, selection_method)
print(f"Melhor valor de x: {best_x}, com f(x): {best_fitness}")