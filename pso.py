import random
import math
import matplotlib.pyplot as plt
import numpy as np

def evaluate_griewank(position):
    d = len(position)
    sum = 0.0
    prod = 1.0

    for i in range(d):
        sum += position[i] ** 2.0
        prod *= np.cos(position[i] / np.sqrt(i + 1.0))

    top = (sum / 4000.0) - prod + 1.0

    return top

def evaluate_ackley(position):
    d = len(position)
    aux1 = 0.0
    aux2 = 0.0

    for i in range(d):
        aux1 += position[i] * position[i]
        aux2 += math.cos(2.0 * math.pi * position[i])

    result = (
        -20.0 * math.exp(-0.2 * math.sqrt(1.0 / d * aux1))
        - math.exp(1.0 / d * aux2)
        + 20.0
        + math.exp(1)
    )

    return result

class Particle:
    def __init__(self, dimensions, lower_bound, upper_bound):
        self.position = [random.uniform(lower_bound, upper_bound)  for _ in range(dimensions)]
        self.velocity = [random.uniform(-5, 5)] * dimensions
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('inf')

    def evaluate_fitness(self):
        fitness = evaluate_griewank(self.position)
        #fitness = evaluate_ackley(self.position)

        # Atualiza o personal best (Partícula)
        if fitness < self.personal_best_fitness:
            self.personal_best_position = self.position.copy()
            self.personal_best_fitness = fitness

        return fitness

# Parâmetros do PSO
num_particles = 50
num_dimensions = 20
max_iterations = 1000

# Limites para as variáveis
#lower_bound = -32.768
#upper_bound = 32.768

lower_bound = -600.0
upper_bound = 600.0

# Inicializar as partículas
particles = [Particle(num_dimensions, lower_bound, upper_bound) for _ in range(num_particles)]
global_best_position = [0.0] * num_dimensions
global_best_fitness = float('inf')

w = 0.3  # Fator de inércia
c1 = 2.05  # Coeficiente de aprendizagem para a melhor posição pessoal
c2 = 2.05  # Coeficiente de aprendizagem para a melhor posição global

k = 2 / (np.abs(2 - (c1 + c2) - np.sqrt((c1 + c2)**2 - 4 * (c1 + c2))))

# Armazenar a média e o melhor global a cada iteração
mean_fitness = []
best_fitness = []

# Iterações do PSO
for _ in range(max_iterations):
    # Avaliação das Partículas
    for particle in particles:
        fitness = particle.evaluate_fitness()

        # Atualiza o global best (Enxame)
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle.position.copy()

        for i in range(num_dimensions):
            rand1 = np.random.rand()
            rand2 = np.random.rand()
            
            # Atualiza a velocidade da partícula
            new_velocity = (w * particle.velocity[i]) + (c1 * rand1 * (particle.personal_best_position[i] - particle.position[i])) + (c2 * rand2 * (global_best_position[i] - particle.position[i]))     
            particle.velocity[i] = new_velocity

            # Atualiza a posição da partícula
            new_position = particle.position[i] + particle.velocity[i]

            # Verifica se a nova posição está fora dos limites
            if new_position < lower_bound:
                new_position = lower_bound
            elif new_position > upper_bound:
                new_position = upper_bound
            particle.position[i] = new_position

    #print(global_best_fitness)
    #print(sum(particle.personal_best_fitness for particle in particles) / num_particles)
    # Armazenar a média e o melhor global
    mean_fitness.append(sum(particle.personal_best_fitness for particle in particles) / num_particles)
    best_fitness.append(global_best_fitness)

print(global_best_fitness)
print(global_best_position)

# Plotar o gráfico
plt.plot(range(max_iterations), mean_fitness, label='Média do Enxame', color="red")
plt.plot(range(max_iterations), best_fitness, label='Melhor Global', color="green")
plt.xlabel('Iterações')
plt.ylabel('Função Objetivo')
plt.title('Convergência do PSO')
plt.legend()
plt.show()
