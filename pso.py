import random
import math
import matplotlib.pyplot as plt

# Armazenar a média e o melhor global a cada iteração
mean_fitness = []
best_fitness = []

class Particle:
    def __init__(self, dimensions, lower_bound, upper_bound):
        self.position = [round(random.uniform(lower_bound, upper_bound), 3) for _ in range(dimensions)]
        self.velocity = [round(random.uniform(0, 10), 3)] * dimensions
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('inf')

    def evaluate_griewank(self):
        top1 = sum(x ** 2 for x in self.position)
        top2 = math.prod(math.cos(p / math.sqrt(i + 1)) for i, p in enumerate(self.position))
        top = (1 / 4000) * top1 - top2 + 1

        self.personal_best_fitness = round(top, 3)

# Parâmetros do PSO
num_particles = 50
num_dimensions = 10
max_iterations = 1000

# Limites para as variáveis
lower_bound = -100.0
upper_bound = 100.0

# Inicializar as partículas
particles = [Particle(num_dimensions, lower_bound, upper_bound) for _ in range(num_particles)]
global_best_position = [0.0] * num_dimensions
global_best_fitness = float('inf')

w = 0.7  # Fator de inércia
c1 = 2.0  # Coeficiente de aprendizagem para a melhor posição pessoal
c2 = 2.0  # Coeficiente de aprendizagem para a melhor posição global

# Iterações do PSO
for _ in range(max_iterations):
    # Avaliação das Partículas
    for particle in particles:
        particle.evaluate_griewank()
        fitness = particle.personal_best_fitness

        # Atualiza o personal best (Partícula)
        if fitness < particle.personal_best_fitness:
            particle.personal_best_position = particle.position.copy()
            particle.personal_best_fitness = fitness

        # Atualiza o global best (Enxame)
        if fitness < global_best_fitness:
            global_best_fitness = fitness
            global_best_position = particle.position.copy()

        rand1 = random.uniform(0, 1)
        rand2 = random.uniform(0, 1)

        for i in range(num_dimensions):
            # Atualiza a velocidade da partícula
            new_velocity = (w * particle.velocity[i]) + (c1 * rand1 * (particle.personal_best_position[i] - particle.position[i])) + (c2 * rand2 * (global_best_position[i] - particle.position[i]))
            particle.velocity[i] = round(new_velocity, 3)

            # Atualiza a posição da partícula
            new_position = particle.position[i] + particle.velocity[i]
            particle.position[i] = round(new_position, 3)

    print(global_best_fitness)
    print(sum(particle.personal_best_fitness for particle in particles) / num_particles)
    # Armazenar a média e o melhor global
    mean_fitness.append(sum(particle.personal_best_fitness for particle in particles) / num_particles)
    best_fitness.append(global_best_fitness)

# Plotar o gráfico
plt.plot(range(max_iterations), mean_fitness, label='Média do Enxame')
plt.plot(range(max_iterations), best_fitness, label='Melhor Global')
plt.xlabel('Iterações')
plt.ylabel('Função Objetivo')
plt.title('Convergência do PSO')
plt.legend()
plt.show()