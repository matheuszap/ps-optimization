import random
import math

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
num_particles = 2
num_dimensions = 10
max_iterations = 1000

# Limites para as variáveis
lower_bound = -100.0
upper_bound = 100.0

# Inicializar as partículas
particles = [Particle(num_dimensions, lower_bound, upper_bound) for _ in range(num_particles)]
global_best_position = [0.0] * num_dimensions
global_best_fitness = float('inf')

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

    # Atualizar as posições das partículas
    for particle in particles:
        for i in range(num_dimensions):
            # Atualizar posição
            position = particle.position[i]
            velocity = particle.velocity[i]
            new_position = position + velocity
            particle.position[i] = round(new_position, 3)

# Informações das partículas
for i, particle in enumerate(particles):
    print("Partícula", i+1)
    print("Posição atual:", particle.position)
    print("Velocidade:", particle.velocity)
    print("Melhor posição:", particle.personal_best_position)
    print("Melhor fitness:", particle.personal_best_fitness)
    print("---------------------------------------")

# Informação do Enxame
print("Melhor posição do enxame:", global_best_position)
print("Melhor fitness do enxame:", global_best_fitness)
