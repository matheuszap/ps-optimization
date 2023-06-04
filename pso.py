import random

class Particle:
    def __init__(self, dimensions, lower_bound, upper_bound):
        self.position = [random.uniform(lower_bound, upper_bound) for _ in range(dimensions)]
        self.velocity = [0.0] * dimensions
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = float('inf')

# Definir os parâmetros do PSO
num_particles = 30
num_dimensions = 10
max_iterations = 100
c1 = 2.0  # Coeficiente cognitivo
c2 = 2.0  # Coeficiente social
w = 0.7  # Inércia

# Definir limites para as variáveis
lower_bound = -100.0
upper_bound = 100.0

# Inicializar as partículas
particles = [Particle(num_dimensions, lower_bound, upper_bound) for _ in range(num_particles)]
global_best_position = [0.0] * num_dimensions
global_best_fitness = float('inf')

# Imprimir informações das partículas
for i, particle in enumerate(particles):
    print("Partícula", i+1)
    print("Posição atual:", particle.position)
    print("Velocidade:", particle.velocity)
    print("Melhor posição pessoal:", particle.personal_best_position)
    print("Melhor fitness pessoal:", particle.personal_best_fitness)
    print("---------------------------------------")