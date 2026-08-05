import numpy as np

class NK_model:
    def __init__(self, N, K):
        self.N = N
        self.K = K
        # one random table per gene, 2^(K+1) rows, fixed for this landscape
        self.fitness_table = np.random.rand(self.N, 2**(self.K + 1))

    def generate_genome(self):
        self.genome = np.random.randint(2, size=self.N)   # {0,1}: used to index tables
        self.inputs = self.genome * 2 - 1                  # {-1,+1}: fed to the MLP
        return self.inputs

    def calculate_fitness(self):
        powers = 2 ** np.arange(self.K, -1, -1)

        total = 0.0
        for i in range(self.N):
            neighbourhood = [(i + j) % self.N for j in range(self.K + 1)]  # gene i + K partners
            bits = self.genome[neighbourhood]        # e.g. [own, partner1, ...]
            index = bits @ powers                    # binary pattern -> row number
            total += self.fitness_table[i, index]
        return total / self.N