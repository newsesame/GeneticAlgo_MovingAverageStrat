import numpy as np 
import pandas as pd 

class GeneticAlgo():
    

    def __init__(self, population_size: int, chromosome_size: int, mutation_rate: float, crossover_rate: float,generations:int, prices: np.ndarray ):
        self.population_size = population_size
        self.chromosome_size = chromosome_size
        self.mutation_rate = mutation_rate
        self.crossover_rate  = crossover_rate
        self.generations = generations
        self.population = self.random_population()
        self.data = pd.DataFrame(prices, columns= ["Close"])
        
        


    def random_population(self) -> list[list[int]]:
        population = np.random.randint(0,2, size = (self.population_size, self.chromosome_size ))
        return population.tolist()
    
    def fitness_level(self, chromosome:list[int])-> float:
        # Convert the chromosome to two integers
        sma = int(''.join(map(str, chromosome[:6])), 2) + 1  # 1 <= length of window for shorter moving average <= 2^(n/2) - 1 
        lma = int(''.join(map(str, chromosome[6:])), 2) + sma+1 # sma < length of window for longer moving average <= sma + (2^(n/2) - 1)

        self.data["SMA"] = self.data["Close"].rolling(window= sma).mean()
        self.data["LMA"] = self.data["Close"].rolling(window= lma).mean()

        # Define the position
        '''
        • Go long (= +1) when the shorter SMA is above the longer SMA.
        • Go short (= -1) when the shorter SMA is below the longer SMA.
        For a long only strategy one would use +1 for a long position and 0 for a neutral position.
        '''
        self.data["Position"] = np.where(self.data['SMA'] > self.data['LMA'], 1, -1)

        self.data['Change'] = np.log(self.data["Close"] / self.data["Close"].shift(1))
        self.data['Return'] = self.data['Position'].shift(1)*self.data['Change']
        result  = self.data['Return'].sum()
        return result
    
    
    def selection(self) -> list[list[int]]:

        
        fitness_level = np.array( [self.fitness_level(chromosome=chrom) for chrom in self.population])
        probabilities = fitness_level / fitness_level.sum()
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        selected_indices = selected_indices.tolist() 
        return [self.population[i] for i in selected_indices]

        

    def crossover(self, parent1: list[int], parent2:list[int]):
        # Crossover two parent chromosome to create offspring chromosomes

        # If a random int is larger than crossover_rate, then we do crossover. Otherwsie, we keep the two prarents
        if np.random.rand() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.chromosome_size-1)
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1, parent2

    def mutate(self, chromosome: list[int])->list[int]:
        # Perform mutation on a chromosome to increase the diversity

        for i in range(len(chromosome)):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i] # 1 -> 0; 0 -> 1
        return chromosome

        
    def evolve(self)-> None :
        # Evolve the population over multiple generations
        for generation in range(self.generations):

            # Select parent chromosomes 
            selected_population = self.selection()
            next_generation = []

            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                next_generation.extend([self.mutate(child1), self.mutate(child2)])

            
            self.population = next_generation
            best_fitness = max([self.fitness_level(chrom) for chrom in self.population])
            print(f'Generation {generation+1} | Best Fitness: {best_fitness:.4f}')

raw = raw = pd.read_csv('./tr_eikon_eod_data.csv',
                              index_col=0, parse_dates=True)
symbol = 'AAPL.O'
data = pd.DataFrame(raw[symbol].dropna())
data.rename(columns={'AAPL.O':'Close'}, inplace=True)

print(data)
ga = GeneticAlgo(population_size=100, chromosome_size=12, mutation_rate=0.01, crossover_rate=0.8, generations=100, prices=data)
print(ga.data)
print(ga.fitness_level([1]*3+[0]*3+[1]*3+[0]*3))
# ga.evolve()
        
        

# # Create an instance of the GeneticAlgorithm class
# ga = GeneticAlgor(population_size=100, chromosome_length=10, mutation_rate=0.01, crossover_rate=0.8, generations=100)

# # Run the genetic algorithm to optimize the trading strategy
# ga.evolve()

        
