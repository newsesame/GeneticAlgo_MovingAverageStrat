import numpy as np 
import pandas as pd 
from tabulate import tabulate
from pylab import  plt
import warnings
warnings.filterwarnings('ignore')
class GeneticAlgo():
    

    def __init__(self, population_size: int, chromosome_size: int, mutation_rate: float, crossover_rate: float,generations:int, prices: np.ndarray ):
        self.population_size = population_size  # No. of chromosome in a generation
        self.chromosome_size = chromosome_size  # No. bits for representing the moving averages of both the shorter and longer window
        self.mutation_rate = mutation_rate      # How likely a bit in a chromosome would be mutated
        self.crossover_rate  = crossover_rate   # How likely we would crossover two parents to generate a pair of two 
        self.generations = generations          # No. of iteration to generate new population
        self.population = self.random_population() # The population at hands
        
        self.data = pd.DataFrame(prices, columns= ["Close"]) # Import the closing price 
        
        '''
        For short term and long term SMA, 50-day and 200-day are generally adopted. 
        The reason why 6-bit is used for short term SMA is 
        that maximum value of a 6-bit unsigned binary number is 2^6 -1 = 63, which is slightly higher than 50.
        Similarly, 2^(14-6) -1 = 2^(8) - 1 =  255, so 8-bit is assigned for long term sma.
        '''
        self.sma_bits = 6
        self.lma_bits = self.chromosome_size-self.sma_bits   # 8
        
        
        self.best_chromosome = [0,None]     # To store the best chromosome.
        

    '''
    Function Description: chromo_to_int

	Purpose:
	•	This function converts a list of int into integer values representing the short-term SMA and the long-term 
        parameters for a trading strategy.
	Parameters:
	•	chromo: An array of int representing the chromosome.
	Returns:
	•	A tuple (sma, lma):
	•	sma: An integer representing the short-term moving average, calculated from the first part of the chromosome.
	•	lma: An integer representing the long-term moving average, calculated from the second part of the chromosome.
    '''
    def chrom_to_int(self, chromo: list[int]) -> int:
        # Convert the binary number of base-2 to int and increment by 1 to make sure no zero values.
        sma = int(''.join(map(str, chromo[:self.sma_bits])), 2)+1  
        lma = int(''.join(map(str, chromo[self.sma_bits:])), 2)+1
        return (sma, lma)


    '''
    Function Description: random_population

	Purpose:
	•	This function generates a random initial population for the genetic algorithm.
	Returns:
	•	A list of lists of integers:
	•	Each sublist represents a chromosome in the population.
    '''
    def random_population(self) -> list[list[int]]:
        population = np.random.randint(0,2, size = (self.population_size, self.chromosome_size ))
        return population.tolist()
    

    '''
    Function Description: fitness_level

	Purpose:
	•	This function evaluates the fitness level of a given chromosome by calculating the cumulative returns of a simple moving average (SMA) crossover trading strategy
        , with the corresponding short term and long term SMA.
	Parameters:
	•	chromosome: A list of integers representing a binary chromosome.
    Returns:
	•	A float representing the cumulative strategy returns.
    '''
    def fitness_level(self, chromosome:list[int])-> float:
        # Convert the chromosome to two integers
        sma = int(''.join(map(str, chromosome[:self.sma_bits])), 2) + 1  # 1 <= length of window for shorter moving average <= 2^(n/2) - 1 
        lma = int(''.join(map(str, chromosome[self.sma_bits:])), 2)  +1 # sma < length of window for longer moving average <= sma + (2^(n/2) - 1)

        # Case to eliminate
        if lma <= sma or lma == 1 or sma == 1  or lma < 70 or sma <10 :
            return 0
        
        self.data["SMA"] = self.data["Close"].rolling(window= sma).mean()
        self.data["LMA"] = self.data["Close"].rolling(window= lma).mean()

        # Define the signal and position
        '''
        • Go long (= +1) when the shorter SMA is above the longer SMA.
        • Go short (= -1) when the shorter SMA is below the longer SMA.
        For a long only strategy one would use +1 for a long position and 0 for a neutral position.
        '''
        self.data["Signal"] = 0
        self.data["Signal"] = np.where(self.data['SMA'] > self.data['LMA'], 1, 0) # Indicate whether the shorter SMA is above the longer SMA
        self.data['Position'] = self.data['Signal'].diff()                        # Indicate the buy and sell position

        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['StrategyReturns'] = self.data['Returns'] * self.data['Position'].shift(1)    # Record the profit of our buy and sell position
        self.data['CumulativeStrategyReturns'] = (1 + self.data['StrategyReturns']).cumprod()   # Calculate the cumulative return
    
        return self.data['CumulativeStrategyReturns'].iloc[-1]
    

    '''
    Function Description: selection

	Purpose:
	•	This function selects chromosomes from the current population based on their fitness levels using a probabilistic approach,
        preparing them for the next generation in the genetic algorithm.
	Returns:
	•	A list of lists of integers:
	•	Each sublist represents a selected chromosome from the current population.
    '''
    def selection(self) -> list[list[int]]:
        # Evaluates the fitness level of each chromosome in the current population.
        fitness_level = np.array( [self.fitness_level(chromosome=chrom) for chrom in self.population]) 

        # Normalizes the fitness levels to create selection probabilities for each chromosome.
        probabilities = fitness_level / fitness_level.sum() 

        # Uses these probabilities to randomly select chromosomes, allowing fitter chromosomes a higher chance of being selected.
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, p=probabilities)
        selected_indices = selected_indices.tolist() 

        #Returns the selected chromosomes for the next generation.
        return [self.population[i] for i in selected_indices]

        
    '''
    Function Description: crossover

	Purpose:
	•	This function performs crossover between two parent chromosomes to produce offspring chromosomes, 
        potentially introducing new genetic material into the population.
	Parameters:
	•	parent1: A list of integers representing the first parent chromosome.
	•	parent2: A list of integers representing the second parent chromosome.
	Returns:
	•	A tuple of two lists of integers:
	•	Each list represents an offspring chromosome resulting from the crossover.
    '''
    def crossover(self, parent1: list[int], parent2:list[int]):
        # Crossover two parent chromosome to create offspring chromosomes

        # If a random int is larger than crossover_rate, then we do crossover. Otherwsie, we keep the two prarents.
        if np.random.rand() < self.crossover_rate:
            
            # we only consider doing crossover at the no. of bits of shorter sma, as a chrom represents two, sma and lma. 
            # Otherwise, a crossover at any point might be meeaningless, as it cannot pass on the genetic material of either the sma or lma.
            crossover_point = self.sma_bits
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            return child1, child2
        return parent1, parent2


    '''
    Function Description: mutate

	Purpose:
	•	This function performs mutation on a chromosome to increase genetic diversity in the population.
	Parameters:
	•	chromosome: A list of integers representing the chromosome to be mutated.
	Returns:
	•	A list of integers representing the mutated chromosome.
    '''
    def mutate(self, chromosome: list[int])->list[int]:
        # Perform mutation on a chromosome to increase the diversity

        for i in range(len(chromosome)):
            # With a probability defined by self.mutation_rate, each gene is mutated:
            if np.random.rand() < self.mutation_rate:

                 # If the gene is 1, it is changed to 0. If the gene is 0, it is changed to 1.
                chromosome[i] = 1 - chromosome[i]  
        return chromosome


    '''
    Function Description: evolve

	Purpose:
	•	This function runs the genetic alogrithm process. 
    •   It evolves the population of chromosomes over multiple generations to optimize the SMA crossover trading strategy.
	Parameters:
	•	None.
	Returns:
	•	None.
    '''
    def evolve(self)-> None :
        # Evolve the population over multiple generations
        for generation in range(self.generations):

            # Selects parent chromosomes based on their fitness levels. 
            selected_population = self.selection()
            next_generation = []

            for i in range(0, self.population_size, 2):
                parent1, parent2 = selected_population[i], selected_population[i+1]

                # Performs crossover on pairs of parent chromosomes to create offspring. 
                child1, child2 = self.crossover(parent1, parent2)

                # Mutates the offspring to introduce genetic diversity.
                next_generation.extend([self.mutate(child1), self.mutate(child2)])

            # Updates the population with the new generation of chromosomes.
            self.population = next_generation

            # Evaluates the fitness of each chromosome in the new population.
            # Also, Keeps track of the best fitness value across all the generation and the corresponding chromosome.
            best_fitness = 0
            best_chrom = None
            for chrom in self.population:
                score =  self.fitness_level(chrom)

                # Local Best Chrom
                if score > best_fitness:
                    best_fitness = score
                    best_chrom = chrom
                    
            # Best Chrom across all generations 
            if best_fitness > self.best_chromosome[0]:
                self.best_chromosome[0] = best_fitness
                self.best_chromosome[1] = best_chrom

            sma, lma  = self.chrom_to_int(best_chrom)
            print(f'Generation {generation+1} | Best Fitness: {best_fitness:.4f} | S/L Window: {sma, lma}' )
        
        print("This is the end of the genetic algorithm optimization.\n")
        


    '''
    Function Description: position

	Purpose:
	•	This function shows the buy and sell positions based on a given chromosome and displays the relevant trading signals.
	Parameters:
	•	chromo: A list of integers representing the chromosome.
	Returns:
	•	A DataFrame containing the buy and sell positions.
    '''    
    def position(self, chromo):
        print(self.data)

        # Refresh the columns in self.data by the target chromosome
        _ = self.fitness_level(chromo)  

        # Filters the data to include only rows where a position change occurs (either buy or sell).
        self.data['StrategyReturns'] = self.data['StrategyReturns'].shift(-1)
        self.data['CumulativeStrategyReturns'] = self.data['CumulativeStrategyReturns'].shift(-1)
        df_new = self.data[(self.data['Position'] == 1) | (self.data['Position'] == -1)]

        # Transforms the position indicator to a human-readable format (Buy for 1, Sell for -1).
        df_new['Position'] = df_new['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')

        
        # Prints the filtered DataFrame in a tabulated format.
        print(tabulate(df_new, headers = 'keys', tablefmt = 'psql'))
        return df_new
    

    '''
    Function Description: plot_chromo

	Purpose:
	•	This function visualizes the trading signals and moving averages for a given chromosome, 
        allowing for graphical analysis of the SMA crossover strategy.
	Parameters:
	•	chromo: A list of integers representing the chromosome.
    '''
    def plot_chromo(self,chromo):
        # Visualising the signals

        _ = self.fitness_level(chromo)  # Refresh the columns in self.data by the target chromosome

        sma, lma = self.chrom_to_int(chromo)

        plt.figure(figsize=(12, 8))
        plt.plot(self.data['Close'], label=symbol+ ' Close Price')
        plt.plot(self.data['SMA'], label= str(sma)+'-day SMA Short Term')
        plt.plot(self.data['LMA'], label=str(lma) + '-day SMA Long Term')
        plt.plot(self.data[self.data['Position'] == 1].index, self.data['SMA'][self.data['Position'] == 1], '^', markersize=10, color='g', label='Buy signal')
        plt.plot(self.data[self.data['Position'] == -1].index, self.data['SMA'][self.data['Position'] == -1], 'v', markersize=10, color='r', label='Sell signal')
        plt.title(symbol+' Simple Moving Average Crossover Strategy')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

raw = raw = pd.read_csv('./tr_eikon_eod_data.csv',
                              index_col=0, parse_dates=True)
symbol = 'AAPL.O'
data = pd.DataFrame(raw[symbol].dropna())
data.rename(columns={symbol:'Close'}, inplace=True)


ga = GeneticAlgo(population_size=2500, chromosome_size=14, mutation_rate=0.4, crossover_rate=0.8, generations=20, prices=data)
ga.evolve()
print("### The Chrom with the highest fitness score ###")
sma, lma = ga.chrom_to_int(ga.best_chromosome[1])
print("Chrom: ", str(ga.best_chromosome[1]))
print("Short-term Window: " + str(sma)+ " | " +" Long-term Window: " + str(lma) )
print("Fitness Score: ", ga.fitness_level(ga.best_chromosome[1]))
print()
print("### Table of positions ###")
ga.position(ga.best_chromosome[1])
ga.plot_chromo(ga.best_chromosome[1])

