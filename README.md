# GeneticAlgo_MovingAverageStrat

## Genetic Algorithm Overview

Genetic Algorithms (GAs) are optimization techniques inspired by the principles of natural selection and genetics. They are used to find approximate solutions to complex problems by mimicking the process of natural evolution. The basic steps and concepts of a Genetic Algorithm are as follows:

### Steps

1. **Initialize Population**: Generate an initial population consisting of multiple individuals (possible solutions). Each individual is typically represented as a string or array.

2. **Evaluate Fitness**: Assess the fitness of each individual in the population. Fitness is a measure of how well an individual solves the problem or meets the desired criteria.

3. **Selection**: Select individuals for reproduction based on their fitness. Individuals with higher fitness have a higher probability of being chosen. Common selection methods include roulette wheel selection, tournament selection, and rank selection.

4. **Crossover (Recombination)**: Create new individuals (offspring) by combining parts of two parent individuals. This process mimics genetic recombination in biological reproduction. Common crossover techniques include single-point crossover, multi-point crossover, and uniform crossover.

5. **Mutation**: Introduce random changes to individuals to maintain genetic diversity within the population and prevent premature convergence to local optima. Mutation rates are typically low to ensure small, incremental changes.

6. **Replacement**: Form a new population by replacing some or all of the old population with the new individuals. This step ensures the evolution of the population over successive generations.

7. **Termination**: Repeat the evaluation, selection, crossover, and mutation steps for several generations until a stopping criterion is met. Common stopping criteria include reaching a maximum number of generations, achieving a satisfactory fitness level, or observing no significant improvement over several generations.

### Concepts

- **Population**: A collection of individuals representing possible solutions to the problem.
- **Individual (Chromosome)**: A single solution encoded as a string or array.
- **Gene**: A part of an individual's encoding, representing a particular parameter or attribute.
- **Fitness Function**: A function that quantifies the quality or performance of an individual.
- **Selection Pressure**: The degree to which fitter individuals are favored in the selection process.
- **Genetic Diversity**: The variety of different solutions in the population, crucial for avoiding premature convergence.

By iteratively applying these steps, the Genetic Algorithm evolves the population towards optimal or near-optimal solutions.


## About this project

### Project Structure

- `tr_eikon_eod_data.csv`: data
- `SMA.ipynb` : Implemenet the SMA Crossover Strategy
- `main.py` : Optimize the short-term and long-term window for SMA Crossover Strategy by Genetic Algorithm
  
### How to run this code
- `python3 main.py`

### Assumption
- Short-term & Long-term Window  
For short term and long term SMA, 50-day and 200-day are generally adopted.  
In this project, each chromosome contains 14-bit binary number. The first 6-bit is used for short-term SMA. The maximum value of a 6-bit unsigned binary number is `2^6 -1 = 63`, which is slightly higher than 50.
Similarly, `2^(14-6) -1 = 2^(8) - 1 =  255`, so 8-bit is assigned for long term sma.  
For example, for a chromosome `[1,0,0,0,0,1,0,0,0,1,1,0,1,1]`.  
The first 6-bit is `[1,0,0,0,0,1]`, representing `32+1+(1) = 34`  
The last 8-bit is `[0,0,0,1,1,0,1,1]`, representing `16+8+2+1+(1) = 28`  
*** Every integer represented by a  will be incremented by 1, to avoid zero value




- Fitness Score
  For meauring the fitness level of a chromosome, the culmulative return of the two window size that the chromosome contains will be used.

### Result
<img width="1028" alt="Output_1" src="https://github.com/user-attachments/assets/7fbc5f3f-e616-4040-902c-2198d7df23d9">
<img width="1028" alt="Output_2" src="https://github.com/user-attachments/assets/cd0d95fa-dda8-4ef5-8c87-a1ef3bb4ae1a">

![Sample](https://github.com/user-attachments/assets/c13566db-802a-4f92-ab8f-026761473c71)


