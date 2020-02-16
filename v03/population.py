'''

'''

# packages


# scripts
from genetic_material import IndividualMaterial


class PopulationDefinition():
    '''
    all other problem classes must be called as
     class Problem(ProblemDefinition)
    '''
    def __init__(self,
                population_size,
                individual_definition):
        self.pop_size = population_size
        self.population = []
        for _ in range(population_size):
            self.population.append(IndividualMaterial(individual_definition))