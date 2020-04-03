'''

'''

# packages


# scripts
from genetic_material import IndividualMaterial


class PopulationDefinition():
    '''
    words
    '''
    def __init__(self,
                population_size):
        self.pop_size = population_size
        self.population = []

    
    def get_fitness(self):
        fitness = []
        for indiv in self.population:
            fitness.append(indiv.fitness.value)
        return fitness


    def add_next_generation(self, next_generation):
        '''
        not clear if we actually need this...here just incase if we find it better to handle adding individuals
        differently that just immediately adding to the rest of the population

        assuming next_generation is a list
        '''
        self.population += next_generation

    def split(self, size):
        new_pop = []
        for i in range(size):
            new_pop.append([])

        for i in range(len(self.population)):
            new_pop[i % size].append(self.population[i])
        self.population = new_pop

    def merge_pop(self):
        new_pop = []

        for sub_pop in self.population:
            for ind in sub_pop:
                new_pop.append(ind)
        self.population = new_pop


class SubPopulationDefinition(PopulationDefinition):
    '''
     * population size
     * number of subpopulations OR subpopulation size
     * population list of of subpopulation individuals
    '''
    def __init__(self,
                population_size,
                number_subpopulations,
                #subpopulation_size = None, # just seems unlikely we'll ever use subpopulation_size to split by
                ):
        self.pop_size = population_size
        self.num_subpops = number_subpopulations
        self.subpop_size = [] # fill in later...size of each subpopulation
        self.population = []

        ''' removing this code in favor of having distribute_population() method to do this
        # for each subpopulation, get it's size
        if number_subpopulations is not None:
            base_count = population_size//number_subpopulations #min size of each subpop
            self.subpop_size = [base_count]*number_subpopulations
        else:
            # then assume we go with subpopulation_size
            number_subpopulations = population_size//subpopulation_size
            self.subpop_size = [subpopulation_size]*number_subpopulations
        # now add the remainder
        for i in range(population_size%number_subpopulations):
            self.subpop_size[i] += 1
        '''


    def add_next_generation(self, ith_subpop, next_generation):
        # assuming next_generation is a list
        self.population[ith_subpop] += next_generation
            

    def distribute_population(self):
        '''
        using len(self.population) and not self.population_size so that we can distribute after
        mating/mutating where the population will temporarily increase in size (assuming we haven't
        already split the population)

        TODO ...but if self.population is a list of lists then just doing len() won't work
        '''
        base_count = len(self.population)//self.num_subpops #min size of each subpop
        self.subpop_size = [base_count]*self.num_subpops
        # now add the remainder
        for i in range(len(self.population)%self.num_subpops):
            self.subpop_size[i] += 1

        # TODO
        '''
        maybe we can yield subpopulations and have this be a generator...
        then self.population will stay as one big list
        '''