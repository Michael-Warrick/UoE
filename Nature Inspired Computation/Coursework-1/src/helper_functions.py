import xml.etree.ElementTree as ET
import random
import copy

class Individual:
    """Class responsible for storing all data linked to an `Individual`."""
    def __init__(self, genes: list, fitness: int, rank: float) -> None:
        self.genes = genes
        self.fitness = fitness
        self.rank = rank

    def __str__(self) -> str:
        return f"Individual:\n\tGenes: {self.genes}\n\tFitness: {self.fitness}\n\tRank: {self.rank}"

def load_cities_from_xml(xml_file_path: str) -> ET.Element:
    """Loads an `ET.Element` from an xml file using a specified file path."""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    return root[5] # <graph/> containing all <vertex/>

def generate_initial_route(cities: ET.Element) -> list:
    """Generates a `route` of all the cities specified in the xml file in order."""
    route = []

    for city in cities:
        route.append(list(cities).index(city))

    return route

def generate_permutation(route: list) -> list:
    """Generates a permutation of the original in-ordered `route`."""
    random.shuffle(route)

    return route

def find_edge_index(city: ET.Element, destination_vertex: int) -> int:
    """Finds `index` of `edge` corresponding to `desired_vertex`."""
    for index, edge in enumerate(city):
        if int(edge.text) == destination_vertex:
            return index
    return -1

def get_edge_cost(city: ET.Element, edge_index: int) -> float:
    """Returns the cost of a given `edge` in a given `city`."""
    return float(city[edge_index].attrib.get('cost'))

def calculate_route_costs(route: list, cities: ET.Element) -> list:
    """Calculates cost of every `edge` specified in `route`."""
    costs = []

    for position in range(len(route)):
        current_city_index = route[position]
        next_city_index = route[(position + 1) % len(route)]

        city = cities[current_city_index]

        selected_edge = find_edge_index(city, next_city_index)
        selected_edge_cost = get_edge_cost(city, selected_edge)

        costs.append(int(selected_edge_cost))

    return costs

def calculate_total_cost(individual_costs: list) -> int:
    """Calculates the total cost of all elements in `individual_costs`"""
    return sum(individual_costs)

def assess_generation_fitness(fitness: int) -> float:
    "Ranks the fitness of an `individual` using the inverse of the fitness as a metric (smaller distances will generate a higher ranking)."
    rank = 1 / fitness

    return rank

def tournament_selection(population: list[Individual], tournament_size: int) -> Individual:
    """Chooses a random individual from the `population`, 
    stores its `fitness` and then returns the `individual` with the best 
    `fitness` after repeating `tournament_size` times.
    """
    chosen_individuals = []

    # Randomly select individuals to compare
    for t in range(tournament_size):
        random_individual = random.choice(population)
        chosen_individuals.append(random_individual)

    # Sort individuals based on rank (descending order)
    chosen_individuals.sort(key=lambda individual: individual.rank, reverse=True)

    return chosen_individuals[0]

def single_point_crossover(a: Individual, b: Individual) -> list:
    """Performs single-point crossover on two parents `a` and `b`."""
    crossover_point = random.randint(0, len(a.genes) - 1)

    child_a_gene_data = a.genes[:crossover_point] + b.genes[crossover_point:]
    child_b_gene_data = b.genes[:crossover_point] + a.genes[crossover_point:]

    return [child_a_gene_data, child_b_gene_data]

def random_swap(arr: list) -> list:
    """A swap function that randomly swaps the values of two indices in the given `list`."""
    index = list(range(len(arr)))

    first_point = random.choice(index)
    second_point = random.choice(index)

    temp = arr[first_point]
    arr[first_point] = arr[second_point]
    arr[second_point] = temp

    return arr

def mutate(c: Individual, d: Individual) -> list:
    """Causes an chromosome (Individual) to mutate by swapping two of its genes."""
    mutation_c_gene_data = random_swap(c.genes)
    mutation_d_gene_data = random_swap(d.genes)

    return [mutation_c_gene_data, mutation_d_gene_data]

def evolutionary_algorithm(population_size: int):
    """Principal algorithm used to optimise the given problem."""

    # Step 1 - Generating population of individuals
    cities = load_cities_from_xml('data/brazil58.xml')
    initial_route = generate_initial_route(cities)

    population = []

    for i in range(population_size):
        gene_data = generate_permutation(copy.deepcopy(initial_route))
        fitness = calculate_total_cost(calculate_route_costs(gene_data, cities))
        rank = assess_generation_fitness(fitness)

        population.append(Individual(gene_data, fitness, rank))

    # Step 2 - Running tournament selection twice for a given tournament size on the population
    parent_a = tournament_selection(population, 2)
    parent_b = tournament_selection(population, 2)

    print(f"Parent_a:\n\t{parent_a}")
    print(f"Parent_b:\n\t{parent_b}")

    # Step 3 - Generating children C and D from parents A and B using single-point crossover
    children = single_point_crossover(parent_a, parent_b)

    # Recalculating individuals after performing single_point_crossover
    child_c_genes = children[0]
    child_c_fitness = calculate_total_cost(calculate_route_costs(child_c_genes, cities))
    child_c_rank = assess_generation_fitness(child_c_fitness)

    child_c = Individual(child_c_genes, child_c_fitness, child_c_rank)

    child_d_genes = children[1]
    child_d_fitness = calculate_total_cost(calculate_route_costs(child_d_genes, cities))
    child_d_rank = assess_generation_fitness(child_d_fitness)
    
    child_d = Individual(child_d_genes, child_d_fitness, child_d_rank)

    print(f"Child_c:\n\t{child_c}")
    print(f"Child_d:\n\t{child_d}")

    # Step 4 - Running mutations and then assessing fitnesses
    mutations = mutate(child_c, child_d)

    # Recalculating individuals after performing mutation
    mutant_e_genes = mutations[0]
    mutant_e_fitness = calculate_total_cost(calculate_route_costs(mutant_e_genes, cities))
    mutant_e_rank = assess_generation_fitness(mutant_e_fitness)

    mutant_e = Individual(mutant_e_genes, mutant_e_fitness, mutant_e_rank)

    mutant_f_genes = mutations[1]
    mutant_f_fitness = calculate_total_cost(calculate_route_costs(mutant_f_genes, cities))
    mutant_f_rank = assess_generation_fitness(mutant_f_fitness)

    mutant_f = Individual(mutant_f_genes, mutant_f_fitness, mutant_f_rank)

    print(f"Mutant_e:\n\t{mutant_e}")
    print(f"Mutant_f:\n\t{mutant_f}")

    # Step 5 - Running the replacement function on the mutants
    

evolutionary_algorithm(50)


# References:
# `xml.etree.ElementTree` - The ElementTree XML API: https://docs.python.org/3/library/xml.etree.elementtree.html
# `random` - Generate pseudo-random numbers: https://docs.python.org/3/library/random.html
# One-Point Crossover: https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)
# Slicing: https://python-reference.readthedocs.io/en/latest/docs/brackets/slicing.html