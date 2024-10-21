from __future__ import annotations

from typing import List, Tuple

import ranges
import random
import copy
import multiprocessing
from Simulation import KuramotoCPG, SineActuator, KuramotoCPG2, HopfCPG, Actuator

CURRENT_EXPERIMENT = 96
ACTUATOR_TYPE = Actuator.ACTUATOR_TYPE_HOPF_CPG


class Evolution:
    def __init__(self, population_size: int, generations: int, storage, simulation_function):
        self.storage = storage
        self.population_size = population_size
        self.generations = generations
        self.process_pool = multiprocessing.Pool(4)
        self.simulation_function = simulation_function
        self.previous_best = 0

    def run_sim(self):
        # population = GeneticParameters.get_initial_population(self.population_size)
        population, gen = GeneticParameters.get_last_known_population(CURRENT_EXPERIMENT, self.storage)
        for i in range(gen + 1, self.generations + gen + 1):
            if i != 0:
                results: List[GeneticParameters] = self.process_pool.map(self.simulation_function, population[10:]) + population[:10]
                results.sort(key=lambda o: o.body_length_per_second, reverse=True)  # Re-sorting is required since top 10 were appended at last and may no longer be top 10
            else:
                results: List[GeneticParameters] = self.process_pool.map(self.simulation_function, population)
            for result in results:
                result.experiment = CURRENT_EXPERIMENT
                result.generation = i + 1
                self.storage.store_with_command(result.get_database_storage_command())
            print("Saved generation: ", i)
            population = Evolution.select_fittest(results)

            # Calculating growth
            if population[0].body_length_per_second > self.previous_best:
                self.previous_best = population[0].body_length_per_second
                print("Growth seen at gen ", i + 1, " with blps: ", self.previous_best)

        print("Evolution complete")
        self.storage.close()

    @staticmethod
    def select_fittest(generation: List[GeneticParameters]) -> List[GeneticParameters]:
        fittest = []
        generation.sort(key=lambda o: o.body_length_per_second, reverse=True)
        # Preserve top 10
        for gene in generation[:10]:
            fittest.append(gene)

        # Crossover top 10
        reproduce_set_1 = generation[:5]
        reproduce_set_2 = generation[5:10]
        for g1 in reproduce_set_1:
            for g2 in reproduce_set_2:
                # crossovers = Evolution.weightless_arithmetic_crossover(
                #     g1.aslist(),
                #     g2.aslist()
                # )
                crossovers = Evolution.one_point_crossover(
                    g1.aslist(),
                    g2.aslist()
                )
                # print("Crossovers: ", len(crossovers))
                for gene in crossovers:
                    fittest.append(GeneticParameters.fromlist(gene))

        # Add 10 mutated genes
        to_mutate: List[GeneticParameters] = generation[:10]
        for gene in to_mutate:
            fittest.append(GeneticParameters.fromlist(Evolution.mutate(gene.aslist())))

        return fittest

    @staticmethod
    def one_point_crossover(a: List, b: List):
        new_genes = []
        if len(a) != len(b):
            print("Error in crossover: list sizes not equal")
            return None
        crossover_point = int(len(a) / 2)
        new_gene_1 = a[:crossover_point] + (b[crossover_point:])
        new_gene_2 = b[:crossover_point] + (a[crossover_point:])
        new_genes.append(new_gene_1)
        new_genes.append(new_gene_2)

        return new_genes

    @staticmethod
    def weightless_arithmetic_crossover(a: List, b: List) -> List[List]:
        if len(a) != len(b):
            print("Error in crossover: list sizes not equal")
            return None
        new_genes = []
        arithmetic_gene = []
        arithmetic_gene.append(int((a[0] + b[0]) / 2))  # Number of nodes
        arithmetic_gene.append((a[1] + b[1]) / 2)  # Mass per node
        arithmetic_gene.append((a[2] + b[2]) / 2)  # Mass radius
        arithmetic_gene.append(a[3])  # Mass arrangement strategy
        arithmetic_gene.append((a[4] + b[4]) / 2)  # Spring Constant
        arithmetic_gene.append((a[5] + b[5]) / 2)  # Spring Damping

        new_crossovers_actuator_params = Evolution.one_point_crossover(a[6], b[6])
        for i in new_crossovers_actuator_params:
            gene = copy.deepcopy(arithmetic_gene)
            gene.append(i)
            new_genes.append(gene)
        new_crossovers_actuator_direction = Evolution.one_point_crossover(a[7], b[7])
        new_crossovers_actuator_distance = Evolution.one_point_crossover(a[8], b[8])
        for i in range(len(new_genes)):
            new_genes[i].append(new_crossovers_actuator_direction[i])
            new_genes[i].append(new_crossovers_actuator_distance[i])
            new_genes[i][3] = random.choice([a[3], b[3]])

        return new_genes

    @staticmethod
    def mutate(a: List) -> List:
        mutate_at = random.randint(0, len(a))
        if mutate_at == 0:
            a[mutate_at] = random.randint(*ranges.NUM_OF_NODES_RANGE)
        elif mutate_at == 1:
            a[mutate_at] = random.randint(*ranges.MASS_OF_NODES_RANGE)
        elif mutate_at == 2:
            a[mutate_at] = random.randint(*ranges.RADIUS_OF_NODES_RANGE)
        elif mutate_at == 3:
            a[mutate_at] = random.choice(ranges.MASS_ARRANGEMENT_STRATS)
        elif mutate_at == 4:
            a[mutate_at] = random.randint(*ranges.SPRING_CONSTANT_RANGE)
        elif mutate_at == 5:
            a[mutate_at] = random.randint(*ranges.SPRING_DAMPING_RANGE)
        elif mutate_at == 6:
            if ACTUATOR_TYPE == Actuator.ACTUATOR_TYPE_SINE_WAVE:
                a[mutate_at] = SineActuator.get_randomized_parameters()
            elif ACTUATOR_TYPE == Actuator.ACTUATOR_TYPE_KURAMOTO_CPG:
                a[mutate_at] = KuramotoCPG.get_randomized_parameters()
            elif ACTUATOR_TYPE == Actuator.ACTUATOR_TYPE_KURAMOTO2_CPG:
                a[mutate_at] = KuramotoCPG2.get_randomized_parameters()
            elif ACTUATOR_TYPE == Actuator.ACTUATOR_TYPE_HOPF_CPG:
                a[mutate_at] = HopfCPG.get_randomized_parameters()
        elif mutate_at == 7:
            a[mutate_at] = [random.randint(0, 1) for _ in range(8)]
        elif mutate_at == 8:
            a[mutate_at] = [random.randint(*ranges.D_ACTUATE_RANGE) for _ in range(8)]

        return a


class GeneticParameters:
    def __init__(self, number_of_nodes: int, mass_per_node: int, mass_radius: int, mass_arrangement_strategy: int,
                 spring_constant: float, spring_damping: int, actuator_params: List[int],
                 actuator_direction_selector: List,
                 actuator_distance_factor: List[int], experiment_num: int = -1, generation_num: int = -1):
        self.number_of_nodes = number_of_nodes
        self.mass_per_node = mass_per_node
        self.mass_radius = mass_radius
        self.mass_arrangement_strategy = mass_arrangement_strategy
        self.d_min = mass_radius * 3
        self.spring_constant = spring_constant
        self.spring_damping = spring_damping
        self.actuator_params = actuator_params
        self.actuator_direction_selector = actuator_direction_selector
        self.actuator_distance_factor = actuator_distance_factor
        self.parameter_count = 8  # Number of source parameters above
        self.displacement = 0.0
        self.runtime = 0.0
        self.body_length_per_second = 0
        self.experiment = experiment_num
        self.generation = generation_num

    def get_database_storage_command(self):
        return 'INSERT INTO simulation VALUES ({}, {}, {}, {}, {}, {}, "{}", {}, "{}", "{}", {}, {}, {}, {}, {});'.format(
            self.number_of_nodes,
            self.mass_per_node,
            self.mass_arrangement_strategy,
            self.d_min,
            self.spring_constant,
            self.spring_damping,
            str(self.actuator_params),
            0,
            str(self.actuator_direction_selector),
            str(self.actuator_distance_factor),
            self.displacement,
            self.runtime,
            self.body_length_per_second,
            self.experiment,
            self.generation
        )

    def aslist(self) -> List:
        return [
            self.number_of_nodes,
            self.mass_per_node,
            self.mass_radius,
            self.mass_arrangement_strategy,
            self.spring_constant,
            self.spring_damping,
            self.actuator_params,
            self.actuator_direction_selector,
            self.actuator_distance_factor
        ]

    @staticmethod
    def fromlist(a) -> GeneticParameters:
        return GeneticParameters(
            *a
        )

    @staticmethod
    def get_initial_population(n: int) -> List:
        populations = []
        for i in range(n):
            actuator_params:List = None
            if ACTUATOR_TYPE == Actuator.ACTUATOR_TYPE_SINE_WAVE:
                actuator_params = SineActuator.get_randomized_parameters()
            elif ACTUATOR_TYPE == Actuator.ACTUATOR_TYPE_KURAMOTO_CPG:
                actuator_params = KuramotoCPG.get_randomized_parameters()
            elif ACTUATOR_TYPE == Actuator.ACTUATOR_TYPE_KURAMOTO2_CPG:
                actuator_params = KuramotoCPG2.get_randomized_parameters()
            elif ACTUATOR_TYPE == Actuator.ACTUATOR_TYPE_HOPF_CPG:
                actuator_params = HopfCPG.get_randomized_parameters()
            gene = GeneticParameters(
                number_of_nodes=random.randint(*ranges.NUM_OF_NODES_RANGE),
                mass_per_node=random.randint(*ranges.MASS_OF_NODES_RANGE),
                mass_radius=random.randint(*ranges.RADIUS_OF_NODES_RANGE),
                mass_arrangement_strategy=random.choice(ranges.MASS_ARRANGEMENT_STRATS),
                spring_constant=random.randint(*ranges.SPRING_CONSTANT_RANGE),
                spring_damping=random.randint(*ranges.SPRING_DAMPING_RANGE),
                actuator_params=actuator_params,
                actuator_direction_selector=[random.randint(0, 1) for _ in range(8)],
                actuator_distance_factor=[random.randint(*ranges.D_ACTUATE_RANGE) for _ in range(8)],
            )
            gene.body_length_per_second = 0
            populations.append(gene)

        return populations

    @staticmethod
    def get_last_known_population(exp: int, storage) -> Tuple[List[GeneticParameters], int]:
        """
        Finds the latest generation for given experiment from database.
        :param exp: The experiment number to look for
        :param storage: The database to look in
        :return: Tuple consisting of the list of @GeneticParameters from the
        latest generation and the latest generation as an integer
        """
        df = storage.get_pandas_dataframe()
        latest_df = df[(df["experiment"] == exp)]
        if len(latest_df) > 0:
            latest_generation = latest_df["generation"].max()
            latest_df = latest_df[df.generation == latest_generation]
            genes = []
            for row in latest_df.iloc:
                gene = GeneticParameters.get_genotype_from_data(row)
                genes.append(gene)
            genes.sort(key=lambda o: o.body_length_per_second, reverse=True)
        else:
            genes = GeneticParameters.get_initial_population(100)
            latest_generation = 0

        return genes, latest_generation

    @staticmethod
    def get_genotype_from_data(dataframe_row) -> GeneticParameters:
        genotype = GeneticParameters(
            number_of_nodes=dataframe_row["n"],
            mass_per_node=dataframe_row["m"],
            mass_radius=dataframe_row["d_min"] / 3,
            mass_arrangement_strategy=dataframe_row["lattice_construct"],
            spring_constant=dataframe_row["k"],
            spring_damping=dataframe_row["damping"],
            actuator_params=eval(dataframe_row["actuator_param_1"]),
            actuator_direction_selector=eval(dataframe_row["actuator_direction_selection"]),
            actuator_distance_factor=eval(dataframe_row["actuator_distance_factor"]),
            experiment_num=dataframe_row["experiment"],
            generation_num=dataframe_row["generation"],
        )
        genotype.displacement = dataframe_row["displacement"]
        genotype.runtime = dataframe_row["runtime"]
        genotype.body_length_per_second = dataframe_row["blps"]
        return genotype

    @classmethod
    def get_systematic_population(cls) -> List[GeneticParameters]:
        populations = []
        for i in range(10):
            gene = GeneticParameters(
                number_of_nodes=random.randint(*ranges.NUM_OF_NODES_RANGE),
                mass_per_node=random.randint(*ranges.MASS_OF_NODES_RANGE),
                mass_radius=random.randint(*ranges.RADIUS_OF_NODES_RANGE),
                mass_arrangement_strategy=0,
                spring_constant=random.randint(*ranges.SPRING_CONSTANT_RANGE),
                spring_damping=random.randint(*ranges.SPRING_DAMPING_RANGE),
                actuator_params=KuramotoCPG.get_randomized_parameters(),
                actuator_direction_selector=[random.randint(0, 1) for _ in range(8)],
                actuator_distance_factor=[random.randint(*ranges.D_ACTUATE_RANGE) for _ in range(8)],
                experiment_num=12
            )
            # for strat in ranges.MASS_ARRANGEMENT_STRATS:
            #     variant = copy.deepcopy(gene)
            #     variant.mass_arrangement_strategy = strat
            #     populations.append(variant)
            populations.append(gene)

        return populations
