import os
import time

import matplotlib.pyplot as pyplot
import numpy
import pandas

import GeneticAlgorithm
import Storage
from Storage import DataStorage
import multiprocessing

# To deal with ALSA warnings on WSL
os.environ['SDL_AUDIODRIVER'] = 'dsp'
import Simulation

RUNTIME = 50  # seconds
STABALIZATION_TIME = 3
WORLD_SIZE = (600, 600)
MASS_RADIUS = 10

_run_count = 0


# def run_sim(data: GeneticAlgorithm.GeneticParameters, terrain_params) -> GeneticAlgorithm.GeneticParameters:
def run_sim(gen_set) -> GeneticAlgorithm.GeneticParameters:
    data = gen_set[0]
    terrain_params = gen_set[1]
    dt = 1.0 / 1000.0
    number_of_steps_stabalize = int(STABALIZATION_TIME / dt)
    number_of_steps = int(RUNTIME / dt)
    environment = Simulation.SimulationEnvironment(
        dt=dt,
        gravity=980,  # cm s-2
        terrain_type=Simulation.SimulationEnvironment.TERRAIN_BUMPY,
        world_size=WORLD_SIZE,
        visualize=True,
        save_video=True,
        terrain_params=terrain_params
    )

    morphology = Simulation.Morphology(
        environment,
        n=data.number_of_nodes,
        r=data.mass_radius,
        mass_per_node=data.mass_per_node,
        mass_arrangement_strategy=data.mass_arrangement_strategy,
        d_min=data.d_min,  # Min distance between nodes
        spring_constant=data.spring_constant,
        damping=data.spring_damping,
        actuation_distance_factors=data.actuator_distance_factor,
        actuator_params=data.actuator_params,
        actuator_direction_selector=data.actuator_direction_selector
    )

    # Let the morphology take its natural orientation under gravity only
    for i in range(number_of_steps_stabalize):
        environment.step(False)

    # Now start the actuation
    amplitudes = []
    for i in range(number_of_steps):
        amplitudes.append(environment.step(True))

    data.displacement = environment.get_displacement()
    data.runtime = RUNTIME
    data.body_length_per_second = abs(data.displacement) / (environment.get_body_len() * RUNTIME)
    data.image = environment.image


    # Amplitude waveform
    # print(data.body_length_per_second)
    # amps = numpy.array(amplitudes).reshape((-1, number_of_steps))
    # for a in amps:
    #     pyplot.plot(a)
    # print(amplitudes[0])
    # Node positions
    # positions = numpy.array(amplitudes)
    # print(positions[0])
    # for node_count in range(data.number_of_nodes):
    #     position_data = positions[:, node_count]
    #     pyplot.plot(position_data[:, 0], WORLD_SIZE[1] - 10 - position_data[:, 1])
    # velocity
    # displacements = numpy.array(amplitudes).reshape(-1)
    # velocity = []
    # for i in range(1, len(displacements)):
    #     velocity.append(abs(displacements[i] - displacements[i-1]) / dt)
    # acceleration = []
    # for i in range(1, len(velocity)):
    #     acceleration.append((abs(velocity[i]) - abs(velocity[i-1])) / dt)
    # pyplot.plot(velocity)
    # pyplot.show()

    environment.export()
    return data


storage = DataStorage("simulation_data.db")
df = storage.get_pandas_dataframe()
latest_experiment = df[df['experiment'] == 94].sort_values(by=["blps"], ascending=False)
fastest_boi = latest_experiment.iloc[0]
gene = GeneticAlgorithm.GeneticParameters.get_genotype_from_data(fastest_boi.to_dict())
# gene.actuator_params = Simulation.HopfCPG.get_randomized_parameters()
# print(numpy.array(gene.actuator_params[4]))
# print(gene.body_length_per_second)
# result = run_sim(gene)
# storage.close()

terrain_param = pandas.read_csv('terr.csv')
# gene_sets = [(gene, param.tolist()) for param in terrain_param.iloc[0:1]]
run_sim((gene, terrain_param.iloc[17]))
# results = multiprocessing.Pool(4).map(run_sim, gene_sets)
# for result in results:
#     result.experiment = 301
#     result.generation = -1
#     storage.store_with_command(result.get_database_storage_command())

storage.close()

# storage = DataStorage("simulation_data.db")
# df = storage.get_pandas_dataframe()
# latest_experiment = df[df['experiment'] == 94].sort_values(by=["blps"], ascending=False)
# fastest_boi = latest_experiment.iloc[0]
# print(fastest_boi.to_dict())
# gene_1 = GeneticAlgorithm.GeneticParameters.get_genotype_from_data(fastest_boi.to_dict())
# latest_experiment = df[df['experiment'] == 96].sort_values(by=["blps"], ascending=False)
# fastest_boi = latest_experiment.iloc[0]
# print(fastest_boi.to_dict())
# gene_2 = GeneticAlgorithm.GeneticParameters.get_genotype_from_data(fastest_boi.to_dict())
# # print(gene_1.aslist())
# # print(gene_2.aslist())
# storage.close()

# exp_no = 96
# experiment = df[df['experiment'] == exp_no]
# generations = experiment['generation'].unique()
# generations.sort()

# hopf_best_genes = []
# hopf_without_feedback = []
# for gen in generations:
#     gen_data = experiment[experiment['generation'] == gen].sort_values(by=["blps"], ascending=False)
#     fastest = gen_data.iloc[0]
#     gene = GeneticAlgorithm.GeneticParameters.get_genotype_from_data(fastest.to_dict())
#     hopf_best_genes.append(gene)
#
# print("Calculating ", len(hopf_best_genes), " without feedback")
# results = multiprocessing.Pool(4).map(run_sim, hopf_best_genes)
# for result in results:
#     result.experiment = 940
#     result.generation = -1
#     storage.store_with_command(result.get_database_storage_command())
#
# storage.close()

# gene = GeneticAlgorithm.GeneticParameters.get_initial_population(1)[0]
# gene.number_of_nodes = 6
# result = run_sim(gene)
# print(result.displacement)
# print(result.body_length_per_second)

# init_population_size = 10
#
# initial_population = GeneticAlgorithm.GeneticParameters.get_systematic_population()
# data_storage = Storage.DataStorage("simulation_data.db")
# #
# #
# process_pool = multiprocessing.Pool(4)
# results = process_pool.map(run_sim, initial_population)
# for result in results:
#     data_storage.store_with_command(result.get_database_storage_command())
#
# print("Saved ", len(initial_population), " results.")
# data_storage.close()

# start = time.time()
# evolution_experiment = GeneticAlgorithm.Evolution(100, 20, Storage.DataStorage("simulation_data.db"), run_sim)
# evolution_experiment.run_sim()
# elapsed = time.time() - start
# print("Completed in ", elapsed / 60, " minutes")
