import GeneticAlgorithm
import Simulation
import Storage
import main
from GeneticAlgorithm import GeneticParameters
from main import run_sim
from PIL import Image


strat_names = ['LT', 'OT', 'GC', 'GN']
mass_arrangement_strats = [
        Simulation.Morphology.MASS_ARRANGEMENT_LINEAR_TRIANGULATION,
        Simulation.Morphology.MASS_ARRANGEMENT_OMNIDIRECTIONAL_TRIANGULATION,
        Simulation.Morphology.MASS_ARRANGEMENT_GEOMETRIC_WITH_CENTER,
        Simulation.Morphology.MASS_ARRANGEMENT_GEOMETRIC_WITHOUT_CENTER,
    ]
for i in range(3, 20):
    for j in range(len(mass_arrangement_strats)):
        gene = GeneticParameters(
                        number_of_nodes=i,
                        mass_per_node=1,
                        mass_arrangement_strategy=mass_arrangement_strats[j],
                        d_min=160,
                        spring_constant=5,
                        spring_damping=5,
                        actuator_params=0,
                        actuator_direction_selector=[0]
                    )

        data = run_sim(gene)
        image = Image.frombytes("RGB", main.WORLD_SIZE, data.image)
        filename = "images/mass_spring_" + str(i) + "_" + strat_names[j] + ".png"
        image.save(filename)

