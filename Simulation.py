import math
import random
from typing import List, Tuple

import numpy
import pygame
import pymunk
import pymunk.pygame_util
import queue
import vidmaker
from scipy.spatial import Delaunay

import Storage
import ranges


class SimulationEnvironment:
    TERRAIN_FLAT_WORLD = 1
    TERRAIN_BUMPY = 2
    TERRAIN_INCLINED = 3
    TERRAIN_FRICTION_EXPERIMENT = 4

    def __init__(self, dt: float, gravity: float, terrain_type: int, world_size: Tuple, visualize: bool,
                 save_video: bool, terrain_params: List):
        self.terrain_params = terrain_params
        self.save_video = save_video
        self.space = pymunk.Space()
        self.space.gravity = (0, gravity)
        self.dt = dt
        self.world_size = world_size

        self.visualize = visualize
        if self.visualize:
            pygame.init()
            self.screen = pygame.display.set_mode(world_size)
            self.clock = pygame.time.Clock()
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        if self.save_video:
            self.video = vidmaker.Video("simulation_unstructured_terrain.mp4", late_export=True)

        self.terrain_type = terrain_type
        self.terrain_y_position = self.world_size[1] - 10
        self.generate_terrain(self.terrain_type)
        self.msd_morphology: Morphology = None  # To be set by the morphology itself when it is constructed
        self.zoom_ratio: pymunk.Transform = None
        self.camera_transformer = pymunk.Transform()
        self.image_saved = True  # Set to false only if you want to save image of the morphology
        self.image: bytes = None

    def zoom_in(self):
        if self.msd_morphology is not None:
            x_coordinates = numpy.array([node.position[0] for node in self.msd_morphology.mass_nodes])
            x_coordinates.sort()
            width = x_coordinates[-1] - x_coordinates[0]
            y_coordinates = numpy.array([node.position[1] for node in self.msd_morphology.mass_nodes])
            y_coordinates.sort()
            height = y_coordinates[-1] - y_coordinates[0]
            scale_ratio = (self.world_size[0] / width, self.world_size[1] / height)
            zoom = pymunk.Transform.scaling(max(scale_ratio))
            # zoom = pymunk.Transform.scaling(3)
            self.zoom_ratio = zoom
            self.draw_options.transform = (
                    pymunk.Transform.translation(self.world_size[0] / 2, self.world_size[1] / 2)
                    @ zoom
                    @ pymunk.Transform.translation(-self.world_size[0] / 2, -self.world_size[1] / 2)
            )

    def generate_terrain(self, terrain_type: int):
        if terrain_type == SimulationEnvironment.TERRAIN_FLAT_WORLD:
            ground = pymunk.Segment(
                self.space.static_body,
                (- 10 * self.world_size[0], self.terrain_y_position),
                (10 * self.world_size[0], self.terrain_y_position),
                0.0
            )

            ground.elasticity = 0.95
            ground.friction = 0.9

            self.space.add(ground)
        elif terrain_type == SimulationEnvironment.TERRAIN_BUMPY:
            ground_1 = pymunk.Segment(
                self.space.static_body,
                (- 10 * self.world_size[0], self.terrain_y_position),
                (1 * self.world_size[0], self.terrain_y_position),
                0.0
            )
            ground_1.elasticity = 0.95
            ground_1.friction = 0.9
            self.space.add(ground_1)

            for i in range(50):
                start_x = 1 * self.world_size[0] + i * 10
                ground = pymunk.Segment(
                    self.space.static_body,
                    (start_x, self.terrain_y_position),
                    (start_x + 10, self.terrain_y_position),
                    self.terrain_params[i] * 0.1
                )
                ground.elasticity = 0.95
                ground.friction = 0.9
                self.space.add(ground)

            ground_2 = pymunk.Segment(
                self.space.static_body,
                (1 * self.world_size[0] + 500, self.terrain_y_position),
                (1 * self.world_size[0] + 700, self.terrain_y_position),
                0.0
            )
            ground_2.elasticity = 0.95
            ground_2.friction = 0.9
            self.space.add(ground_2)

        elif terrain_type == SimulationEnvironment.TERRAIN_FRICTION_EXPERIMENT:
            plane_1 = pymunk.Segment(
                self.space.static_body,
                (0, 300),
                (self.world_size[0] / 2 + 100, self.terrain_y_position),
                0
            )
            plane_1.elasticity = 0.95
            plane_1.friction = 0.9

            plane_2 = pymunk.Segment(
                self.space.static_body,
                (self.world_size[0] / 2 + 100, self.terrain_y_position),
                (self.world_size[0] * 3, self.terrain_y_position),
                0
            )
            plane_2.elasticity = 0.95
            plane_2.friction = 0.9

            self.space.add(plane_1)
            self.space.add(plane_2)

    def step(self, actuate: bool) -> List:
        actuator_vals = []
        if actuate:
            for actuator in self.msd_morphology.actuators:
                amp = actuator.actuate(self.dt)
                # Amplitude waveform study
                actuator_vals.append(amp)
            # Node position study
            # for node in self.msd_morphology.mass_nodes:
            #     actuator_vals.append((node.position[0], node.position[1]))
            # actuator_vals.append(self.get_displacement())
            # Storage.DataStorage.store_to_csv(actuator_vals)

        corrected_world_center = (
            (self.world_size[0] / 2), (self.world_size[1] / 2)
        )
        centroid = self.msd_morphology.get_centroid()
        to_translate = (corrected_world_center[0] - centroid[0], 0)

        self.space.step(self.dt)
        if self.visualize:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    exit()
            try:
                translation = pymunk.Transform.translation(
                    to_translate[0], to_translate[1]
                )
                self.draw_options.transform = (
                    translation
                )
                self.screen.fill(pygame.Color("white"))

                for actuator in self.msd_morphology.actuators:
                    corrected_pos_1 = (actuator.body_1.position[0] + to_translate[0], actuator.body_1.position[1])
                    p1 = pymunk.pygame_util.to_pygame(corrected_pos_1, self.screen)
                    pygame.draw.circle(self.screen, pygame.Color("red"), p1, int(self.msd_morphology.r) + 2, 3)

                    corrected_pos_2 = (actuator.body_2.position[0] + to_translate[0], actuator.body_2.position[1])
                    p2 = pymunk.pygame_util.to_pygame(corrected_pos_2, self.screen)
                    pygame.draw.circle(self.screen, pygame.Color("red"), p2, int(self.msd_morphology.r) + 2, 3)

                for i, node in enumerate(self.msd_morphology.mass_nodes):
                    if i == self.msd_morphology.select_node:
                        node_pos = (node.position[0] + to_translate[0], node.position[1])
                        px = pymunk.pygame_util.to_pygame(node_pos, self.screen)
                        pygame.draw.circle(self.screen, pygame.Color("green"), px, int(self.msd_morphology.r) + 2, 3)

                # Draw coordinate axes markers
                for i in range(int(self.world_size[0] * 3)):
                    start_x = (i * 10) + to_translate[0]
                    start_y = self.terrain_y_position
                    end_x = start_x
                    end_y = start_y + 10
                    pygame.draw.line(self.screen, (200, 0, 0), (start_x, start_y), (end_x, end_y), width=1)

                self.space.debug_draw(self.draw_options)
                if self.save_video:
                    self.video.update(pygame.surfarray.pixels3d(self.screen).swapaxes(0, 1), inverted=False)
                pygame.display.flip()
            except TypeError:
                print("TypeError encountered")

        if not self.image_saved and self.visualize:
            self.screen.fill(pygame.Color("white"))
            self.space.debug_draw(self.draw_options)
            for actuator in self.msd_morphology.actuators:
                corrected_pos_1 = (actuator.body_1.position[0] + to_translate[0], actuator.body_1.position[1])
                p1 = pymunk.pygame_util.to_pygame(corrected_pos_1, self.screen)
                pygame.draw.circle(self.screen, pygame.Color("red"), p1, int(self.msd_morphology.r) + 2, 3)

                corrected_pos_2 = (actuator.body_2.position[0] + to_translate[0], actuator.body_2.position[1])
                p2 = pymunk.pygame_util.to_pygame(corrected_pos_2, self.screen)
                pygame.draw.circle(self.screen, pygame.Color("red"), p2, int(self.msd_morphology.r) + 2, 3)
            pygame.display.flip()
            self.image = pygame.image.tobytes(self.screen, "RGB")
            self.image_saved = True

        return actuator_vals

    def export(self):
        if self.save_video:
            self.video.export(verbose=True)

    def get_displacement(self):
        initial_position = self.msd_morphology.initial_position
        final_position = self.msd_morphology.get_centroid()
        disp = numpy.linalg.norm(final_position - initial_position)

        return disp

    def get_body_len(self):
        left_most_node_x = numpy.array([node.position[0] for node in self.msd_morphology.mass_nodes]).min()
        right_most_node_x = numpy.array([node.position[0] for node in self.msd_morphology.mass_nodes]).max()
        return right_most_node_x - left_most_node_x


class Morphology:
    MASS_ARRANGEMENT_GEOMETRIC_WITHOUT_CENTER = 0
    MASS_ARRANGEMENT_GEOMETRIC_WITH_CENTER = 1
    MASS_ARRANGEMENT_LINEAR_TRIANGULATION = 2
    MASS_ARRANGEMENT_OMNIDIRECTIONAL_TRIANGULATION = 3

    def __init__(self, simulation_environment: SimulationEnvironment, n: int, r: int, mass_per_node: float,
                 mass_arrangement_strategy: int, d_min: float, spring_constant: float, damping: float,
                 actuation_distance_factors: List[float], actuator_direction_selector: List, actuator_params: List):
        """
        This class is responsible for the generation of lattices (masses and springs) for the mass spring damper systems.
        :param actuator_direction_selector: A byte that indicates what directions, in multiples of pi/4, are the
            actuators to be selected in
        """
        self.actuator_params = actuator_params
        self.actuator_direction_selector = actuator_direction_selector  # List of boolean values mapped to directions
        # indicating whether a pair is to be found<in that direction
        self.actuation_distance_factors = actuation_distance_factors  # How far from the centroid are the selected
        # pairs are
        self.damping = damping
        self.spring_constant = spring_constant
        self.max_conn_per_node = 3
        self.simulation_environment = simulation_environment
        self.r = r
        if n < 3:
            print("Number of nodes cannot be less than 3. Setting n = 3")
            self.n = 3
        elif mass_arrangement_strategy == Morphology.MASS_ARRANGEMENT_GEOMETRIC_WITH_CENTER and n < 4:
            self.n = 4
        else:
            self.n = n
        self.mass_per_node = mass_per_node
        self.mass_arrangement_strategy = mass_arrangement_strategy
        if d_min < 3 * self.r:
            print("d_min must be at least 3 times the radius of mass nodes")
        self.d_min = d_min
        self.d_max = 2 * d_min

        self.mass_nodes = []
        self.mass_nodes_shapes = []
        self.springs = []
        self.connection_matrix = numpy.zeros((self.n, self.n))
        self.mass_spring_rotation_lock = numpy.zeros(self.n)
        self.distance_matrix = numpy.zeros((self.n, self.n))
        # 1|0 matrix to store if a connection is used as an actuator
        self.actuator_use_matrix = numpy.zeros((self.n, self.n))

        self.actuators = []
        self.points_of_actuation = []

        self.select_node = -1
        self.select_mass = 20

        self.construct_morphology()
        self.initial_position = self.get_centroid()

    def construct_morphology(self):
        self.arrange_nodes()
        # Prepare distance matrix for connection generator
        self.update_distance_matrix()
        self.create_connections()  # Populates springs
        for i in range(len(self.mass_nodes)):
            self.simulation_environment.space.add(self.mass_nodes[i], self.mass_nodes_shapes[i])
        for spring in self.springs:
            self.simulation_environment.space.add(spring)

        self.points_of_actuation = self.get_points_of_actuation()
        for i in range(len(self.points_of_actuation)):
            pair = self.points_of_actuation[i]
            phi = (numpy.pi / 2) * i
            # self.actuators.append(SineActuator(pair[0], pair[1], *self.actuator_params, phi))
            # self.actuators.append(
            #     KuramotoCPG(pair[0], pair[1], self.actuator_params[0][i], *self.actuator_params[1:], phi, self.spring_constant))
            # self.actuators.append(KuramotoCPG2(pair[0], pair[1], self.actuator_params[0][i], self.actuator_params[1], phi, i))
            self.actuators.append(
                HopfCPG(pair[0], pair[1], self.actuator_params[0][i], self.actuator_params[1], self.actuator_params[2], self.actuator_params[3], phi, i))
        # KuramotoCPG2.OSCILLATOR_COUPLINGS = self.actuator_params[-1]
        HopfCPG.OSCILLATOR_COUPLINGS = self.actuator_params[-1]
        Actuator.actuators = self.actuators

    def arrange_nodes(self) -> None:
        """
        Determine spatial positioning of nodes and make them at least self.d_min apart.
        :return: None
        """
        self.simulation_environment.msd_morphology = self
        first_node_coordinates = (
            self.simulation_environment.world_size[0] / 2, 0)
        if self.mass_arrangement_strategy == Morphology.MASS_ARRANGEMENT_GEOMETRIC_WITHOUT_CENTER:
            radius_of_geometry = (self.d_min * self.n) / (
                    2 * numpy.pi)  # We want a radius such that nodes are roughly d_min apart

            position_phase_step = (2 * numpy.pi) / self.n
            for i in range(self.n):
                angular_position = (numpy.pi / 4) + position_phase_step * i
                if i == self.select_node:
                    node_body, node_shape = self.get_node_and_body_special(self.select_mass)

                    node_body.position = (
                        first_node_coordinates[0] + radius_of_geometry * numpy.sin(angular_position),
                        first_node_coordinates[1] + radius_of_geometry * numpy.cos(angular_position)
                    )
                    self.mass_nodes.append(node_body)
                    self.mass_nodes_shapes.append(node_shape)
                else:
                    self.add_node_at_position(
                        first_node_coordinates[0] + radius_of_geometry * numpy.sin(angular_position),
                        first_node_coordinates[1] + radius_of_geometry * numpy.cos(angular_position)
                    )

        elif self.mass_arrangement_strategy == Morphology.MASS_ARRANGEMENT_GEOMETRIC_WITH_CENTER:
            radius_of_geometry = max((self.d_min * (self.n - 1)) / (2 * numpy.pi),
                                     self.d_min)  # We only have to accommodate n-1 nodes

            # The first node is placed at (0, 0)
            self.add_node_at_position(first_node_coordinates[0], first_node_coordinates[1])
            position_phase_step = (2 * numpy.pi) / (self.n - 1)
            for i in range(self.n - 1):
                angular_position = (numpy.pi / 4) + position_phase_step * i
                if i == self.select_node:
                    node_body, node_shape = self.get_node_and_body_special(self.select_mass)

                    node_body.position = (
                        first_node_coordinates[0] + radius_of_geometry * numpy.sin(angular_position),
                        first_node_coordinates[1] + radius_of_geometry * numpy.cos(angular_position)
                    )
                    self.mass_nodes.append(node_body)
                    self.mass_nodes_shapes.append(node_shape)
                else:
                    self.add_node_at_position(
                        first_node_coordinates[0] + radius_of_geometry * numpy.sin(angular_position),
                        first_node_coordinates[1] + radius_of_geometry * numpy.cos(angular_position)
                    )

        elif self.mass_arrangement_strategy == Morphology.MASS_ARRANGEMENT_LINEAR_TRIANGULATION:
            n = self.n
            factors = [
                [self.d_min * numpy.cos(numpy.pi / 4), 0, 1],
                [self.d_min * numpy.cos(numpy.pi / 4), self.d_min * numpy.sin(numpy.pi / 4), 0],
                [self.d_min * numpy.cos(numpy.pi / 4), -self.d_min * numpy.sin(numpy.pi / 4), 1]
            ]
            i = 0
            while n > 0:
                for f in factors:
                    if n > 0:
                        if self.n - n == self.select_node:
                            node_body, node_shape = self.get_node_and_body_special(self.select_mass)

                            node_body.position = (
                                first_node_coordinates[0] + f[0] * i,
                                first_node_coordinates[1] + f[1]
                            )
                            self.mass_nodes.append(node_body)
                            self.mass_nodes_shapes.append(node_shape)
                        else:
                            self.add_node_at_position(
                                first_node_coordinates[0] + f[0] * i,
                                first_node_coordinates[1] + f[1]
                            )
                        n -= 1
                        i += f[2]

        elif self.mass_arrangement_strategy == Morphology.MASS_ARRANGEMENT_OMNIDIRECTIONAL_TRIANGULATION:
            n = self.n
            expanded_node_indices = queue.Queue()
            population_matrix = numpy.zeros((self.n, self.n))
            current_node_index = numpy.array([math.floor(self.n / 2), math.floor(self.n / 2)])
            self.add_node_at_position(
                first_node_coordinates[0] + self.d_min * numpy.cos(numpy.pi / 4) * current_node_index[0],
                first_node_coordinates[1] + self.d_min * numpy.sin(numpy.pi / 4) * current_node_index[1]
            )
            placement_transforms = numpy.array([
                [[1, 1], [1, -1]],
                [[-1, 1], [1, 1]],
                [[-1, 1], [-1, -1]],
                [[-1, -1], [1, -1]]
            ])
            population_matrix[current_node_index[0]][current_node_index[1]] = 1
            i = 0
            n -= 1
            expanded_node_indices.put(current_node_index)
            while n > 0:
                transform = placement_transforms[i % len(placement_transforms)]
                index_1 = current_node_index + transform[0]
                index_2 = current_node_index + transform[1]
                for index in [index_1, index_2]:
                    if population_matrix[index[0]][index[1]] != 1 and n > 0:
                        population_matrix[index[0]][index[1]] = 1
                        expanded_node_indices.put(index)
                        if self.n - n == self.select_node:
                            node_body, node_shape = self.get_node_and_body_special(self.select_mass)

                            node_body.position = (
                                first_node_coordinates[0] + self.d_min * numpy.cos(numpy.pi / 4) * index[0],
                                first_node_coordinates[1] + self.d_min * numpy.sin(numpy.pi / 4) * index[1]
                            )
                            self.mass_nodes.append(node_body)
                            self.mass_nodes_shapes.append(node_shape)
                        else:
                            self.add_node_at_position(
                                first_node_coordinates[0] + self.d_min * numpy.cos(numpy.pi / 4) * index[0],
                                first_node_coordinates[1] + self.d_min * numpy.sin(numpy.pi / 4) * index[1]
                            )
                        n -= 1
                i += 1
                current_node_index = expanded_node_indices.get()
        # We will pull down the construction to the level of the ground
        lowest_node_y = numpy.max([node.position[1] for node in self.mass_nodes])
        offset = self.simulation_environment.terrain_y_position - lowest_node_y - self.r - 1
        for node in self.mass_nodes:
            node.position = (node.position[0], node.position[1] + offset)

    def create_connections(self):
        """
        Break down the position of nodes into triangles using Delaunay criterion and connect the thus formed triangles
        with springs.
        """
        points = [numpy.array(node.position) for node in self.mass_nodes]
        triangles = Delaunay(points).simplices
        for triangle_indices in triangles:
            for i in triangle_indices:
                for j in triangle_indices:
                    if i != j and self.connection_matrix[i][j] != 1 and self.distance_matrix[i][j] <= self.d_max:
                        self.connection_matrix[i][j] = 1  # [j,i] is not set otherwise the next part will recreate connections

        for i in range(self.n):
            for j in range(self.n):
                if self.connection_matrix[i][j] == 1 and self.distance_matrix[i][j] <= self.d_max:
                    spring = pymunk.DampedSpring(
                        a=self.mass_nodes[i], b=self.mass_nodes[j],
                        anchor_a=(0, 0), anchor_b=(0, 0),
                        rest_length=self.distance_matrix[i][j],
                        stiffness=self.spring_constant, damping=self.damping
                    )

                    if self.mass_spring_rotation_lock[i] == 0:
                        self.mass_spring_rotation_lock[i] = 1
                        spring.pre_solve = Morphology.correct_angle

                    self.springs.append(spring)
                elif self.connection_matrix[i][j] == 1 and self.distance_matrix[i][j] > self.d_max:
                    self.connection_matrix[i][j] = 0

    def get_points_of_actuation(self) -> List:
        """
        Finds n pairs of bodies that are the furthest from centroid in directions from dirs
        :return body_pairs_for_actuation: List of pairs to be selected for actuation
        """
        if len(self.actuator_direction_selector) != len(self.actuation_distance_factors):
            print("Direction selector and distance factors should have the same dimensions")
        body_pairs_for_actuation = []
        centroid = self.get_centroid()
        max_pairs = self.n / 2  # The maximum number of pairs that can be selected cannot exceed the number of mass-nodes connections
        for x in range(len(self.actuator_direction_selector)):
            if self.actuator_direction_selector[x] == 1 and max_pairs > 0:
                direction = x * (numpy.pi / 4)
                distance_from_centroid = self.actuation_distance_factors[x] * self.d_min
                furthest_point_in_direction = numpy.array([
                    centroid[0] + distance_from_centroid * numpy.cos(direction),
                    centroid[1] + distance_from_centroid * numpy.sin(direction)
                ])
                node_points = numpy.array([node.position for node in self.mass_nodes])
                distances = [numpy.linalg.norm(node_point - furthest_point_in_direction) for node_point in node_points]
                index_of_sorted_distances = numpy.argsort(distances)
                pair = [self.mass_nodes[index_of_sorted_distances[0]]]
                for i in range(1, len(index_of_sorted_distances)):
                    if self.connection_matrix[index_of_sorted_distances[0]][index_of_sorted_distances[i]] == 1 and \
                            (self.actuator_use_matrix[index_of_sorted_distances[0]][index_of_sorted_distances[i]] == 0 and self.actuator_use_matrix[index_of_sorted_distances[i]][index_of_sorted_distances[0]] == 0):
                        pair.append(self.mass_nodes[index_of_sorted_distances[i]])
                        # Both are required because indexing can be a,b and b,a in a matrix
                        self.actuator_use_matrix[index_of_sorted_distances[0]][index_of_sorted_distances[i]] = 1
                        self.actuator_use_matrix[index_of_sorted_distances[i]][index_of_sorted_distances[0]] = 1
                        break
                if len(pair) < 2:
                    print("Impossible case: unconnected mass")
                else:
                    body_pairs_for_actuation.append(pair)
                    max_pairs -= 1

        return body_pairs_for_actuation

    def update_distance_matrix(self):
        for i in range(len(self.mass_nodes)):
            for j in range(len(self.mass_nodes)):
                distance = numpy.linalg.norm(numpy.array(self.mass_nodes[i].position) - self.mass_nodes[j].position)
                self.distance_matrix[i][j] = distance

    def get_node_and_body(self):
        node_body = pymunk.Body(
            self.mass_per_node, pymunk.moment_for_circle(
                mass=self.mass_per_node,
                inner_radius=0,
                outer_radius=self.r,
                offset=(0, 0)
            )
        )

        node_shape = pymunk.Circle(node_body, self.r, (0, 0))
        node_shape.elasticity = 0.95
        node_shape.friction = 0.9

        return node_body, node_shape

    def get_node_and_body_special(self, mass):
        node_body = pymunk.Body(
            mass, pymunk.moment_for_circle(
                mass=mass,
                inner_radius=0,
                outer_radius=self.r,
                offset=(0, 0)
            )
        )

        node_shape = pymunk.Circle(node_body, self.r, (0, 0))
        node_shape.elasticity = 0.95
        node_shape.friction = 0.9

        return node_body, node_shape

    def add_node_at_position(self, pos_x, pos_y):
        node_body, node_shape = self.get_node_and_body()
        node_body.position = (
            pos_x,
            pos_y
        )
        self.mass_nodes.append(node_body)
        self.mass_nodes_shapes.append(node_shape)

    def get_centroid(self):
        morphology_coordinates = numpy.array([node.position for node in self.mass_nodes])
        centroid = morphology_coordinates.mean(axis=0)
        return centroid

    @staticmethod
    def angle(a, b):
        return math.atan2((b[1] - a[1]), (b[0] - a[0]))

    @staticmethod
    def correct_angle(constraint, space):
        constraint.a.angular_velocity = 0
        constraint.a.angle = Morphology.angle(constraint.a.position, constraint.b.position)
        pass


class Actuator:
    ACTUATOR_TYPE_SINE_WAVE = 0
    ACTUATOR_TYPE_KURAMOTO_CPG = 1
    ACTUATOR_TYPE_KURAMOTO2_CPG = 2
    ACTUATOR_TYPE_HOPF_CPG = 3
    actuators: List = None

    def __init__(self, body_1: pymunk.Body, body_2: pymunk.Body):
        self.body_1 = body_1
        self.body_2 = body_2
        self.time_passed = 0.0

    def get_amplitude_for_timestep(self, dt):
        return 0

    def actuate(self, t: float) -> float:
        self.time_passed += t
        point_of_force = (
            (self.body_1.position[0] + self.body_2.position[0]) / 2,
            (self.body_1.position[1] + self.body_2.position[1]) / 2,
        )
        amplitude = self.get_amplitude_for_timestep(self.time_passed)
        # print(amplitude)
        body_1_force_vector = (
            (self.body_1.position[0] - point_of_force[0]) * amplitude,
            (self.body_1.position[1] - point_of_force[1]) * amplitude
        )
        body_2_force_vector = (
            (self.body_2.position[0] - point_of_force[0]) * amplitude,
            (self.body_2.position[1] - point_of_force[1]) * amplitude
        )
        self.body_1.apply_force_at_world_point(
            body_1_force_vector,
            point_of_force
        )
        self.body_2.apply_force_at_world_point(
            body_2_force_vector,
            point_of_force
        )
        return amplitude


class SineActuator(Actuator):
    def __init__(self, body_1: pymunk.Body, body_2: pymunk.Body, A: float, phi: float):
        super().__init__(body_1, body_2)
        self.phi = phi
        self.A = A

    def get_amplitude_for_timestep(self, dt):
        return (self.A * numpy.sin(2 * numpy.pi * dt + self.phi)) / 2
        # return (self.A * numpy.sin(2 * numpy.pi + self.phi)) / 2

    @classmethod
    def get_randomized_parameters(cls) -> List:
        return [random.randint(*ranges.AMPLITUDE_OF_ACTUATION_RANGE)]


class KuramotoCPG(Actuator):
    OSCILLATOR_COUPLINGS: List[List] = None

    def __init__(self, body_1: pymunk.Body, body_2: pymunk.Body, omega: float, A: float, K: float, phi: float, spring_stiffness: float):
        super().__init__(body_1, body_2)
        self.phi = phi
        self.omega = omega * 0.1 * 2 * numpy.pi
        self.theta = 0.0
        self.true_theta = 0.0
        self.A = A
        self.K = K
        self.natural_frequency = (spring_stiffness * (self.body_1.mass + self.body_2.mass) / (self.body_1.mass * self.body_2.mass))**0.5
        self.spring_rest_length = numpy.linalg.norm(numpy.array(self.body_1.position) - self.body_2.position)
        self.prev_thetas = [0.0 for _ in range(2)]
        self.last_spring_length = 0
        self.change_counter = 0
        self.wavelength = 0

    def get_feedback_term(self, theta) -> float:
        f = self.true_theta - theta
        return numpy.sin(numpy.pi * f)

    def get_amplitude_for_timestep(self, dt):
        feedback = (self.K / len(Actuator.actuators)) * numpy.array(
            [actuator.get_feedback_term(self.theta) for actuator in Actuator.actuators]).sum()
        self.prev_thetas.append(self.true_theta)
        self.prev_thetas.pop(0)
        # length_feedback = 0
        current_spring_length = abs(numpy.linalg.norm(numpy.array(self.body_1.position) - self.body_2.position))
        length_feedback = numpy.sin(numpy.pi * (self.last_spring_length - current_spring_length))
        self.last_spring_length = current_spring_length
        length_feedback = 0
        self.true_theta += (self.omega + feedback + length_feedback) * dt * 0.01
        self.theta = 0
        for prev_theta in self.prev_thetas:
            self.theta += prev_theta
        self.theta /= 2

        force = (self.A * numpy.sin((self.theta / self.K) + self.phi)) / 2  # A simple low pass filter
        return force  # Dividing by K to normalize the theta value between zero and +-one

    @classmethod
    def get_randomized_parameters(cls) -> List:
        # return [random.randint(*ranges.OMEGA_RANGE), 2000,
        #         random.randint(*ranges.K_RANGE)]
        return [[random.randint(*ranges.OMEGA_RANGE) for _ in range(8)], random.randint(*ranges.AMPLITUDE_OF_ACTUATION_RANGE), random.randint(*ranges.K_RANGE)]


class KuramotoCPG2(Actuator):
    OSCILLATOR_COUPLINGS: List[List[int]] = None

    def __init__(self, body_1: pymunk.Body, body_2: pymunk.Body, omega: float, A: float, phi: float, index: int):
        super().__init__(body_1, body_2)
        self.phi = phi
        self.omega = omega * 0.01
        self.theta = 0.0
        self.true_theta = 0.0
        self.A = A
        self.i = index
        self.spring_rest_length = numpy.linalg.norm(numpy.array(self.body_1.position) - self.body_2.position)
        self.prev_thetas = [0.0 for _ in range(2)]
        self.change_counter = 0
        self.wavelength = 0

    def get_feedback_term(self, theta, j) -> float:
        f = self.true_theta - theta
        return KuramotoCPG2.OSCILLATOR_COUPLINGS[self.i][j] * numpy.sin(f)

    def get_amplitude_for_timestep(self, dt):
        feedback = numpy.array(
            [actuator.get_feedback_term(self.theta, self.i) for actuator in Actuator.actuators]).sum()
        # self.prev_thetas.append(self.true_theta)
        # self.prev_thetas.pop(0)
        # length_feedback = 0
        current_spring_length = numpy.linalg.norm(numpy.array(self.body_1.position) - self.body_2.position)
        length_feedback = ((self.spring_rest_length - current_spring_length) / self.spring_rest_length) * 0.01
        self.true_theta += (self.omega + feedback + length_feedback) * dt * 0.01
        # self.theta = 0
        # for prev_theta in self.prev_thetas:
        #     self.theta += prev_theta
        # self.theta /= 2

        force = (self.A * numpy.sin(self.true_theta + self.phi)) / 2  # A simple low pass filter
        return force  # Dividing by K to normalize the theta value between zero and +-one

    @classmethod
    def get_randomized_parameters(cls) -> List:
        coupling_matrix = [[0 for _ in range(8)] for __ in range(8)]
        for i in range(8):
            for j in range(8):
                if i == j:
                    coupling_matrix[i][i] = 0
                else:
                    coupling_matrix[i][j] = random.randint(*ranges.K_RANGE)
        for i in range(8):
            for j in range(i, 8):
                coupling_matrix[i][j] = coupling_matrix[j][i]
        return [
            [random.randint(*ranges.OMEGA_RANGE) for _ in range(8)],
            random.randint(*ranges.AMPLITUDE_OF_ACTUATION_RANGE),
            coupling_matrix
        ]


class HopfCPG(Actuator):
    OSCILLATOR_COUPLINGS: List[List[int]] = None

    def __init__(self, body_1: pymunk.Body, body_2: pymunk.Body, omega: float, A: float, alpha: int, beta: int, phi: float, index: int):
        super().__init__(body_1, body_2)
        self.beta = beta * 0.01
        self.alpha = alpha * 0.01
        self.omega = omega * 0.01
        self.phi = phi
        self.i = index
        self.A = A
        self.x = 1
        self.y = 1
        self.spring_rest_length = numpy.linalg.norm(numpy.array(self.body_1.position) - self.body_2.position)

    def r2(self):
        return self.x**2 + self.y**2

    def dx(self):
        return self.alpha * (1 - self.r2()) * self.x + self.omega * self.y

    def dy(self):
        return self.beta * (1 - self.r2()) * self.y - self.omega * self.x + numpy.array([actuator.coupling_term(self.i) for actuator in Actuator.actuators]).sum()

    def coupling_term(self, j):
        return HopfCPG.OSCILLATOR_COUPLINGS[self.i][j] * self.y

    def get_amplitude_for_timestep(self, dt):
        current_spring_length = numpy.linalg.norm(numpy.array(self.body_1.position) - self.body_2.position)
        feedback = (self.spring_rest_length - current_spring_length) / self.spring_rest_length
        self.x += self.dx()
        self.y += self.dy() + feedback * 0.01
        return self.A * self.x / 2

    @classmethod
    def get_randomized_parameters(cls) -> List:
        coupling_matrix = [[0 for _ in range(8)] for __ in range(8)]
        for i in range(8):
            for j in range(8):
                if i == j:
                    coupling_matrix[i][i] = 0
                else:
                    coupling_matrix[i][j] = random.randint(*ranges.K_RANGE) * 0.0001
        for i in range(8):
            for j in range(i, 8):
                coupling_matrix[i][j] = coupling_matrix[j][i]
        # print(numpy.array(coupling_matrix))
        return [
            [1 for _ in range(8)],
            random.randint(*ranges.AMPLITUDE_OF_ACTUATION_RANGE),
            # 8000,
            random.randint(*ranges.ALPHA_BETA_RANGE),
            random.randint(*ranges.ALPHA_BETA_RANGE),
            coupling_matrix
        ]
