import copy
import numpy as np


class Particle:
    def __init__(self, x_0, v):
        self.x = x_0
        self.v = v
        self.p_best = None
        self.pos_p_best = None


class Swarm:
    def __init__(self, function, dimensions, bounds, constraints=None, n_particle=20, iterations=12, args=None):
        self.function = function
        self.swarm = []
        self.n_particle = n_particle
        self.g = None
        self.dimensions = dimensions
        self.iterations = iterations
        self.constraints = constraints
        self.bounds = bounds
        self.reference = []
        self.args = args

        # generate the swarm positions and velocities
        if constraints is not None:
            self.__generate_swarm_heuristic()
            self.__create_reference()
        else:
            self.__generate_swarm(n_particle)

        # assign best values and positions
        self.__p_best()
        # best in the neighborhood
        self.__best_neighborhood()
        # update positions and  velocities
        for k in range(iterations-1):
            self.__update_velocities_positions(k)
            if self.constraints is not None:
                self.__repair_operator()
            self.__p_best()
            self.__best_neighborhood()

    def __generate_swarm(self, n_particle):
        vel = np.ones(self.dimensions)
        for _ in range(n_particle):
            random_point = np.empty(self.dimensions)
            for i in range(self.dimensions):
                random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
            self.swarm.append(Particle(random_point, vel))

    # generate velocities
    def __update_velocities_positions(self, k):
        w = (1.2 - 0.5) * ((self.iterations-(k+1))/self.iterations) + 0.5 # Time-Varying Inertia Weight (TVIW)
        c1 = 2.05
        c2 = 2.05
        phi = c1 + c2
        chi = 2.0/abs((2-phi-np.sqrt(phi**2 - 4 * phi))) #constriction factor

        for particle in self.swarm:
            U_1 = np.random.rand()
            U_2 = np.random.rand()
            local_adjustment = c1 * U_1 * (particle.pos_p_best - particle.x)
            global_adjustment = c2 * U_2 * (self.g.pos_p_best - particle.x)
            particle.v = chi * (w * particle.v + local_adjustment + global_adjustment)
            particle.x += particle.v

    # assign best values and positions
    def __p_best(self):
        for particle in self.swarm:
            cost = self.function(particle.x, *self.args)
            if particle.p_best is None or particle.p_best > cost:
                particle.p_best = cost
                particle.pos_p_best = copy.deepcopy(particle.x)

    # the best particle among all particles
    def __best_neighborhood(self):
        for particle in self.swarm:
            if self.g is None or particle.p_best < self.g.p_best:
                self.g = particle

    def __generate_swarm_heuristic(self):
        random_point = np.empty(self.dimensions)
        vel = np.ones(self.dimensions)
        for i in range(self.dimensions):
            random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
        while not self.__feasible_point(random_point):
            for i in range(self.dimensions):
                random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
        self.swarm.append(Particle(random_point, vel))
        for _ in range(self.n_particle-1):
            random_point = np.empty(self.dimensions)
            for i in range(self.dimensions):
                random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
            if not self.__feasible_point(random_point):
                feasible_particle_pos = self.swarm[np.random.randint(len(self.swarm)) - 1].x
                while not self.__feasible_point(random_point):
                    random_point = 0.6*random_point + 0.4*feasible_particle_pos
            self.swarm.append(Particle(random_point, vel))

    def __create_reference(self, reference_dim=5):
        for _ in range (reference_dim):
            random_point = np.empty(self.dimensions)
            vel = np.zeros(self.dimensions)
            for i in range(self.dimensions):
                random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
            while not self.__feasible_point(random_point):
                for i in range(self.dimensions):
                    random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]

            self.reference.append(Particle(random_point, vel))

    def __repair_operator(self):
        for particle in self.swarm:
            if not self.__feasible_point(particle.x):
                reference_particle_pos = self.reference[np.random.randint(len(self.reference)) - 1].x
                while not self.__feasible_point(particle.x):
                    particle.x = 0.9 * particle.x + 0.1 * reference_particle_pos

    def __feasible_point(self, random_point):
        eps = 0.001
        for cons in self.constraints:
            type = cons['type']
            fun = cons['fun']
            args = cons['args']
            if type =="ineq":
                if not fun(random_point, *args) > 0:
                    return False
            if type == "eq":
                if not abs(fun(random_point, *args)) <= eps:
                    return False

        return True
