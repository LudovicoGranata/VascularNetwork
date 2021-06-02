import copy
import numpy as np


class Particle:
    def __init__(self, x_0, v):
        self.x = x_0
        self.v = v
        self.n_constraints = None
        self.p_best = None
        self.pos_p_best = None
        self.feasible_best = None
        self.n_constraints = None


class Swarm:
    def __init__(self, function, dimensions, bounds, constraints=None, n_particle=25, iterations=40, args=None):
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
        self.__generate_swarm()
        # assign best values and positions
        self.__p_best()
        # best in the neighborhood
        self.__best_neighborhood()
        k = 0
        non_improving = 0
        eps = 0.1
        while k < (iterations-1) and non_improving < 5:
            current_best = copy.deepcopy(self.g.p_best)
            self.__update_velocities_positions(k)
            self.__p_best()
            self.__best_neighborhood()
            if 0 < abs(current_best - self.g.p_best) < eps:
                non_improving += 1
            k += 1

    def __generate_swarm(self):
        vel = np.ones(self.dimensions)
        for _ in range(self.n_particle):
            random_point = np.empty(self.dimensions)
            for i in range(self.dimensions):
                random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
            self.swarm.append(Particle(random_point, vel))

    # generate velocities
    def __update_velocities_positions(self, k):
        w = (1.2 - 0.1) * ((self.iterations-(k+1))/self.iterations) + 0.1 # Time-Varying Inertia Weight (TVIW)
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
    '''
    Assign best values and positions so that:
        Any feasible solution is preferred to any infeasible solution.
        Between two feasible solutions, the one having better objective function value is preferred.
        Between two infeasible solutions, the one having smaller constraint violation is preferred.
    '''

    def __p_best(self):
        for particle in self.swarm:
            if self.args is None:
                cost = self.function(particle.x)
            else:
                cost = self.function(particle.x, *self.args)
            particle.n_constraints = self.__n_constraints_unsatisfied(particle.x)
            if particle.p_best is None:
                particle.p_best = cost
                particle.pos_p_best = copy.deepcopy(particle.x)
                particle.n_constraints_best = copy.deepcopy(particle.n_constraints)
                continue
            if particle.n_constraints < particle.n_constraints_best:
                particle.p_best = cost
                particle.pos_p_best = copy.deepcopy(particle.x)
                particle.n_constraints_best = copy.deepcopy(particle.n_constraints)
                continue

            if particle.n_constraints == 0 and particle.n_constraints == particle.n_constraints_best and cost < particle.p_best:
                particle.p_best = cost
                particle.pos_p_best = copy.deepcopy(particle.x)
                particle.n_constraints_best = copy.deepcopy(particle.n_constraints)

   
   
    # the best particle among all particles (global neighborhood)
    def __best_neighborhood(self):
        for particle in self.swarm:
            if self.g is None:
                self.g = copy.deepcopy(particle)
                continue
            if particle.n_constraints_best < self.g.n_constraints_best:
                self.g = copy.deepcopy(particle)
                continue
            if particle.n_constraints_best == 0 and particle.n_constraints_best == self.g.n_constraints_best and particle.p_best < self.g.p_best:
                self.g = copy.deepcopy(particle)


    def __generate_swarm_heuristic(self):
        random_point = np.empty(self.dimensions)
        vel = np.ones(self.dimensions)
        for i in range(self.dimensions):
            random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
        while not self.__is_feasible(random_point):
            for i in range(self.dimensions):
                random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
        self.swarm.append(Particle(random_point, vel))
        for _ in range(self.n_particle-1):
            random_point = np.empty(self.dimensions)
            for i in range(self.dimensions):
                random_point[i] = np.random.rand() * (abs(self.bounds[i][1] - self.bounds[i][0])) + self.bounds[i][0]
            if not self.__is_feasible(random_point):
                feasible_particle_pos = self.swarm[np.random.randint(len(self.swarm)) - 1].x
                while not self.__is_feasible(random_point):
                    random_point = 0.6*random_point + 0.4*feasible_particle_pos
            self.swarm.append(Particle(random_point, vel))


    def __n_constraints_unsatisfied(self, point):
        eps = 0.0001
        n_constraints = 0
        for cons in self.constraints:
            type = cons['type']
            fun = cons['fun']
            args = cons['args']
            if type =="ineq":
                if not fun(point, *args) > 0:
                     n_constraints += 1
            if type == "eq":
                if not abs(fun(point, *args)) <= eps:
                    n_constraints += 1
        return n_constraints

    def __is_feasible(self, point):
        return self.__n_constraints_unsatisfied(point) == 0
