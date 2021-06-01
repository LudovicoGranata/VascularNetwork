# By defining the terminal pressures
# to be uniform, we can determine the global resistance from the
# total flow rate, Q0, and pressure drop from source to terminals,
# deltaPs, allowing the root radius, r0, to be calculated.


import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import norm

viscosity = 0.64
delta_Ps = 40000


class Segment:
    def __init__(self, x_start, x_term, father=None, left=None, right=None, b_left=1, b_right=1, q_flow=0.125):
        self.x_start = x_start
        self.x_term = x_term
        self.father = father
        self.left = left
        self.right = right
        self.b_left = b_left
        self.b_right = b_right
        self.q_flow = q_flow
        self.R = None

    def length(self):
        return math.sqrt((self.x_term[0] - self.x_start[0]) ** 2 +
                         (self.x_term[1] - self.x_start[1]) ** 2 +
                         (self.x_term[2] - self.x_start[2]) ** 2)

    def radius(self):
        result = 1
        child = self
        father = self.father
        while father is not None:
            if father.left is child:
                result *= father.b_left
            if father.right is child:
                result *= father.b_right
            child = father
            father = father.father

        result = result * (((child.q_flow * child.R) / delta_Ps) ** (1.0 / 4))
        return result

    def lateral_surface(self):
        return 2 * math.pi * self.radius() * self.length()

    def update_start(self, x_new):
        father = self.father
        if father is not None:
            father.x_term = x_new
            father.left.x_start = x_new
            father.right.x_start = x_new

    def update_end(self, x_new):
        self.x_term = x_new
        if self.left is not None or self.right is not None:
            self.left.x_start = x_new
            self.right.x_start = x_new
            return

    def update_radii(self):
        child = self.father
        while child is not None:
            RL = child.left.R
            RR = child.right.R
            QL = child.left.q_flow
            QR = child.right.q_flow
            rr_rl = ((RR * QR) / (RL * QL)) ** (1 / 4)
            child.q_flow = QR + QL
            child.b_right = (1 + rr_rl ** (-3)) ** (-1.0 / 3)
            child.b_left = (1 + rr_rl ** 3) ** (-1.0 / 3)
            child.R = ((8 * viscosity) / math.pi) * child.length() + (
                        ((child.b_left ** 4) / child.left.R) + ((child.b_right ** 4) / child.right.R)) ** (-1)
            child = child.father

    def distance_point(self, x):
        p = x
        a = self.x_start
        b = self.x_term
        d = np.divide(b - a, np.linalg.norm(b - a))
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)
        h = np.maximum.reduce([s, t, 0])
        c = np.cross(p - a, d)
        return np.hypot(h, np.linalg.norm(c))


class Tree:
    def __init__(self, root):
        self.root = root
        self.root.R = ((8 * viscosity) / math.pi) * root.length()
        self.segments = [root]
        self.q_flow = 0.125

    def add(self, x_term, segment):
        p_start = segment.x_start
        p_end = segment.x_term
        middle_point = np.array([(p_start[0] + p_end[0]) / 2.0,
                                 (p_start[1] + p_end[1]) / 2.0,
                                 (p_start[2] + p_end[2]) / 2.0])
        new_segment = Segment(middle_point, x_term, father=segment)
        other_new_segment = Segment(middle_point, p_end, father=segment, left=segment.left, right=segment.right,
                                    b_left=segment.b_left, b_right=segment.b_right)
        segment.x_term = middle_point
        segment.right = new_segment
        segment.left = other_new_segment
        if other_new_segment.left is not None or other_new_segment.left is not None:
            other_new_segment.left.father = other_new_segment
            other_new_segment.right.father = other_new_segment
        self.segments.append(new_segment)
        self.segments.append(other_new_segment)
        new_segment.R = ((8 * viscosity) / math.pi) * new_segment.length()
        other_new_segment.R = ((8 * viscosity) / math.pi) * other_new_segment.length()
        if other_new_segment.left is not None or other_new_segment.right is not None:
            other_new_segment.R = other_new_segment.R + (
                        ((other_new_segment.b_left ** 4) / other_new_segment.left.R) + (
                            (other_new_segment.b_right ** 4) / other_new_segment.right.R)) ** (-1)

        self.update_all()
        return new_segment, other_new_segment, segment

    def delete(self, segment):
        if segment.left is not None or segment.right is not None:
            raise Exception("Sorry you can't delete a segments that has children")
        father = segment.father
        other_child = None
        if father.left is segment:
            other_child = father.right

        if father.right is segment:
            other_child = father.left

        father.x_term = other_child.x_term
        father.left = other_child.left
        father.right = other_child.right
        if other_child.left is not None or other_child.right is not None:
            other_child.left.father = father
            other_child.right.father = father

        self.segments.remove(segment)
        self.segments.remove(other_child)

    def lateral_surface(self):
        total = 0
        for seg in self.segments:
            total += seg.lateral_surface()
        return total

    def __visualize_cylinder_two_point_rad (self, points_radius):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print("generating visualization...")

        for i in range(len(points_radius)):
            p0 = points_radius[i][0][0]
            p1 = points_radius[i][0][1]
            R = points_radius[i][1]

            # vector in direction of axis
            v = p1 - p0
            # find magnitude of vector
            mag = norm(v)
            # unit vector in direction of axis
            v = v / mag
            # make some vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            # make vector perpendicular to v
            n1 = np.cross(v, not_v)
            # normalize n1
            n1 /= norm(n1)
            # make unit vector perpendicular to v and n1
            n2 = np.cross(v, n1)
            # surface ranges over t from 0 to length of axis and 0 to 2*pi
            t = np.linspace(0, mag, 2)
            theta = np.linspace(0, 2 * np.pi, 6)
            # use meshgrid to make 2d arrays
            t, theta = np.meshgrid(t, theta)
            # generate coordinates for surface
            X, Y, Z = [p0[i] + v[i] * t + R * np.sin(theta) * n1[i] + R * np.cos(theta) * n2[i] for i in [0, 1, 2]]
            ax.plot_surface(X, Y, Z, color="red")
            # plot axis
            ax.set_xlim(-5, 15)
            ax.set_ylim(-5, 15)
            ax.set_zlim(-5, 15)
        plt.show()

    def print(self):
        coordinates_radius = []
        for seg in self.segments:
            coordinates_radius.append([[seg.x_start, seg.x_term], seg.radius()])
            print(str(seg) + " : " +
                  str(seg.x_start) +
                  str(seg.x_term) +
                  " length: " + str(seg.length())
                  + " radius: " + str(seg.radius()))
        self.__visualize_cylinder_two_point_rad(coordinates_radius)

    def find_neighborhood(self, x, dim=10):
        neighborhood = []
        distances = []
        for i in range(len(self.segments)):
            distance = self.segments[i].distance_point(x)
            distances.append(distance)
            neighborhood.append(self.segments[i])

        distances = np.array(distances)
        neighborhood = np.array(neighborhood)
        sorted_index = distances.argsort()[0:dim]
        neighborhood = neighborhood[sorted_index]
        return neighborhood

    def update_all(self):
        already_updated = []
        for seg in self.segments:
            if seg.left is None or seg.right is None:
                seg.q_flow = self.q_flow
                if seg not in already_updated:
                    already_updated.append(seg)
                    already_updated.append(self.brother_segment(seg))
                    seg.update_radii()

    def brother_segment(self, seg):
        father = seg.father
        if seg.father is None:
            return None
        if father.left == seg:
            return father.right
        if father.right == seg:
            return father.left

    def cost_function(self, x, seg, tree):
        seg.update_start(x)
        self.update_all()
        return tree.lateral_surface()
