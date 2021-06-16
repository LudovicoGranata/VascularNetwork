
#in the present work, the total surface area was
#used as cost function for optimization, while the required total flow and the
#size of the perfusion area were kept constant


from segment import Segment
from segment import Tree
import numpy as np
import random
from scipy.optimize import minimize

'''
The generation of the tree has the following structure:
     generation of the starting segment (root) randomly in the tissue volume
     for t_term :
        generate a terminal location
        generate the neighborhood segments of the terminal location
        for every element in the neighborhood:
             attach the terminal location to the segment creating a bifurcation
             optimize the position of the bifurcation minimizing the lateral surface of the entire tree
        take the best segment and the best bifurcation and make it permanent

'''
def generate_tree(t_term=50, n_near=2, box_dimension=10, start=np.array([10, 10, 9])):
    root = Segment(start, np.array([random.random() * box_dimension,
                                    random.random() * box_dimension,
                                    random.random() * box_dimension]))
    tree = Tree(root)
    for i in range(t_term - 1):
        print("\r generating terminal segment number: " + str(i+1) + "/" + str(t_term), end="")
        x_term = generate_terminal_location(tree, box_dimension=box_dimension)
        neighborhood = tree.find_neighborhood(x_term, dim=n_near)
        best = -1
        best_seg = None
        best_x = None
        for seg in neighborhood:
            added_segment, other_segment, parent_segment = tree.add(x_term, seg)
            cons = [{'type': 'ineq', 'fun': constraint1, 'args': (added_segment,)},
                    {'type': 'ineq', 'fun': constraint2, 'args': (other_segment,)},
                    {'type': 'ineq', 'fun': constraint3, 'args': (parent_segment,)}
                    ]
            result = minimize(tree.cost_function, x0=[10, 10, 9], args=(added_segment, tree), constraints=cons)
            if best == -1:
                best = result.g.p_best
                best_seg = seg
                best_x = result.g.pos_p_best
            else:
                if result.g.p_best < best:
                    best = result.g.p_best
                    best_seg = seg
                    best_x = result.g.pos_p_best

            tree.delete(added_segment)

        added_segment, other_segment, parent_segment = tree.add(x_term, best_seg)
        added_segment.update_start(best_x)
    tree.update_all()
    tree.print()


'''
Generate the terminal location, we can generate so that the new location is positioned
not too far from the other segments and not too near (d_min and d_max)
'''
def generate_terminal_location(tree, box_dimension=10, d_min=2, d_max=10):
    found = 0
    x_term = None
    while found == 0:
        x_term = np.array([random.random() * box_dimension,
                           random.random() * box_dimension,
                           random.random() * box_dimension])
        d_min_exceed = 0
        d_max_not_found = 0
        for seg in tree.segments:
            if seg.distance_point(x_term) < d_max:
                d_max_not_found = 1
            if seg.distance_point(x_term) < d_min:
                d_min_exceed = 1
                break

        if d_min_exceed == 0 and d_max_not_found == 1:
            found = 1
    return x_term

'''
the following three constraints have to be used so that the three segments resulting from
the creation of a new bifurcation don't degenerate to zero.
'''
def constraint1(x, seg):
    radius = seg.radius()
    seg.update_start(x)
    return seg.length() - 2*radius


def constraint2(x, seg):
    radius = seg.radius()
    seg.update_start(x)
    return seg.length() - 2*radius


def constraint3(x, seg):
    radius = seg.radius()
    seg.update_end(x)
    return seg.length() - 2*radius


if __name__ == '__main__':
    generate_tree()
