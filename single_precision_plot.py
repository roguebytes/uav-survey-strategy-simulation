import argparse
from timeit import timeit
from xml.sax import parse
import numpy as np
import codecs, json
import timeit
import time
import NumpyEncoder

"""
Last updated Friday 8th Nov

@author: Frank
f.loewenich@hdr.qut.edu.au

Key variables
simulation_log : array that logs individual experiment results as json
density_increment_log : array that logs density increment results as json
R : target RECALL
discount : reduction parameter applied to P_range


"""

from minefield_util import gen_flag_field, try_gen_flagfield, try_gen_minefield, get_tour, compute_tour_cost, plot_TSP

def run_simulation(rows, cols, low_density, high_density):
    '''
    :param rows:
    :param cols:
    :param low_density:
    :param high_density:
    :return:
    '''
    simulation_log = []
    R = 0.98    # Target Recall
    discount = .98
    p_1m = .9 # Precision at 1m established with actual drone camera and YOLOv9
    #p_4m = .61 # Precision at 4m established with actual drone camera and YOLOv9
    n_density = 3  # 20
    n_experiment = 1  # 30
    P_range = [p_1m * discount ** k for k in range(3)] # original 20
    D_range = list(np.linspace(low_density, high_density, n_density))
    # result_array = np.zeros((len(P_range), n_density, n_experiment), dtype=np.float32)

    density = D_range[1]
    print(f"Density: {density}")
    M = try_gen_minefield(r=int(rows), c=int(cols), d=float(density))
    print("Mine field generated")
    # print(M)

    # Compute TP, FP and FN
    P = P_range[1]  # p # p_1m * discount
    print(f"Precision: {P}")
    TP = R * len(M.nonzero()[0])
    FP = TP * (1 / P - 1)
    FN = TP * (1 / R - 1)
    F = gen_flag_field(M, TP, FP, FN)

    print("Flag field generated")
    # print(F)

    # Generate TSP path to visit flagged locations
    tsp_path, G = get_tour(F, cell_side=1)
    print("Shortest TSP Path:", tsp_path)
    # plot_TSP(tsp_path, G)

    # Compute and print the tour cost
    tour_cost = compute_tour_cost(G, tsp_path)
    print("Cost of the TSP Tour:", tour_cost)




if __name__ == "__main__":
    # Define the command-line parser
    parser = argparse.ArgumentParser(description='Single-Experiment Relative Precision Plot')
    parser.add_argument('--rows', action="store", dest='rows', default=1)
    parser.add_argument('--cols', action="store", dest='cols', default=1)
    parser.add_argument('--lowdens', action="store", dest='lowdens', default=1)
    parser.add_argument('--highdens', action="store", dest='highdens', default=1)

    # Parse and store command-line arguments
    args = parser.parse_args()

    run_simulation(int(args.rows), int(args.cols), float(args.lowdens), float(args.highdens))