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

def run_simulation():

    # Initialize simulation parameters
    i_density = 0 # index into density array
    i_precision = 0 # index into precision array
    # Minefield attributes
    rows = 30
    cols = 30
    low_density = .01
    high_density = .9
    simulation_log = []
    # Object Detector
    R = 0.98    # Target Recall

    # Generate a range of precision values
    #   starting from p_1m and
    #   every subsequent value reduced by 'discount'
    # In this script values for PRECISION and DENSITY are MANUALLY selected for each experiment
    p_1m = .9  # Precision at 1m established with actual drone camera and YOLOv9
    # p_4m = .61 # Precision at 4m established with actual drone camera and YOLOv9
    discount = .98
    P_range = [p_1m * discount ** k for k in range(3)] # original 20
    # Generate a range of densities between the low and high values
    n_density = 3  # 20
    D_range = list(np.linspace(low_density, high_density, n_density))

    # Set the density
    density = D_range[i_density] # MANUALLY select the density value
    print(f"Density: {density}")
    # Generate a minefield
    M = try_gen_minefield(r=int(rows), c=int(cols), d=float(density))
    print("Mine field generated")
    # print(M)

    # Compute TP, FP and FN
    P = P_range[i_precision]  # MANUALLY select the precision value
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
    run_simulation()