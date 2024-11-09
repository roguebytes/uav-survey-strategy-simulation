import argparse
from timeit import timeit
from xml.sax import parse
import numpy as np
import codecs, json
import timeit
import time
import NumpyEncoder
import statistics

"""
Last updated Saturday 9th Nov

@author: Frank
f.loewenich@hdr.qut.edu.au

"""

from minefield_util import gen_flag_field, try_gen_flagfield, try_gen_minefield, get_tour, compute_tour_cost, plot_TSP

def run_simulation():
    ### Logging ###
    experiment_start_time = time.time()
    ###############

    # Simulation attributes
    i_density = 1 # index into density array
    i_precision = 0 # index into precision array
    c_experiment = 0 # iteration count of the experiment-loop
    n_density = 20 # number of values to generate for the P_density array
    n_precision = 20 # number of values to generate for the P_range array
    n_experiment = 30 # number of experiments for each density/precision combination
    p_1m = .9  # Precision at 1m established with actual drone camera and YOLOv9
    discount = .98 # Used to reduce precision when generating a range
    R = 0.98  # Target Recall

    # Minefield attributes
    rows = 30
    cols = 30
    low_density = .01
    high_density = .9

    # Note: In this script values for PRECISION and DENSITY are MANUALLY selected for each experiment
    # Generate a range of precision values
    #   starting from p_1m and
    #   every subsequent value reduced by 'discount'
    P_range = [p_1m * discount ** k for k in range(n_precision)]
    print(f"Precision range: {P_range}")
    # Generate a range of densities between the low and high values
    D_range = list(np.linspace(low_density, high_density, n_density))
    print(f"Density range: {D_range}")

    # Set the density
    density = D_range[i_density] # MANUALLY select the density value
    print(f"Density: {density}")
    result_array = []
    for k in range(n_experiment):
        c_experiment += 1
        # Generate a minefield
        M = try_gen_minefield(r=int(rows), c=int(cols), d=float(density))
        print("Mine field generated")
        # print(M)

        # Compute TP, FP and FN
        P = P_range[i_precision]  # Set the precision value
        print(f"Precision: {P}")
        TP = R * len(M.nonzero()[0])
        FP = TP * (1 / P - 1)
        FN = TP * (1 / R - 1)
        F = gen_flag_field(M, TP, FP, FN)

        print("Flag field generated")
        # print(F)

        ### Logging ###
        tsp_start_time = time.time()
        ###############

        # Generate TSP path to visit flagged locations
        tsp_path, G = get_tour(F, cell_side=1)
        print("Shortest TSP Path:", tsp_path)
        # plot_TSP(tsp_path, G)

        ### Logging ###
        tsp_end_time = time.time()
        tsp_exec_time = tsp_end_time - tsp_start_time
        print(f"TSP exec time: {round(tsp_exec_time, 2)} seconds")
        ###############

        # Compute and print the tour cost
        tour_cost = compute_tour_cost(G, tsp_path)
        print("Cost of the TSP Tour:", round(tour_cost, 2))
        result_array.append(tour_cost)

        ### Logging ###
        experiment_end_time = time.time()
        experiment_exec_time = experiment_end_time - experiment_start_time
        print(f"Experiment exec time: {round(experiment_exec_time, 2)} seconds")

        experiment_log = {}
        experiment_log['count'] = c_experiment
        experiment_log['density'] = density
        experiment_log['precision'] = P
        experiment_log['cost'] = round(tour_cost, 2)
        experiment_log['tsp_exec_time'] = round(tsp_exec_time, 2)

        e_file_path = "experiment_log.json"
        json.dump(experiment_log, open(e_file_path, "a"), indent=4)

        if c_experiment == n_experiment-1:
            pd_log = {}
            pd_log['density'] = density
            pd_log['precision'] = P
            pd_log['experiments'] = n_experiment
            # pd_log['result_list'] = result_array
            pd_log['mean_cost'] = statistics.mean(result_array)

            p_file_path = "precision_density_log.json"
            json.dump(pd_log, open(p_file_path, "a"), indent=4)
        ###############


if __name__ == "__main__":
    run_simulation()