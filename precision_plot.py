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
    D_range = list(enumerate(np.linspace(low_density, high_density, n_density)))

    result_array = np.zeros((len(P_range), n_density, n_experiment), dtype=np.float32)
    for i_density, density in enumerate(np.linspace(low_density, high_density, n_density)):
        # DENSITY iteration
        density_increment_start_time = time.time()
        density_increment_log = {}
        density_increment_log['density'] = density
        for i_p, p in enumerate(P_range):
            # PRECISION iteration
            simulation_start_time = time.time()
            # Nested loop to compute average and standard deviation
            for k in range(n_experiment):
                # EXPERIMENT iteration
                experiment_start_time = time.time()
                print("\n-----------------------------------------\n")
                print(f"Running density increment number {str(i_density + 1)} of {str(n_density)}.")
                print(f"Running simulation number {i_p + 1} of {str(len(P_range))}.")
                print(f"Running simulation with density {str(density)}.")
                print(f"Running simulation with precision (P) of {p}.")
                print(f"Running experiment number {str(k + 1)} of {str(n_experiment)}.")
                M = try_gen_minefield(r=int(rows), c=int(cols), d=float(density))
                print("Mine field generated")
                # print(M)

                # Compute TP, FP and FN
                P = p # p_1m * discount
                TP = R * len(M.nonzero()[0])
                FP = TP * (1/P - 1)
                FN = TP * (1/R - 1)
                F = gen_flag_field(M, TP, FP, FN)

                print("Flag field generated")
                # print(F)

                tsp_path, G = get_tour(F, cell_side=1)
                print("Shortest TSP Path:", tsp_path)

                # Compute and print the tour cost
                tour_cost = compute_tour_cost(G, tsp_path)
                print("Cost of the TSP Tour:", tour_cost)

                result_array[i_p, i_density, k] = tour_cost

                experiment_end_time = time.time()
                experiment_execution_time = round(experiment_end_time - experiment_start_time, 2)
                print(f"Experiment execution time was {round(experiment_execution_time / 60, 2)} minutes ({experiment_execution_time} seconds).")
                # Log
                density_increment_log['density_increment'] = i_density
                density_increment_log['simulation_number'] = i_p
                density_increment_log['density_value'] = density
                density_increment_log['experiment_number'] = k
                density_increment_log['tour_cost'] = tour_cost
                density_increment_log['experiment_execution_time'] = experiment_execution_time
                if k < (n_experiment - 1):
                    simulation_log.append(density_increment_log)
            #dens += dens_increment
            simulation_end_time = time.time()
            simulation_execution_time = round(simulation_end_time - simulation_start_time, 2)
            print(f"Simulation execution time was {round(simulation_execution_time / 60, 2)} minutes ({simulation_execution_time} seconds)")
            density_increment_log['simulation_execution_time'] = simulation_execution_time
            if i_p < (len(P_range) - 1):
                simulation_log.append(density_increment_log)
            # plot_TSP(tsp_path, G)
        density_increment_end_time = time.time()
        density_increment_execution_time = round(density_increment_end_time - density_increment_start_time, 2)
        print(f"Density increment execution time was {round(density_increment_execution_time / 60, 2)} minutes")
        density_increment_log['density_increment_execution_time'] = density_increment_execution_time
        simulation_log.append(density_increment_log)
    mean_array = np.mean(result_array, axis=2)      # result_array is 3D, mean_array is 2D
    std_array = np.std(result_array, axis=2)        # result_array is 3D, std_array is 2D

    # Save the simulation log to JSON file
    with open('simulation_log.json', 'w') as json_file:
        json.dump(simulation_log, json_file, indent=4)
    #
    # # Save the result arrays to JSON files
    # with open('mean_array.json', 'w') as json_file:
    #     json.dump(mean_array, json_file, indent=4, cls=NumpyEncoder)
    #
    # with open('std_array.json', 'w') as json_file:
    #     json.dump(std_array, json_file, indent=4, cls=NumpyEncoder)

    a = np.arange(10).reshape(2, 5)  # a 2 by 5 array
    mean_list = mean_array.tolist()  # nested lists with same data, indices
    std_list = std_array.tolist()

    file_path_mean = "/mean_log.json"  ## your path variable
    json.dump(mean_list, codecs.open(file_path_mean, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)  ### this saves the array in .json format

    file_path_std = "/std_log.json"  ## your path variable
    json.dump(std_list, codecs.open(file_path_std, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)  ### this saves the array in .json format

    # with open('precision_plot_log.json', 'w') as json_file:
    #     json.dumps({'mean': mean_array, 'std':std_array}, cls=NumpyEncoder)





# def run_simulation():
#     discount = .98
#     P_range =  [ p_1m * discount**k for k in range(20)]  # to get 20 values
#
#     low_density, high_density = 0.01, 0.9
#
#     # have the number of cells in M as a constant
#
#     for density in np.linspace(low_density, high_density, 20):  # should play with other values than 20
#         for p in P_range:
#             # make mine field M
#             # compute cost strategy 1
#             # compute TP, FP, FN as discussed  in the chat yesterday evening
#             F = gen_flag_field(M, TP, FP, FN):
#             # compute cost strategy 2
#             # accumulate in result table


if __name__ == "__main__":
    # Define the parser
    parser = argparse.ArgumentParser(description='Relative Precision Plot')
    # Declare an argument (`--xxxx`), saying that the
    # corresponding value should be stored in the `xxxx`
    # field, and using a default value if the argument
    # isn't given
    parser.add_argument('--TP', action="store", dest='TP', default=1)
    parser.add_argument('--FN', action="store", dest='FN', default=1)
    parser.add_argument('--FP', action="store", dest='FP', default=1)
    parser.add_argument('--TN', action="store", dest='TN', default=1)
    parser.add_argument('--rows', action="store", dest='rows', default=1)
    parser.add_argument('--cols', action="store", dest='cols', default=1)
    parser.add_argument('--dens', action="store", dest='dens', default=1)
    parser.add_argument('--lowdens', action="store", dest='lowdens', default=1)
    parser.add_argument('--highdens', action="store", dest='highdens', default=1)

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()

    # tp_rate, fp_rate, fn_rate = calculate_rate(int(args.TP), int(args.FP), int(args.FN), int(args.TN))
    # print(f"TP rate={tp_rate}, FP rate={fp_rate}, FN rate={fn_rate}")

    run_simulation(int(args.rows), int(args.cols), float(args.lowdens), float(args.highdens))
    # execution_time = timeit.timeit(run_simulation(int(args.rows), int(args.cols), float(args.lowdens), float(args.highdens)), number=1)#, dens_increment=0.05, iterations=3)
    # print(f"Execution time: {execution_time} seconds")
    # M = try_gen_minefield(r=int(args.rows), c=int(args.cols), d=float(args.dens))
    # print("\nMine field")
    # print(M)

    # F = try_gen_flagfield(M, TP_rate=tp_rate, FP_rate=fp_rate, FN_rate=fn_rate)
    # print("\nFlag field")
    # print(F)
    #
    # tsp_path, G = get_tour(F, cell_side=1)
    # print("Shortest TSP Path:", tsp_path)
    #
    # # Compute and print the tour cost
    # tour_cost = compute_tour_cost(G, tsp_path)
    # print("Cost of the TSP Tour:", tour_cost)
    #
    # plot_TSP(tsp_path, G)