import argparse
from xml.sax import parse
import numpy as np

from minefield_util import gen_flag_field, try_gen_flagfield, try_gen_minefield, get_tour, compute_tour_cost, plot_TSP

def calculate_rate(TP, FP, FN, TN):
    # TP rate = TP / (TP + FN)
    # FP rate = FP / (FP + TN)
    # FN rate = FN / (FN + TP)

    tp_rate = TP / (TP + FN)
    fp_rate = FP / (FP + TN)
    fn_rate = FN / (FN + TP)

    return round(tp_rate, 2), round(fp_rate, 2), round(fn_rate, 2)

def run_simulation(rows, cols, low_density, high_density):
    R = 0.98    # Target Recall
    discount = .98
    p_1m = .9 # Precision at 1m established with actual drone camera and YOLOv9
    p_4m = .61 # Precision at 4m established with actual drone camera and YOLOv9
    P_range = [p_1m * discount ** k for k in range(20)]

    for density in np.linspace(low_density, high_density, 20):
        for p in P_range:
            print(f"Running simulation with density {str(density)}")
            # Nested loop to compute average and standard deviation
            M = try_gen_minefield(r=int(rows), c=int(cols), d=float(density))
            print("\nMine field generated")
            # print(M)

            # Compute TP, FP and FN
            P = p_1m * discount
            TP = R * len(M.nonzero()[0])
            FP = TP * (1/P - 1)
            FN = TP * (1/R - 1)
            F = gen_flag_field(M, TP, FP, FN)

            print("\nFlag field generated")
            # print(F)

            tsp_path, G = get_tour(F, cell_side=1)
            print("Shortest TSP Path:", tsp_path)

            # Compute and print the tour cost
            tour_cost = compute_tour_cost(G, tsp_path)
            print("Cost of the TSP Tour:", tour_cost)

            #dens += dens_increment

            # plot_TSP(tsp_path, G)

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

    tp_rate, fp_rate, fn_rate = calculate_rate(int(args.TP), int(args.FP), int(args.FN), int(args.TN))
    print(f"TP rate={tp_rate}, FP rate={fp_rate}, FN rate={fn_rate}")

    run_simulation(int(args.rows), int(args.cols), float(args.lowdens), float(args.highdens))#, dens_increment=0.05, iterations=3)

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