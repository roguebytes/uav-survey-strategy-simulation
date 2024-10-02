import argparse
from xml.sax import parse

from minefield_util import try_gen_flagfield, try_gen_minefield, get_tour, compute_tour_cost, plot_TSP

def calculate_rate(TP, FP, FN, TN):
    # TP rate = TP / (TP + FN)
    # FP rate = FP / (FP + TN)
    # FN rate = FN / (FN + TP)

    tp_rate = TP / (TP + FN)
    fp_rate = FP / (FP + TN)
    fn_rate = FN / (FN + TP)

    return round(tp_rate, 2), round(fp_rate, 2), round(fn_rate, 2)





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


    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()

    tp_rate, fp_rate, fn_rate = calculate_rate(int(args.TP), int(args.FP), int(args.FN), int(args.TN))
    print(f"TP rate={tp_rate}, FP rate={fp_rate}, FN rate={fn_rate}")

    M = try_gen_minefield(r=int(args.rows), c=int(args.cols), d=float(args.dens))
    print("\nMine field")
    print(M)

    F = try_gen_flagfield(M, TP_rate=tp_rate, FP_rate=fp_rate, FN_rate=fn_rate)
    print("\nFlag field")
    print(F)

