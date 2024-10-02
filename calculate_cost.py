import argparse
import math

def calculate_diagonal(length, width):
    diagonal = math.sqrt(length**2 + width**2)
    return diagonal

if __name__ == "__main__":
    # Define the parser
    parser = argparse.ArgumentParser(description='Cost Calculator')
    # Declare an argument (`--xxxx`), saying that the
    # corresponding value should be stored in the `xxxx`
    # field, and using a default value if the argument
    # isn't given
    parser.add_argument('--fov', action="store", dest='fov', default=1)
    parser.add_argument('--area', action="store", dest='area', default=1)
    parser.add_argument('--speed', action="store", dest='speed', default=1)
    parser.add_argument('--overlap', action="store", dest='overlap', default=1)
    parser.add_argument('--length', action="store", dest='length', default=1)

    # Now, parse the command line arguments and store the
    # values in the `args` variable
    args = parser.parse_args()

    # W = Field of view width of our drone camera at a given altitude in meters
    # A = Area to be surveyed in m^2
    # S = Drone speed in metres per second
    # Ov = Overlap percentage; needed as we don't want to have any gaps between rows in the observation
    # L = Length (in metres) of the area being surveyed

    # Length of the field - W/2
    # Go back to the top left corner at the end of the survey
    # Calculate number of rows in the field
    # Lawnmower S-pattern
    # Dont worry about distance


    W = float(args.fov)
    A = float(args.area)
    S = float(args.speed)
    Ov = float(args.overlap)
    L = float(args.length)

    # Calculate the flight distance for the survey area in metres
    Ws = W * (1 - Ov) # calculate the effective swath width
    # Calculate the number of passes needed to survey the whole area
    passes = L / Ws
    # Calculate distance needed to return to home point
    RTH = 0.0
    if passes % 2 == 0:
        pass  # Even - RTH is a straight line
        RTH = L
    else:
        pass  # Odd - RTH is a diagonal
        width = A / L
        RTH = calculate_diagonal(L, width)

    # Calculate distance travelled
    # Add L to take into account the distance travelled between rows
    # Add RTH to include the distance for the drone to return to the home point once the survey is complete.
    D = A / Ws + L + RTH
    print(f"Total flight distance: {D} metres")

    # Calculate the flight time
    Ts = D / S
    print(f"Total flight time: {Ts} seconds.")
    Tm = Ts / 60
    print(f"Total flight time: {Tm} minutes.")
    Th = Ts / 3600
    print(f"Total flight time: {Th} hours.")
