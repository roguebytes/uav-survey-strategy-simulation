import argparse
import math

def calculate_diagonal(length, width):
    diagonal =
    return diagonal



def calculate_cost(fov, area, speed, overlap, length):
    # W = Field of view width of our drone camera at a given altitude in meters
    # A = Area to be surveyed in m^2
    # S = Drone speed in metres per second (NOT USED)
    # Ov = Overlap percentage; needed as we don't want to have any gaps between rows in the observation
    # L = Length (in metres) of the area being surveyed
    # RTH = Return To Home distance that is added to the calculated survey distance

    # Length of the field - W/2
    # Go back to the top left corner at the end of the survey
    # Calculate number of rows in the field
    # Lawnmower S-pattern
    # Dont worry about distance


    W = float(fov)
    A = float(area)
    # S = float(speed)
    Ov = float(overlap) # we don't want a percentage; need size of the object
    L = float(length)

    # Calculate the flight distance for the survey area in metres
    Ws = W * (1 - Ov) # calculate the effective swath width
    # Calculate the number of passes needed to survey the whole area
    passes = L / Ws
    # Calculate distance needed to return to home point

    if passes % 2 == 0:
        # Even - RTH is a straight line
        RTH = L
    else:
        # Odd - RTH is a diagonal
        width = A / L
        RTH = math.sqrt(L**2 + width**2)

    # Calculate distance travelled
    # Add L to take into account the distance travelled between rows
    # Add RTH to include the distance for the drone to return to the home point once the survey is complete.
    D = A / Ws + L + RTH
    print(f"Total flight distance: {D} metres")

    return D

    # # Calculate the flight time
    # Ts = D / S
    # print(f"Total flight time: {Ts} seconds.")
    # Tm = Ts / 60
    # print(f"Total flight time: {Tm} minutes.")
    # Th = Ts / 3600
    # print(f"Total flight time: {Th} hours.")
