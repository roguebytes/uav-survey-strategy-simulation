#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:01:21 2024

Last updated Thu 19th Sep

@author: frederic
f.maire@gmail.com



Some standard definitions:

Precision (P): is the ratio of true positives to 
the sum of true positives and false positives: 
    P = TP / (TP+FP) 

Recall (R): (also called sensitivity or true positive rate) is the ratio of 
true positives to the sum of true positives and false negatives: 
    R = TP / (TP+FN)

Key variables
D : mine density, the probability that a cell has a mine
P : precision of the detector
R : recall of the detector

Generation of a mine field with density D
    Generate the minefield as a grid. For each cell of the grid, put a mine 
    with a probability D
 
Generation of the flag field
    The detector considers each cell in turn. If the detector predicts a mine, 
    it flags this cell to verify it on a flight at lower altitude.
 
In order to simulate which cell the detector would flag, we have to compute
two probabilities, 
    prob( flag | not mine )
    prob( flag | mine )

prob( flag | mine) = TP / (TP+FN)     that’s the recall  R

TP, FP and FN  can be estimated with drone flights

 
Note that TP+FN  is also the number of mines in the field. 
A number that we have once we have generated a mine field.

Let express TP, FP, TN, and FN in terms of precision and recall, 
TP = R x (TP+FN)
FP = TP x (1/P -1)
FN = TP x (1/R -1)
 
We have  prob( flag | not mine) = FP/(FP+TN)
 
Let’s C be the total number of cells. We have  C = “mine cells” + “not mine cells”
We have C = (TP+FN) + (FP+TN)   
Therefore   prob( flag | not mine ) = FP/(C-(TP+FN))
 
In conclusion, given R, P and C,  we can compute
prob( flag | mine) and prob( flag | not mine)
and therefore generate the flag grid.
 

For the the traveling salesman problem, we rely on
https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.traveling_salesman.traveling_salesman_problem.html#networkx.algorithms.approximation.traveling_salesman.traveling_salesman_problem


"""

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt

from networkx.algorithms.approximation import traveling_salesman_problem


def gen_minefield(density,  shape):
    '''
    Generate a minefield

    The expression
        len(M.nonzero()[0])
    counts the number of mines in the generated grid
    
    Parameters
    ----------
    density : float
        probability that a cell contains a mine.
    shape : tuple (nrows, ncols)
        number of rows and number of columns of the grid representing the minefield.

    Returns
    -------
    a 2D Boolean numpy array M where 
        M[r,c] == False if no mine in cell at loc (r,c) in the grid
        M[r,c] == True if there is a mine in cell (r,c)

    '''
    return np.random.rand(*shape) <= density
    
    

def gen_flag_field(M, TP, FP, FN ):
    """
    Two probabilities to compute, 
    pf_nm = prob( flag | not mine)
    pf_m = prob( flag | mine)

    prob( flag | mine) = TP / (TP+FN)     that’s the recall  R
     
    TP+FN  is also the number of mines in the field. A number that we have once 
    we have generated a minefield.

    BELOW THE PARAMETERS ARE CALCULATED THAT ARE USED TO GENERATE THE FLAG FIELD
    Let express TP, FP, TN, and FN in terms of precision and recall, 
    TP = R x Number of mines in the field (TP+FN)
    FP = TP x (1/P -1)
    FN = TP x (1/R -1)

    
     
    We have  prob( flag | not mine ) = FP/(FP+TN)
     
    Let’s C be the total number of cells. We have  C = “mine cells” + “not mine cells”
    We have C = (TP+FN) + (FP+TN)   
    Therefore   prob( flag | not mine) = FP/(C-(TP+FN))


    Parameters
    ----------
    M : TYPE
        DESCRIPTION.
    TP : TYPE
        DESCRIPTION.
    FP : TYPE
        DESCRIPTION.
    FN : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    pf_m =  TP / (TP+FN)  
    C = int(np.prod(M.shape)) # number of cells in the grid
    pf_nm =  FP/(C-(len(M.nonzero()[0])))
    
    Z = np.random.rand(*M.shape) # random array same size as the grid
    # We put a flag with probability pf_m if the cell has a mine
    # We put a flag with probability pf_nm if the cell has no mine
    F = ((Z <= pf_m) & M) | ((Z <= pf_nm) & ~M)  # flag boolean array
    
    return  F
    
    
def try_gen_minefield(r=7, c=7, d=0.2):
    '''
        Generate a minefield of r=4 rows and c=6 columns with a mine density d=0.2
    '''
    M = gen_minefield(d, shape=(r,c))
    return M
    

def try_gen_flagfield(M, TP_rate = 0.6, FP_rate = 0.1, FN_rate = 0.1):
    '''
        Generate a flag field from the mine field M
    '''
    # M = gen_minefield(0.2, shape=(8,12))
    # print("\nMine field")
    # print(M)
    
    # TP_rate = 0.6
    # FP_rate = 0.1
    # FN_rate = 0.1
    # therefore, TN_rate is 1-(TP_rate+FP_rate+FN_rate) = 1 - (0.6+0.1+0.1)

    C = int(np.prod(M.shape)) # number of cells in the grid
    TP, FP, FN = TP_rate*C, FP_rate*C, FN_rate*C 
    F = gen_flag_field(M, TP, FP, FN)
    
    # print("\nFlag field")
    # print(F)
    
    return F


def get_tour(F, cell_side = 1):
    '''
    Return a tour of the flags
    
    Origin top left (row 0, column 0).
    Let a = cell_side/2
    The column coord is x, the row coord is y
    The cell F[r=1,c=3] centre is at position
    y = a*(1+2*r) and x = a*(1+2*r)

    Parameters
    ----------
    F : TYPE
        DESCRIPTION.
    cell_side : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    

    '''
    a = cell_side/2
    R, C = F.nonzero()
    Y = a*(1+2*R)
    X = a*(1+2*C)
    
    # Create a graph from the coordinates
    G = nx.Graph()
    
    # Add nodes with coordinates as attributes
    for i in range(len(X)):
        G.add_node(i, pos=(X[i], Y[i]))
    
    # Add edges with Euclidean distances between nodes as weights
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            distance = np.linalg.norm([X[i] - X[j], Y[i] - Y[j]])  # Euclidean distance
            G.add_edge(i, j, weight=distance)
    
    # Use networkx's approximation for the Traveling Salesman Problem
    tsp_path = traveling_salesman_problem(G, weight='weight')
    
    return tsp_path, G
    
    
def plot_TSP(tsp_path, G):
    # Plotting the graph
    pos = nx.get_node_attributes(G, 'pos')  # Get positions of nodes
    
    # Draw the nodes
    nx.draw(G, pos, node_color='lightblue', with_labels=True, node_size=500)
    
    # Highlight the TSP tour by drawing it
    tsp_edges = [(tsp_path[i], tsp_path[i + 1]) for i in range(len(tsp_path) - 1)] + [(tsp_path[-1], tsp_path[0])]
    nx.draw_networkx_edges(G, pos, edgelist=tsp_edges, edge_color='red', width=2)
    
    # Show the plot
    plt.title("Graph with TSP Tour")
    plt.show()
    

def compute_tour_cost(G, tsp_path):
    # Compute the cost of the TSP tour
    # tsp_path is closed loop
    cost = 0
    # Add the cost for each consecutive pair of nodes in the tsp_path
    for i in range(len(tsp_path) - 1):
        cost += G[tsp_path[i]][tsp_path[i + 1]]['weight']
    return cost



if __name__ == "__main__":

    M = try_gen_minefield()
    print("\nMine field")
    print(M)
    
    F = try_gen_flagfield(M, TP_rate = 0.73, FP_rate = 0.03, FN_rate = 0.27)
    print("\nFlag field")
    print(F)
    
    tsp_path, G = get_tour(F, cell_side=1)
    print("Shortest TSP Path:", tsp_path)

    # Compute and print the tour cost
    tour_cost = compute_tour_cost(G, tsp_path)
    print("Cost of the TSP Tour:", tour_cost)

    plot_TSP(tsp_path, G)
    
    