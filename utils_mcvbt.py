from typing import Union

from IPython.core.ultratb import ListTB

import src.tequila as tq
from src.tequila.quantumchemistry import QuantumChemistryBase
from src.tequila.circuit import QCircuit
from src_vb.qvalence.utils import *
from src.tequila.hamiltonian import QubitHamiltonian
from typing import Tuple
import numpy as np
from src.tequila.simulators.simulator_api import INSTALLED_SIMULATORS, INSTALLED_BACKENDS, SUPPORTED_BACKENDS

def create_spa_circuit(graphs:list, mol: QuantumChemistryBase, deloc: str|None = None, use_openfermion = False) -> list[QCircuit]:

    circuits = []
    for i, graph in enumerate(graphs):
        U = mol.make_spa_ansatz(edges=graph)
        for edge in graph:
            U += Rot(idx=edge, mol=mol, label=i)
        U = add_delocalization(circuit=U,strategy=deloc, graph=graph, mol=mol)

        circuits.append(U)

    return circuits


def add_delocalization(circuit, strategy:str|None, graph: list, mol: QuantumChemistryBase) -> QCircuit:

    if strategy == "shift":
        shift_1 = []
        shift_2 = []
        for i,edge in enumerate(graph):
            shift_1.append(((edge[0]+1)%len(graph*2),(edge[1]+1)%len(graph*2)))
            if i != len(graph)-1:
                shift_2.append(((graph[i][0]), (graph[(i+1)% len(graph)][0])))
                shift_2.append(((graph[i][1]) , (graph[(i+1)% len(graph)][1])))
        for i,edge in enumerate(shift_1+shift_2):
            circuit += Rot(idx=edge,mol=mol, label=f"{strategy}{i}")

    elif strategy == "triangle":
        pass
    elif strategy is None:
        pass
    else:
        raise ValueError("Unknown strategy {}".format(strategy))

    return circuit


def run_mcvbt_optimization(circuits:list[QCircuit], H: QubitHamiltonian) -> Tuple[float,list[float]]:


    variables_preopt = {}
    energies = []
    for U in circuits:
        E = tq.ExpectationValue(U=U, H=H)

        result = tq.minimize(E, silent=True)
        variables_preopt = {**variables_preopt, **result.variables}
        energies.append(result.energy)
    exact_energy = min(energies)


    variables = {**variables_preopt}
    energies = []
    for i in range(2,len(circuits)+1):
        print(i)
        v, _ = gem_fast(circuits=circuits[:i], variables=variables, H=H)
        energies.append(v[0])

    for j in range(1, len(circuits)+1):
        for i in range(j, len(circuits)+1):
            if (i==1) and (j==1):
                pass
            else:
                print(i,j)
                v,_,variables = GNM(circuits=circuits[:i], variables=variables, H=H, M=j)
                energies.append(v[0])

    return exact_energy, energies


def project_to_plane(geometry):
    """
    Projects a set of 3D points onto the closest axis-aligned 2D plane (XY, YZ, or XZ)
    based on the smallest average perpendicular distance from all points to each plane.

    Parameters:
    points (np.ndarray): An (N, 3) array of 3D points.

    Returns:
    np.ndarray: (N, 2) array of 2D projected points.
    str: The name of the plane ('XY', 'YZ', or 'XZ') the points were projected onto.
    """
    points = parse_geom(geometry)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be an Nx3 array of 3D points.")

    # Average distances to each plane
    avg_dist_to_xy = np.mean(np.abs(points[:, 2]))  # Distance from XY plane (z-axis)
    avg_dist_to_yz = np.mean(np.abs(points[:, 0]))  # Distance from YZ plane (x-axis)
    avg_dist_to_xz = np.mean(np.abs(points[:, 1]))  # Distance from XZ plane (y-axis)

    # Choose the plane with the minimum average distance
    distances = {'XY': avg_dist_to_xy, 'YZ': avg_dist_to_yz, 'XZ': avg_dist_to_xz}
    best_plane = min(distances, key=distances.get)

    # Project onto the chosen plane
    if best_plane == 'XY':
        projected = points[:, [0, 1]]
    elif best_plane == 'YZ':
        projected = points[:, [1, 2]]
    else:  # 'XZ'
        projected = points[:, [0, 2]]

    return projected


def parse_geom(geometry):
    """
    Converts a string of atomic positions into a NumPy array of coordinates.

    Parameters:
    geometry (str): Multiline string, each line in format 'h x y z'

    Returns:
    np.ndarray: An (N, 3) NumPy array of float coordinates.
    """
    lines = geometry.strip().splitlines()
    coords = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 4:
            # Skip the label, parse the 3 float coordinates
            x, y, z = map(float, parts[1:])
            coords.append([x, y, z])

    return np.array(coords)


def project_points_to_circle(geometry, radius=None):
    """
    Projects 2D points onto a circle centered at their center of mass.

    Parameters:
    points_2d (np.ndarray): (N, 2) array of 2D points.
    radius (float, optional): Radius of the circle. If None, uses average distance from center.

    Returns:
    np.ndarray: (N, 2) array of points projected onto the circle.
    np.ndarray: (2,) array representing the center of the circle.
    float: Radius of the circle used.
    """
    points_2d = parse_geom(geometry)

    # Step 1: Center of mass
    center = points_2d.mean(axis=0)

    # Step 2: Vectors from center
    vectors = points_2d - center

    # Step 3: Distances and optional radius
    distances = np.linalg.norm(vectors, axis=1)
    if radius is None:
        radius = distances.mean()

    # Step 4: Normalize and scale to radius
    unit_vectors = vectors / distances[:, np.newaxis]
    projected = center + radius * unit_vectors

    return projected


def generate_rumor_diagrams(geometry, n_diagrams=1):
    """
    Generate up to n_diagrams of non-crossing matchings between points on a circle.

    Parameters:
    n_points (int): Total number of points on the circle (must be even).
    n_diagrams (int): Number of diagrams to return.

    Returns:
    List[List[Tuple[int, int]]]: Each list is a non-crossing diagram of index pairs.
    """
    n_points = project_points_to_circle(geometry)

    N = len(n_points)
    if N % 2 != 0:
        raise ValueError("Number of points must be even.")

    all_matchings = []

    def is_crossing(pair1, pair2):
        a, b = sorted(pair1)
        c, d = sorted(pair2)
        return (a < c < b < d) or (c < a < d < b)

    def is_valid(pair, current):
        return all(not is_crossing(pair, other) for other in current)

    def backtrack(available, current):
        if not available:
            all_matchings.append(current[:])
            return
        if len(all_matchings) >= n_diagrams:
            return
        i = available[0]
        for j in available[1:]:
            pair = (i, j)
            if is_valid(pair, current):
                next_avail = [x for x in available if x not in pair]
                backtrack(next_avail, current + [pair])

    indices = list(range(N))
    backtrack(indices, [])

    return all_matchings[:n_diagrams]
