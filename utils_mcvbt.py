from typing import Union

from IPython.core.ultratb import ListTB

import src.tequila as tq
from src.tequila.quantumchemistry import QuantumChemistryBase
from src.tequila.circuit import QCircuit
from src_vb.qvalence.utils import *
from src.tequila.hamiltonian import QubitHamiltonian
from typing import Tuple

def create_spa_circuit(graphs:list, mol: QuantumChemistryBase, deloc: str|None = None) -> list[QCircuit]:

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