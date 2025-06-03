
import src.tequila as tq
from src.tequila.quantumchemistry import QuantumChemistryBase

from src_vb.qvalence.utils import *





def run_optimization(mol: QuantumChemistryBase, *args, **kwargs):
    data1 = {}
    data2 = {}
    geomeotry = mol.parameters.geometry
    edges1 = [(0, 1), (2, 3)]
    edges2 = [(0, 2), (1, 3)]
    edges3 = [(0, 3), (1, 2)]
    graphs = [edges1, edges2, edges3]

    H = mol.make_hamiltonian()

    circuits = []


    for i, edges in enumerate(graphs):
        U = mol.make_ansatz(name="SPA", edges=edges, label="G{}".format(i))
        UR = tq.QCircuit()
        for e in edges:
            UR += Rot(e, mol, i)
        circuits.append(U + UR)

    # pre-optimize the circuits
    variables_preopt = {}
    energies = []
    for U in circuits:
        E = tq.ExpectationValue(U=U, H=H)
        result = tq.minimize(E, silent=True)
        variables_preopt = {**variables_preopt, **result.variables}
        energies.append(result.energy)


    best = min(energies)

    data1[(1, 0)] = best
    variables = {**variables_preopt}
    # compute static energies with the pre-optimized basis
    v, vv = gem_fast(circuits=circuits[:2], variables=variables, H=H)
    data1[(2, 0)] = v[0]
    v, vv = gem_fast(circuits=circuits[:3], variables=variables, H=H)
    data1[(3, 0)] = v[0]

    # relax circuit parameters
    v, vv, variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True, M=1)
    data1[(2, 1)] = v[0]

    v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=1)
    data1[(3, 1)] = v[0]

    v, vv, variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True)
    data1[(2, 2)] = v[0]

    v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=2)
    data1[(3, 2)] = v[0]

    v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True)
    data1[(3, 3)] = v[0]

    # add more freedeom in orbital rotations
    # this is G(N,M)+UR
    for i in range(len(circuits)):
        e0 = graphs[i][0]
        e1 = graphs[i][1]
        UR = Rot((e0[0], e1[0]), mol, i)
        UR += Rot((e0[1], e1[1]), mol, i)
        UR += Rot((e0[0], e1[1]), mol, i)
        circuits[i] += UR

    # reset variables to pre-opt
    variables = variables_preopt

    # relax circuit parameters
    v, vv, variables = GNM(circuits=circuits[:1], variables=variables, H=H, silent=True, M=1)
    data2[(1, 0)] = v[0]
    v, vv, variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True, M=1)
    data2[(2, 1)] = v[0]
    v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=1)
    data2[(3, 1)] = v[0]

    v, vv, variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True)
    data2[(2, 2)] = v[0]

    v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=2)
    data2[(3, 2)] = v[0]
    v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True)
    data2[(3, 3)] = v[0]

    SPA = mol.make_ansatz(name="SPA", edges=edges)

    energy = {"energy": v[0]}


    return energy

