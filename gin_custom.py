
import src.tequila as tq
from src.tequila.quantumchemistry import QuantumChemistryBase

from src_vb.qvalence.utils import *
from utils_mcvbt import create_spa_circuit, run_mcvbt_optimization


def run_optimization(mol: QuantumChemistryBase, *args, **kwargs):
    data1 = {}
    data2 = {}
    geomeotry = mol.parameters.geometry

    geometry1 = "H 1.5 0.0 0.0\nH 0.0 0.0 0.0\nH 1.5 0.0 1.5\nH 0.0 0.0 1.5"
    # linear
    geometry2 = "H 0.0 0.0 0.0\nH 0.0 0.0 1.5\nH 0.0 0.0 3.0\nH 0.0 0.0 4.5"

    mol = tq.Molecule(geometry=geometry1, basis_set="sto-6g")
    # replace with "orthonormalize_basis_orbitals()" for tq.version < 1.8.4
    mol = mol.use_native_orbitals()
    fci = mol.compute_energy("fci")

    edges1 = [(0, 1), (2, 3)]
    edges2 = [(0, 2), (1, 3)]
    edges3 = [(0, 3), (1, 2)]
    graphs = [edges1, edges2, edges3]

    H = mol.make_hamiltonian()

    circuits = create_spa_circuit(graphs=graphs,mol=mol, deloc=None)


    # pre-optimize the circuits
    exact_energies, energies = run_mcvbt_optimization(circuits=circuits,H=H)

    for energy in energies:
        error = abs(energy-fci)
        print("G(,): {:+2.5f}".format( error))
    exit()
    # add more freedeom in orbital rotations
    # this is G(N,M)+UR
    # for i in range(len(circuits)):
    #     e0 = graphs[i][0]
    #     e1 = graphs[i][1]
    #     UR = Rot((e0[0], e1[0]), mol, i)
    #     UR += Rot((e0[1], e1[1]), mol, i)
    #     UR += Rot((e0[0], e1[1]), mol, i)
    #     circuits[i] += UR
    #
    # # reset variables to pre-opt
    # variables = variables_preopt
    #
    # # relax circuit parameters
    # v, vv, variables = GNM(circuits=circuits[:1], variables=variables, H=H, silent=True, M=1)
    # data2[(1, 0)] = v[0]
    # v, vv, variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True, M=1)
    # data2[(2, 1)] = v[0]
    # v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=1)
    # data2[(3, 1)] = v[0]
    #
    # v, vv, variables = GNM(circuits=circuits[:2], variables=variables, H=H, silent=True)
    # data2[(2, 2)] = v[0]
    #
    # v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True, M=2)
    # data2[(3, 2)] = v[0]
    # v, vv, variables = GNM(circuits=circuits[:3], variables=variables, H=H, silent=True)
    # data2[(3, 3)] = v[0]



    energy = {"energy": v[0]}


    return energies[-1]

