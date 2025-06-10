
import src.tequila as tq
from src.tequila.quantumchemistry import QuantumChemistryBase

from src_vb.qvalence.utils import *
from utils_mcvbt import create_spa_circuit, run_mcvbt_optimization, generate_rumor_diagrams, create_ferionic_generators
import openfermion


def run_optimization(mol: QuantumChemistryBase, *args, **kwargs):

    geometry = mol.parameters.geometry
    rumor_diagrams = generate_rumor_diagrams(geometry,n_diagrams=3)

    mol = tq.Molecule(geometry=geometry, basis_set="sto-6g")
    mol = mol.use_native_orbitals()
    fci = mol.compute_energy("fci")

    edges1 = [(0, 1), (2, 3)]
    edges2 = [(0, 2), (1, 3)]
    edges3 = [(0, 3), (1, 2)]
    graphs = [edges1, edges2, edges3]

    H = mol.make_hamiltonian()
    H_fermion = make_fermionic_Ham(mol=mol)

    circuits = create_spa_circuit(graphs=graphs,mol=mol, deloc=None)
    # generatros = create_ferionic_generators(graphs=graphs)



    # pre-optimize the circuits
    exact_energies, energies = run_mcvbt_optimization(circuits=circuits,H=H, H_Fermion=H_fermion)

    for energy in energies:
        error = abs(energy-fci)
        print("G(,): {:+2.5f}".format( error))

    energy_output = {"energy": energies[-1]}


    return energy_output

