import tequila as tq
from src_sunsrise.utils_fqe import make_fermionic_hamiltonian, create_fermioinc_generators
from tequila.objective.objective import Variable

geometry1 = "H 1.5 0.0 0.0\nH 0.0 0.0 0.0\nH 1.5 0.0 1.5\nH 0.0 0.0 1.5"
mol = tq.Molecule(geometry=geometry1, basis_set="sto-6g")
mol = mol.use_native_orbitals()
H = mol.make_hamiltonian()
H_of = make_fermionic_hamiltonian(mol=mol)

fci = mol.compute_energy("fci")

# define the graphs
edges1 = [(0,1),(2,3)]
edges2 = [(0,2),(1,3)]
edges3 = [(0,3),(1,2)]
graphs = [edges1]

circuits = []
for i,edges in enumerate(graphs):
    U = mol.make_ansatz(name="SPA", edges=edges, label="G{}".format(i))
    for j, e in enumerate(edges):
        U += mol.UR(e[0], e[1], label = j)
    circuits.append(U)

variables_preopt = {}
energies = []
for U in circuits:
    E = tq.ExpectationValue(U=U, H=H)
    result = tq.minimize(E, silent=True)
    variables_preopt = {**variables_preopt, **result.variables}
    energies.append(result.energy)
l = list(variables_preopt.keys())


print(variables_preopt)

create_fermioinc_generators(mol,variables=variables_preopt)
