from tequila.quantumchemistry import QuantumChemistryBase
from tequila.hamiltonian import QubitHamiltonian
from tequila import TequilaException


import openfermion
import ast
from collections import defaultdict

import warnings
import typing


def make_fermionic_hamiltonian(mol:QuantumChemistryBase, *args, **kwargs) -> QubitHamiltonian:
    """
    Parameters
    ----------
    molecule object in use
    Returns
    -------
    Qubit Hamiltonian in the Fermion-to-Qubit transformation defined in self.parameters
    """

    # warnings for backward comp
    if "active_indices" in kwargs:
        warnings.warn(
            "active space can't be changed in molecule. Will ignore active_orbitals passed to make_hamiltonian")

    of_molecule = mol.make_molecule()
    fop = of_molecule.get_molecular_hamiltonian()
    fop = openfermion.transforms.get_fermion_operator(fop)

    return fop


def make_excitation_generator_op(indices: typing.Iterable[typing.Tuple[int, int]], form: str = 'fermionic',
                                 remove_constant_term: bool = True, mol:QuantumChemistryBase = None) \
                                 -> openfermion.FermionOperator:
    """
    Notes
    ----------
    Creates the transformed hermitian generator of UCC type unitaries:
          M(a^\dagger_{a_0} a_{i_0} a^\dagger{a_1}a_{i_1} ... - h.c.)
          where the qubit map M depends is self.transformation

    Parameters
    ----------
    indices : typing.Iterable[typing.Tuple[int, int]] :
        List of tuples [(a_0, i_0), (a_1, i_1), ... ] - recommended format, in spin-orbital notation (alpha odd numbers, beta even numbers)
        can also be given as one big list: [a_0, i_0, a_1, i_1 ...]
    form : str : (Default value None):
        Manipulate the generator to involution or projector
        set form='involution' or 'projector'
        the default is no manipulation which gives the standard fermionic excitation operator back
    remove_constant_term: bool: (Default value True):
        by default the constant term in the qubit operator is removed since it has no effect on the unitary it generates
        if the unitary is controlled this might not be true!
    Returns
    -------
    type
        1j*Transformed qubit excitation operator, depends on self.transformation
    """


    # check indices and convert to list of tuples if necessary
    if len(indices) == 0:
        raise TequilaException("make_excitation_operator: no indices given")
    elif not isinstance(indices[0], typing.Iterable):
        if len(indices) % 2 != 0:
            raise TequilaException("make_excitation_generator: unexpected input format of indices\n"
                                   "use list of tuples as [(a_0, i_0),(a_1, i_1) ...]\n"
                                   "or list as [a_0, i_0, a_1, i_1, ... ]\n"
                                   "you gave: {}".format(indices))
        converted = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
    else:
        converted = indices

    # convert everything to native python int
    # otherwise openfermion will complain
    converted = [(int(pair[0]), int(pair[1])) for pair in converted]

    # convert to openfermion input format
    ofi = []
    dag = []
    for pair in converted:
        assert (len(pair) == 2)
        ofi += [(int(pair[0]), 1),
                (int(pair[1]), 0)]  # openfermion does not take other types of integers like numpy.int64
        dag += [(int(pair[0]), 0), (int(pair[1]), 1)]

    op = openfermion.FermionOperator(tuple(ofi), 1.j)  # 1j makes it hermitian
    op += openfermion.FermionOperator(tuple(reversed(dag)), -1.j)

    if isinstance(form, str) and form.lower() != 'fermionic':
        # indices for all the Na operators
        Na = [x for pair in converted for x in [(pair[0], 1), (pair[0], 0)]]
        # indices for all the Ma operators (Ma = 1 - Na)
        Ma = [x for pair in converted for x in [(pair[0], 0), (pair[0], 1)]]
        # indices for all the Ni operators
        Ni = [x for pair in converted for x in [(pair[1], 1), (pair[1], 0)]]
        # indices for all the Mi operators
        Mi = [x for pair in converted for x in [(pair[1], 0), (pair[1], 1)]]

        # can gaussianize as projector or as involution (last is default)
        if form.lower() == "p+":
            op *= 0.5
            op += openfermion.FermionOperator(Na + Mi, 0.5)
            op += openfermion.FermionOperator(Ni + Ma, 0.5)
        elif form.lower() == "p-":
            op *= 0.5
            op += openfermion.FermionOperator(Na + Mi, -0.5)
            op += openfermion.FermionOperator(Ni + Ma, -0.5)

        elif form.lower() == "g+":
            op += openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, -1.0)
            op += openfermion.FermionOperator(Ni + Ma, -1.0)
        elif form.lower() == "g-":
            op += openfermion.FermionOperator([], -1.0)  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, 1.0)
            op += openfermion.FermionOperator(Ni + Ma, 1.0)
        elif form.lower() == "p0":
            # P0: we only construct P0 and don't keep the original generator
            op = openfermion.FermionOperator([], 1.0)  # Just for clarity will be subtracted anyway
            op += openfermion.FermionOperator(Na + Mi, -1.0)
            op += openfermion.FermionOperator(Ni + Ma, -1.0)
        else:
            raise TequilaException(
                "Unknown generator form {}, supported are G, P+, P-, G+, G- and P0".format(form))



    return op


def create_fermioinc_generators(mol: QuantumChemistryBase, instructions = None, variables = None):

    if (instructions is None) and (variables is None):
        raise ValueError("Cannot create fermionic generators without instructions or variables")
    #todo check for variables format and instruction format
    generators={}

    if variables is None:
        for angle_idx, fermionic_group in enumerate(instructions):
            if (len(fermionic_group[0]) != len(x) for x in fermionic_group):
                raise TequilaException("Trying to assign same angles to different generators")
            for fermionic_circuit in fermionic_group:
                if angle_idx in generators:
                    generators[angle_idx] = generators[angle_idx] + make_excitation_generator_op(indices=fermionic_circuit, mol=mol)
                else:
                    generators[angle_idx] = make_excitation_generator_op(indices=fermionic_circuit, mol=mol)
    if instructions is None:
        indeces = extract_indices_from_variables(variables)
        print(indeces)


def extract_indices_from_variables(vairables):
    try:
        entries = ast.literal_eval(str(list(vairables.keys())))
    except (ValueError, SyntaxError) as e:
        raise ValueError("Invalid input string. Must be a valid list of tuples.") from e

    grouped = defaultdict(list)

    for item in entries:
        if isinstance(item, tuple):
            if len(item) == 3 and item[1] == 'D':
                key = item[2]  # e.g. 'G0'
                grouped[key].append(item[0])
            elif len(item) == 4 and item[0] == 'R':
                key = f"R{item[3]}"  # e.g. 0 -> 'R0'
                grouped[key].append((item[1], item[2]))
            elif len(item) == 4 and item[0] == 'C':
                key = f"C{item[3]}"
                grouped[key].append((item[1], item[2]))
            else:
                raise NotImplementedError(f"Unrecognized variable type {item}")

    return dict(grouped)



