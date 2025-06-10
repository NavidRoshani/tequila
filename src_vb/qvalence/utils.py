import scipy
import qulacs
from openfermion import FermionOperator
from sympy.physics.units import action

import tequila as tq
from tequila.hamiltonian import QubitHamiltonian
from tequila import TequilaException
from tequila.quantumchemistry import QuantumChemistryBase
import openfermion
import fqe
import numpy
import scipy
import typing

import warnings

from tequila.quantumchemistry import QuantumChemistryBase

warnings.filterwarnings("ignore", category=tq.TequilaWarning)

"""
Convenience Implementations and Structures to speed up simulation times
"""

def Rot(idx, mol, label=None, s=1.e-4):
    """
    Convenience implementation of Rotation gates as described in the paper
    See also ArXiv:2207.12421 Eq.(6)
    In tequila version >= 1.8.4 this is equivalent to mol.UR
    """
    angle=tq.Variable((tuple(idx),label))
    tmp = mol.make_excitation_gate(indices=[(2*idx[0],2*idx[1])], angle=(angle+s)*numpy.pi)
    tmp+= mol.make_excitation_gate(indices=[(2*idx[0]+1,2*idx[1]+1)], angle=(angle+s)*numpy.pi)
    return tmp

def Corr(i,j, label=None):
    """
    Convenience initialization of paired two-body correlator
    See ArXiv:2207.12421 Eq.(22)
    In tequila version >= 1.8.4 this is equivalent to mol.UC
    """
    return tq.gates.QubitExcitation(target=[2*i,2*j,2*i+1,2*j+1], angle=(i,j,label))

def make_excitation_generator_op(indices: typing.Iterable[typing.Tuple[int, int]], form: str = None, remove_constant_term: bool = True) -> FermionOperator:
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


class BraKetQulacs:
    """
    Hacky Replacement of tq.BraKet
    Speedup of underlying simulation
    Limitations: Only Qulacs can be backend, not differentiable at the moment
    """
    def __init__(self, bra,ket,H):
        # translate tq -> qulacs
        E1 = tq.compile(tq.ExpectationValue(U=bra,H=H), backend="qulacs")
        E2 = tq.compile(tq.ExpectationValue(U=ket,H=H), backend="qulacs")
        # extract qulacs structures
        self.bra = E1.get_expectationvalues()[0]._U
        self.ket = E2.get_expectationvalues()[0]._U
        self.H = E1.get_expectationvalues()[0]._H[0]
        self.n_qubits = ket.n_qubits
        self.is_overlap = H.n_qubits == 0
    def __call__(self, variables, *args, **kwargs):
        # call qulacs structures
        # similar as tequila would, but exploits storing wavefunctions
        self.ket.update_variables(variables)
        self.bra.update_variables(variables)
        state_bra = self.bra.initialize_state(self.n_qubits)
        state_ket = self.ket.initialize_state(self.n_qubits)
        self.bra.circuit.update_quantum_state(state_bra)
        self.ket.circuit.update_quantum_state(state_ket)
        if self.is_overlap:
            vector1 = state_bra.get_vector()
            vector2 = state_ket.get_vector()
            result = vector1.conj().T.dot(vector2)
        else:
            result = self.H.get_transition_amplitude(state_bra, state_ket)

        result=result.real
        return result

class BraKetOpenfermion():

    def __init__(self, bra, ket, H: QubitHamiltonian, H_Fermion):

        # E1 = tq.compile(tq.ExpectationValue(U=bra, H=H), backend="cirq")
        # E2 = tq.compile(tq.ExpectationValue(U=ket, H=H), backend="cirq")
        # self.bra = E1.get_expectationvalues()[0]._U
        # self.ket = E2.get_expectationvalues()[0]._U
        # self.H = H.to_openfermion()
        # self.n_qubits = ket.n_qubits
        # self.is_overlap = H.n_qubits == 0
        E1 = tq.compile(tq.ExpectationValue(U=bra, H=H), backend="qulacs")
        E2 = tq.compile(tq.ExpectationValue(U=ket, H=H), backend="qulacs")

        # extract qulacs structures
        self.bra = E1.get_expectationvalues()[0]._U
        self.ket = E2.get_expectationvalues()[0]._U
        self.H = H_Fermion
        self.n_qubits = ket.n_qubits
        self.is_overlap = H.n_qubits == 0
    def __call__(self, variables, *args, **kwargs):
        # call qulacs structures
        # similar as tequila would, but exploits storing wavefunctions
        self.ket.update_variables(variables)
        self.bra.update_variables(variables)
        # state_bra = self.bra.initialize_qubit(self.n_qubits)
        # state_ket = self.ket.initialize_qubit(self.n_qubits)
        #self.bra.circuit.update_quantum_state(state_bra)
        #self.ket.circuit.update_quantum_state(state_ket)
        state_bra = self.bra.initialize_state(self.n_qubits)
        state_ket = self.ket.initialize_state(self.n_qubits)
        self.bra.circuit.update_quantum_state(state_bra)
        self.ket.circuit.update_quantum_state(state_ket)

        wfn = fqe.from_cirq(state_ket.get_vector(), thresh=0.001)
        brawfn = fqe.from_cirq(state_bra.get_vector(), thresh=0.001)

        # H_fqe = FermionOperator()
        # print(self.H)
        # for term in self.H:
        #     action = str(term)[str(term).find("[")+1:-1]
        #     if action == "":
        #         action=None
        #     else:
        #         pass
        #
        #     print(action,term.constant)
        #     H_fqe += FermionOperator(action,term.constant)
        h_op = fqe.get_hamiltonian_from_openfermion(self.H)
        if self.is_overlap:
            vector1 = state_bra.get_vector()
            vector2 = state_ket.get_vector()
            result = vector1.conj().T.dot(vector2)
        else:
            result = fqe.expectationValue(wfn=wfn, ops=h_op, brawfn=brawfn)

        result=result.real

        return result


def gem_fast(circuits, H, H_Fermion, solver, variables, silent=True):
    """
    Fast implementation of tq.apps.gem 
    works only with qulacs backend
    not differentiable
    """
    #solver="qulacs"
    E = [tq.simulate(tq.ExpectationValue(H=H, U=U), variables=variables, silent=silent) for U in circuits]
    SS = numpy.eye(len(circuits))
    EE = numpy.eye(len(circuits))
    for i in range(len(circuits)):
        EE[i,i] = E[i]
        for j in range(i+1,len(circuits)):
            if solver == "qulacs":
                f=BraKetQulacs(circuits[i], circuits[j], H)
                ff=BraKetQulacs(circuits[i], circuits[j], H=tq.paulis.I())
            elif solver == "openfermion":
                f = BraKetOpenfermion(circuits[i], circuits[j], H=H, H_Fermion=H_Fermion)
                ff = BraKetOpenfermion(circuits[i], circuits[j], H=tq.paulis.I(), H_Fermion=FermionOperator(''))
            else:
                raise ValueError("Unknown solver {}".format(solver))

            EE[i,j] = f(variables)
            EE[j,i] = EE[i,j]
            SS[i,j] = ff(variables)
            SS[j,i] = SS[i,j]
    print("EE",EE,"SS",SS)
    v,vv = scipy.linalg.eigh(EE,SS)

    return v,vv

class BigExpVal:
    """
    Convenience to initialize an expectation value as described in Eq.(7) of the paper with the Qulacs only structure
    """

    def __init__(self, circuits, H,H_Fermion, coeffs, solver):
        n = len(circuits)
        self.n = n
        E = [tq.compile(tq.ExpectationValue(H=H, U=U)) for U in circuits]
        SS = []
        EE = []
        for i in range(n):
            tmp1 = []
            tmp2 = []
            for j in range(i):
                if solver == "qulacs":
                    xEE = BraKetQulacs(circuits[i], circuits[j], H=H)
                    xSS = BraKetQulacs(circuits[i], circuits[j], H=tq.paulis.I())
                elif solver == "openfermion":
                    xEE = BraKetOpenfermion(circuits[i], circuits[j], H=H, H_Fermion=H_Fermion)
                    xSS = BraKetOpenfermion(circuits[i], circuits[j], H=H, H_Fermion=FermionOperator(''))
                else:
                    raise ValueError("Unknown solver {}".format(solver))
                tmp1.append(xEE)
                tmp2.append(xSS)
            tmp1.append(E[i])
            tmp2.append(1.0)
            EE.append(tmp1)
            SS.append(tmp2)
        self.SS = SS
        self.EE = EE
        self.coeffs = coeffs
        variables={}
        for U in circuits:
            variables = {**variables, **{x:0.0 for x in U.extract_variables()}}
        for c in coeffs:
            variables = {**variables, **{x:0.0 for x in  c.extract_variables()}}
        self.variables=list(variables.keys())

    def __call__(self, x,*args, **kwargs):
        n = self.n
        assert len(x) <= len(self.variables)
        values={self.variables[i]:x[i] for i in range(len(self.variables))}
        c = [self.coeffs[i](values) for i in range(n)]
        f = 0.0
        s = 0.0
        for i in range(n):
            f+=self.EE[i][i](values)*c[i]**2
            s+=c[i]**2
            for j in range(i):
               f+=2.0*self.EE[i][j](values)*c[i]*c[j]
               s+=2.0*self.SS[i][j](values)*c[i]*c[j]

        f=f.real
        s=s.real
        if s>0.0:
            r=f/s
        else:
            # failsave for optimizer, only happens with bad variable initialization
            r=1e5
        return r

def GNM(circuits, H, H_Fermion, variables, solver="openfermion", silent=True, maxiter=10, M=None):
    """
    the G(M,N) method from the paper, N is implicitly given over the number of circuits
    """
    circs = [x for x in circuits]
    N = len(circs)
    if M is None:
        M = len(circs)
    
    # fix variables for circuits that will not be part of the optimization
    for i in range(M, N):
        U = circs[i]
        U = U.map_variables(variables)
        circs[i] = U

    vkeys = []
    for U in circs:
        vkeys+=U.extract_variables()
    
    variables = {**{k:0.0 for k in vkeys if k not in variables}, **variables}
   
    v,vv = gem_fast(circuits=circs,H=H, H_Fermion=H_Fermion,variables=variables, solver=solver)
    
    x0 = {k:variables[k] for k in vkeys}

    coeffs = []
    for i in range(len(circs)):
        c=tq.Variable(("c",i))
        coeffs.append(c)
        x0[c] = vv[i,0]
        vkeys.append(c)
        
    energy = 1.0
    def callback(x):
        energy=f(x)
        if not silent:
            print("current energy: {:+2.4f}".format(energy))
    
    f = BigExpVal(circuits=circs, H=H,H_Fermion=H_Fermion, coeffs=coeffs, solver=solver)

    for i in range(maxiter):
        result = scipy.optimize.minimize(f, x0=list(x0.values()), jac="2-point",
                                         method="bfgs", options={"finite_diff_rel_step":1.e-5, "disp":False},
                                         callback=callback)

        x0 = {vkeys[i]:result.x[i] for i in range(len(result.x))}
        v,vv = gem_fast(circuits=circs,H=H,H_Fermion=H_Fermion,variables=x0,solver=solver)
        for i in range(len(coeffs)):
            x0[coeffs[i]]=vv[i,0]
        if numpy.isclose(energy, v[0], atol=1.e-4):
            print("not converged")
            print(energy)
            print(v[0])
            energy = v[0]
        else:
            energy = v[0]
            break
    
    for k in vkeys:
        variables[k] = x0[k]
    return v,vv,variables


def make_fermionic_Ham(mol:QuantumChemistryBase, *args, **kwargs) -> QubitHamiltonian:
    """
    Parameters
    ----------
    occupied_indices: will be auto-assigned according to specified active space. Can be overridden by passing specific lists (same as in open fermion)
    active_indices: will be auto-assigned according to specified active space. Can be overridden by passing specific lists (same as in open fermion)

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