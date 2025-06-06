from src.tequila.utils import BitString, BitNumbering, BitStringLSB, initialize_bitstring, TequilaException, TequilaWarning
from src.tequila.circuit import gates, QCircuit, NoiseModel, compile_circuit, CircuitCompiler
from src.tequila.hamiltonian import paulis, QubitHamiltonian, PauliString
from src.tequila.objective import Objective, VectorObjective,\
    ExpectationValue, Variable, assign_variable, format_variable_dictionary,\
    vectorize
from src.tequila.objective import QTensor
from src.tequila.objective.braket import BraKet, make_transition, make_overlap, Overlap, Fidelity

# backward compatibility
braket = BraKet

from src.tequila.optimizers import INSTALLED_OPTIMIZERS, show_available_optimizers
from src.tequila.optimizers import minimize, minimize_scipy, minimize_gd, optimizer_scipy

from src.tequila.simulators.simulator_api import simulate, compile, compile_to_function, draw, pick_backend, \
    INSTALLED_SAMPLERS, \
    INSTALLED_SIMULATORS, SUPPORTED_BACKENDS, INSTALLED_BACKENDS, show_available_simulators
from src.tequila.wavefunction import QubitWaveFunction
from src.tequila.circuit.qasm import export_open_qasm, import_open_qasm, import_open_qasm_from_file
from src.tequila.circuit.pyzx import convert_to_pyzx, convert_from_pyzx
import src.tequila.quantumchemistry as chemistry # shortcut
from src.tequila.quantumchemistry import Molecule, MoleculeFromOpenFermion, MoleculeFromTequila

# make sure to use the jax/autograd numpy for objectives
from src.tequila.circuit.gradient import grad
from src.tequila.autograd_imports import numpy, jax, __AUTOGRAD__BACKEND__

# import tools
from src.tequila.tools.random_generators import make_random_circuit, make_random_hamiltonian

# get rid of the jax GPU/CPU warnings
import warnings

warnings.filterwarnings("ignore", module="jax")
warnings.filterwarnings("ignore", module="absl")
warnings.filterwarnings("default", category=TequilaWarning)

# load applications and helpers
from src.tequila.apps import adapt

from .version import __version__, __author__
