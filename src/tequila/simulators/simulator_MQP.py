from tequila.simulators.simulator_qiskit import BackendCircuitQiskit, BackendExpectationValueQiskit, TequilaQiskitException
from mqp.qiskit_provider import MQPProvider, MQPBackend
from qiskit_aqt_provider import AQTProvider
from qiskit_aqt_provider.aqt_resource import AQTResource
from qiskit_aqt_provider.primitives import AQTEstimator, AQTSampler
import qiskit_aqt_provider
from tequila.wavefunction.qubit_wavefunction import QubitWaveFunction
from tequila import TequilaException, TequilaWarning




def get_aqt_backend(token: str = "") -> AQTResource | MQPBackend:

    if token == "":
        provider = AQTProvider("INVALID_TOKEN")
        backend = provider.get_backend('offline_simulator_no_noise')
        # backend = AQTEstimator(backend=aqtbackend, options={"shots": 100})
        print("using dummy backend")

    else:
        try:
            provider = MQPProvider(token)
            [backend] = provider.backends('AQT20')
        except:
            raise Exception('Invalid token')

    return backend


class TequilaAQTException(TequilaQiskitException):
    def __str__(self):
        return "Error in qiskit backend:" + self.message


class BackendCircuitAQT(BackendCircuitQiskit):
    token = "yoC3HyYJsgbEVJnNdcvlrfn5hLc7oJqxSTtRSiVnvOhH0hoadBdlKBGfaXtUSoYW" #input lrz token as string
    STATEVECTOR_DEVICE_NAME = get_aqt_backend(token=token) # todo let the token beaccessed from outside the file

    def do_simulate(self, variables, initial_state=0, *args, **kwargs) -> QubitWaveFunction:
        """
        Helper function for performing simulation.
        Parameters
        ----------
        variables:
            variables to pass to the circuit for simulation.
        initial_state:
            indicate initial state on which the unitary self.circuit should act.
        args
        kwargs

        Returns
        -------
        QubitWaveFunction:
            the result of simulation.
        """
        if self.noise_model is None:
            if self.device is None:
                qiskit_backend = self.retrieve_device(self.STATEVECTOR_DEVICE_NAME)
            else:
                if 'statevector' not in str(self.device):
                    raise TequilaException(
                        'For simulation, only state vector simulators are supported; received device={}, you might have'
                        ' forgoten to set the samples keyword - e.g. (device={}, samples=1000). If not set, tequila '
                        'assumes that full wavefunction simualtion is demanded which is not compatible with qiskit '
                        'devices or fake devices except for device=statevector'.format(self.device, self.device))
                else:
                    qiskit_backend = self.retrieve_device(self.device)
        else:
            raise TequilaQiskitException("wave function simulation with noise cannot be performed presently.")

        optimization_level = None
        if "optimization_level" in kwargs:
            optimization_level = kwargs['optimization_level']

        circuit = self.circuit.assign_parameters(self.resolver)

        circuit = self.add_state_init(circuit, initial_state)

        circuit = self.add_measurement(circuit, [0,1,2,3,4,5,6,7])

        #circuit.save_statevector() todo why doesnt it work ?????????
        aqt_shots = 200
        if "shots" in kwargs:
            aqt_shots = kwargs['shots']
        wavefunction = QubitWaveFunction(len(circuit.qubits))

        if type(qiskit_backend) == qiskit_aqt_provider.aqt_resource.OfflineSimulatorResource:
            estimator = AQTSampler(backend=qiskit_backend, options={"shots": aqt_shots})
            backend_result = estimator.run(circuit).result()
            print(backend_result)
            wavefunction._state = backend_result.quasi_dists[0]  # todo make initializer

        else:
            print("Using LRZ AQT backend")
            job = qiskit_backend.run(circuit,shots=aqt_shots, queued=True)
            backend_result = job.result()
            print(backend_result.results[0].data.counts)
            backend_dict = {}
            for key, value in backend_result.results[0].data.counts.items():
                backend_dict[int(key, 2)] = value / aqt_shots
            print(backend_dict)
            wavefunction._state = backend_dict  # todo make initializer

        return wavefunction


class BackendExpectationValueAQT(BackendExpectationValueQiskit):
    BackendCircuitType = BackendCircuitAQT