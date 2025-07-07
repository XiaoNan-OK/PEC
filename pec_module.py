import numpy as np
import qiskit as q
from numpy.linalg import inv
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

#Construct Pauli Pairs
pauli = ["I", "X", "Y", "Z"]
pauli_pair = {}
for i in pauli:
    for j in pauli:
        pauli_pair.update({f"{i}{j}":np.array(Pauli(f"{i}{j}").to_matrix())})        

#Build Cnot Matrrix
Cnot = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0 ,0, 0, 1],
                 [0, 0, 1, 0]])

# --- Connect To IBMQ ---
def load_ibm_backend(token: str, instance: str, backend_name: str):
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=token,
        instance=instance,
        set_as_default=True,
        overwrite=True
    )
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    noise_model = NoiseModel.from_backend(backend)
    return backend, noise_model, service

# --- Prepare Initial State ---
def build_initial_states(qubit_count = 2):
    qreg = QuantumRegister(qubit_count, 'q')
    creg = ClassicalRegister(qubit_count, 'c')

    initial_states = {}
    default_circuits = QuantumCircuit(qreg, creg)
    
    # Prepare State of Qubit1 if needed
    q1_states = {
        'zero': default_circuits.copy(),
        'one': default_circuits.copy(),
        'plus': default_circuits.copy(),
        'right': default_circuits.copy()
    }
    q1_states['one'].x(1)
    q1_states['plus'].h(1)
    q1_states['right'].h(1); q1_states['right'].s(1)

    # Prepare Sate of Qubit0
    default_circuits.reset(range(2))
    q0_states = {
        'zero': default_circuits.copy(),
        'one': default_circuits.copy(),
        'plus': default_circuits.copy(),
        'right': default_circuits.copy()
    }
    q0_states['one'].x(0)
    q0_states['plus'].h(0)
    q0_states['right'].h(0); q0_states['right'].s(0)

    if qubit_count == 1:
        return q0_states

    for q0k, qc0 in q0_states.items():
        for q1k, qc1 in q1_states.items():
            new_circ = qc0.compose(qc1)
            initial_states[f"q0{q0k}_q1{q1k}"] = new_circ

    return initial_states

# --- Prepare Measurement ---
def build_measurement_pauli(qubit_count=2, observable_number=2):
    Measurement = {}
    pauli_labels = ['I', 'X', 'Y', 'Z']
    if qubit_count == 1:
        for p in pauli_labels:
            Measurement[f"meas{p}"] = [Pauli(p)]
    else:
        for p1 in pauli_labels:
            for p2 in pauli_labels:
                key = f"meas{p2}{p1}"
                Measurement[key] = [Pauli('I'*(observable_number-2) + p2 + p1)]
    return Measurement

# --- Run Simulation ---
def run_measurements(circuits, measurements, backend, shots=1000):
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = shots
    batched_inputs = []

    # Detect if 2D (Twirling-like) or 1D (basic circuit dict)
    is_2D = isinstance(next(iter(circuits.values())), dict)

    if is_2D:
        # circuits is 2D
        for pauli, circuit_set in circuits.items():
            for MeasName, Measure in measurements.items():
                for CirName, Cir in circuit_set.items():
                    CirTran = q.compiler.transpile(Cir, backend=backend, optimization_level=0)
                    batched_inputs.append((CirTran, Measure))
    else:
        for MeasName, Measure in measurements.items():
            for CirName, Cir in circuits.items():
                CirTran = q.compiler.transpile(Cir, backend=backend, optimization_level=0)
                batched_inputs.append((CirTran, Measure))

    jobs = estimator.run(batched_inputs)
    return jobs

# --- Construct Gram Matrix ---
def collect_results(jobs, qubit_count=2):
    # results = []
    # for name, job in jobs.items():
    #     res = job.result()
        # results.append(res[0].data.evs)
    result = jobs.result()
    data = [res.data.evs for res in result]
    dim = 4 if qubit_count == 1 else 16
    g = np.array(data).reshape(dim, dim)
    return g

# --- Construct Gram Matrix ---
def collect_twirling_results(jobs, qubit_count=2):
    # results = []
    # for name, job in jobs.items():
    #     res = job.result()
        # results.append(res[0].data.evs)
    result = jobs.result()
    data = [res.data.evs for res in result]
    dim = 4 if qubit_count == 1 else 16
    g = np.array(data).reshape(dim, dim, dim)
    return g

# --- Construct A Matrix ---
def get_preparation_matrix(qubit_count=2):
    A_single = np.array([[1,  1, 1, 1],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1],
                         [1, -1, 0, 0]])
    A = A_single if qubit_count == 1 else np.kron(A_single, A_single)
    return A

# --- Calculate the coefficient of Noisy Observable ---
def build_corrected_observables(g, A, qubit_count=2):
    B = np.matmul(g, inv(A))
    B_inv = inv(B)

    a_i = np.array([[1, 0, 0, 0]])
    a_x = np.array([[0, 1, 0, 0]])
    a_y = np.array([[0, 0, 1, 0]])
    a_z = np.array([[0, 0, 0, 1]])
    a = {'I':a_i, 'X':a_x, 'Y':a_y, 'Z':a_z}

    qq = {}
    if qubit_count == 1:
        for k, val in a.items():
            q_mix = np.matmul(val, B_inv)
            qq[k] = q_mix
    else:
        for f, first in a.items():
            for s, second in a.items():
                key = f'{s}{f}'
                a_mix = np.kron(first, second)
                q_mix = np.matmul(a_mix, B_inv)
                qq[key] = q_mix
    return B, qq

# --- Build the Ideal Observable ---
def build_ideal_measurement(qq, qubit_count=2, observable_number=2):
    IdealMeasurement = {}
    if qubit_count == 1:
        Observable = ['I', 'X', 'Y', 'Z']
        for obs in Observable:
            IdealObservable = SparsePauliOp(['I' * (observable_number-1) + 'I'], coeffs=[qq[obs][0][0]])
            for i in range(1, len(Observable)):
                IdealObservable += SparsePauliOp(['I' * (observable_number-1) + Observable[i]], coeffs=[qq[obs][0][i]])
                # IdealObservable += SparsePauliOp([(Observable[i])], coeffs=[qq[obs][0][i]])
                IdealMeasurement[f"meas{obs}"] = IdealObservable
    else:
        Observable = ['II', 'XI', 'YI', 'ZI', 'IX', 'XX', 'YX', 'ZX', 'IY', 'XY', 'YY', 'ZY', 'IZ', 'XZ', 'YZ', 'ZZ']       
        for obs in Observable:
            IdealObservable = SparsePauliOp(['I' * (observable_number-2) + 'II'], coeffs=[qq[obs][0][0]])
            for i in range(1, len(Observable)):
                IdealObservable += SparsePauliOp([('I' * (observable_number-2) + Observable[i])], coeffs=[qq[obs][0][i]])
                # IdealObservable += SparsePauliOp([(Observable[i])], coeffs=[qq[obs][0][i]])
                IdealMeasurement[f"meas{obs}"] = IdealObservable
    return IdealMeasurement

# --- Prepare Circuit With Cnot gate ---
def add_cnot_gate(Initial_state): 
    CnotCircuit = {}
    for InitName, Initial in Initial_state.items():
        cir = Initial.copy()
        cir.cx(0, 1)
        CnotCircuit.update({f'{InitName}':cir})
    return CnotCircuit

# --- Build Cnot Gate Noisy ---
def build_cnot_noisy_channel(A, B, g):
    return np.matmul(np.matmul(inv(B), g), inv(A))

# --- Quantum State Evolution ---
def evolution(gate, state):
        return np.matmul(np.matmul(gate, state), np.conj(gate).T) 

# --- PTM of Cnot Gate ---
def build_cnot_ptm():
    IdealCnot = np.ones((16,16),  dtype=complex)
    i = 0
    for matrix1 in pauli_pair.values():
        j = 0
        for matrix2 in pauli_pair.values():
            applycnot = evolution(Cnot, matrix1)
            element = np.matmul(matrix2, applycnot)
            IdealCnot[i][j] = np.round(np.trace(element)/4)
            j+=1
        i+=1 
    return IdealCnot

# Pauli Twirling ----------------------------------------------------------------------------------------------------------

# --- Compare Pauli in front of and behind Cnot gate ---
def pauli_transfer_matrix():
    pauli_transfer = {}
    for channel, matrix in pauli_pair.items():
        new_matrix = evolution(Cnot, matrix)
        for new_channel, comp_matrix in pauli_pair.items():
            if np.array_equal(new_matrix, comp_matrix) or np.array_equal(new_matrix, -comp_matrix):
                pauli_transfer.update({channel:new_channel})
    return pauli_transfer

# --- Generate Pauli Twirling Circuit ---
def generate_pauli_circuits(index):
    circuits = {"I": QuantumCircuit(2, 2),
                "X": QuantumCircuit(2, 2),
                "Y": QuantumCircuit(2, 2),
                "Z": QuantumCircuit(2, 2)}
    # X-measurement
    circuits["X"].x(index)
    # Y-measurement
    circuits["Y"].y(index)
    # Z-measurement
    circuits["Z"].z(index)
    return circuits  
    
# --- Combine Pauli ppair and Cnotgate Cirucit ---
def generate_twirling_circuits(InitialState, PauliTwirling, transfer):
    Twirling_circuit_set = {}
    for firstPauli, secondPauli in transfer.items():
        Twirling_circuit = {}
        for InitName, Initial in InitialState.items():
            AddPauli = q.circuit.QuantumCircuit.compose(Initial, PauliTwirling[firstPauli])
            AddPauli.cx(0, 1)
            AddPauliInverse = q.circuit.QuantumCircuit.compose(AddPauli, PauliTwirling[secondPauli])
            Twirling_circuit.update({InitName:AddPauliInverse})
        Twirling_circuit_set.update({firstPauli:Twirling_circuit})
    return Twirling_circuit_set