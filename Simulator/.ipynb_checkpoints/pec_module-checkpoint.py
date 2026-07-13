import numpy as np
from numpy.linalg import inv
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.backend import BackendV1
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# --- IBM 連線與後端設定 ---
def load_ibm_backend(token: str, instance: str, backend_name: str = "ibm_kyiv"):
    QiskitRuntimeService.save_account(
        channel="ibm_quantum",
        token=token,
        instance=instance,
        set_as_default=True,
        overwrite=True
    )
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    noise_model = NoiseModel.from_backend(backend)
    return backend, noise_model

# --- 準備初始狀態電路 ---
def build_initial_states():
    qreg = QuantumRegister(2, 'q')
    creg = ClassicalRegister(2, 'c')

    q0_states = {
        'q0zero': QuantumCircuit(qreg, creg),
        'q0one': QuantumCircuit(qreg, creg),
        'q0plus': QuantumCircuit(qreg, creg),
        'q0right': QuantumCircuit(qreg, creg)
    }
    q0_states['q0one'].x(0)
    q0_states['q0plus'].h(0)
    q0_states['q0right'].h(0); q0_states['q0right'].s(0)

    q1_states = {
        'q1zero': QuantumCircuit(qreg, creg),
        'q1one': QuantumCircuit(qreg, creg),
        'q1plus': QuantumCircuit(qreg, creg),
        'q1right': QuantumCircuit(qreg, creg)
    }
    q1_states['q1one'].x(1)
    q1_states['q1plus'].h(1)
    q1_states['q1right'].h(1); q1_states['q1right'].s(1)

    circuits = {}
    for q0k, qc0 in q0_states.items():
        for q1k, qc1 in q1_states.items():
            new_circ = qc0.compose(qc1)
            circuits[f"{q0k}_{q1k}"] = new_circ
    return circuits

# --- 準備量測基底電路 ---
def build_measurement_pauli(backend):
    Measurement = {}
    pauli_labels = ['I', 'X', 'Y', 'Z']
    for i, p1 in enumerate(pauli_labels):
        for j, p2 in enumerate(pauli_labels):
            key = f"meas{p1}{p2}"
            Measurement[key] = [Pauli(p1), Pauli(p2)]
    return Measurement

# --- 執行模擬任務 ---
def run_measurements(circuits, measurements, backend):
    estimator = Estimator(options={"default_shots": 100000})
    jobs = {}
    for name, circ in circuits.items():
        for m_key, paulis in measurements.items():
            job = estimator.run([(circ, SparsePauliOp(paulis))])
            jobs[f"{name}_{m_key}"] = job
    return jobs

# --- 收集結果 ---
def collect_results(jobs):
    results = []
    for name, job in jobs.items():
        res = job.result()
        results.append(res[0].data.evs)
    g = np.array(results).reshape(16, 16)
    return g

# --- 建立 A 矩陣 ---
def get_preparation_matrix():
    A_single = np.array([[1, 1, 1, 1],
                         [0, 0, 1, 1],
                         [0, 1, 0, 1],
                         [1, 0, 0, 1]])
    A = np.kron(A_single, A_single)
    A_inv = inv(A)
    return A, A_inv

# --- 修正觀察值 ---
def build_corrected_observables(g, A_inv):
    B = np.matmul(g, A_inv)
    B_inv = inv(B)

    a_x = np.array([[0, 1, 0, 0]])
    a_y = np.array([[0, 0, 0, 1]])
    a_z = np.array([[1, 0, 0, 0]])
    a = {'X': a_x, 'Y': a_y, 'Z': a_z}

    qq = {}
    for f, first in a.items():
        for s, second in a.items():
            key = f'{f}{s}'
            a_mix = np.kron(first, second)
            q_mix = np.matmul(a_mix, B_inv)
            qq[key] = q_mix
    return qq

# --- 建立理想觀測量 ---
def build_ideal_measurement(qq, backend_num_qubits):
    Observable = ['I', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
                  'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
    IdealMeasurement = {}
    for obs in Observable:
        IdealObservable = SparsePauliOp([obs], coeffs=[qq[obs][0]])
        IdealMeasurement[f"meas{obs}"] = IdealObservable
    return IdealMeasurement
