import numpy as np
import qiskit as q
from numpy.linalg import inv
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.backend import BackendV1
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

# --- IBM 連線與後端設定 ---
def load_ibm_backend(token: str, instance: str, backend_name: str):
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
def build_initial_states(qubit_count = 2):
    qreg = QuantumRegister(qubit_count, 'q')
    creg = ClassicalRegister(qubit_count, 'c')

    initial_states = {}

    # 單 qubit 狀態 (針對 qubit 0)
    q0_states = {
        'zero': QuantumCircuit(qreg, creg),
        'one': QuantumCircuit(qreg, creg),
        'plus': QuantumCircuit(qreg, creg),
        'right': QuantumCircuit(qreg, creg)
    }
    q0_states['one'].x(0)
    q0_states['plus'].h(0)
    q0_states['right'].h(0); q0_states['right'].s(0)

    if qubit_count == 1:
        return q0_states

    # 如果是兩個 qubit，建立 qubit 1 狀態
    q1_states = {
        'zero': QuantumCircuit(qreg, creg),
        'one': QuantumCircuit(qreg, creg),
        'plus': QuantumCircuit(qreg, creg),
        'right': QuantumCircuit(qreg, creg)
    }
    q1_states['one'].x(1)
    q1_states['plus'].h(1)
    q1_states['right'].h(1); q1_states['right'].s(1)

    for q0k, qc0 in q0_states.items():
        for q1k, qc1 in q1_states.items():
            new_circ = qc0.compose(qc1)
            initial_states[f"q0{q0k}_q1{q1k}"] = new_circ

    return initial_states

# --- 準備量測基底電路 ---
def build_measurement_pauli(qubit_count=2):
    Measurement = {}
    pauli_labels = ['I', 'X', 'Y', 'Z']
    if qubit_count == 1:
        for p in pauli_labels:
            Measurement[f"meas{p}"] = [Pauli(p)]
    else:
        for p1 in pauli_labels:
            for p2 in pauli_labels:
                key = f"meas{p2}{p1}"
                Measurement[key] = [Pauli('I'*0 + p2 + p1)]
    return Measurement

# --- 執行模擬任務 ---
def run_measurements(circuits, measurements, backend, shots=1000):
    estimator = Estimator(mode=backend)
    estimator.options.default_shots = shots
    batched_inputs = []
    # keys = []
    for m_key, paulis in measurements.items():
        for name, circ in circuits.items():
            CirTran = q.compiler.transpile(circ, backend=backend, optimization_level=0)
            batched_inputs.append((CirTran, paulis))
            # keys.append(f"{name}_{m_key}")
    jobs = estimator.run(batched_inputs)
    return jobs

# --- 收集結果 ---
def collect_results(jobs, qubit_count=2):
    # results = []
    # for name, job in jobs.items():
    #     res = job.result()
        # results.append(res[0].data.evs)
    result = jobs.result()
    job = [res.data.evs for res in result]
    dim = 4 if qubit_count == 1 else 16
    g = np.array(job).reshape(dim, dim)
    return g

# --- 建立 A 矩陣 ---
def get_preparation_matrix(qubit_count=2):
    A_single = np.array([[1,  1, 1, 1],
                         [0,  0, 1, 0],
                         [0,  0, 0, 1],
                         [1, -1, 0, 0]])
    A = A_single if qubit_count == 1 else np.kron(A_single, A_single)
    return A

# --- 修正觀察值 ---
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
                key = f'{f}{s}'
                a_mix = np.kron(first, second)
                q_mix = np.matmul(a_mix, B_inv)
                qq[key] = q_mix
    return qq

# --- 建立理想觀測量 ---
def build_ideal_measurement(qq, qubit_count=2):
    IdealMeasurement = {}
    if qubit_count == 1:
        Observable = ['I', 'X', 'Y', 'Z']
        for obs in Observable:
            IdealObservable = SparsePauliOp(['I'], coeffs=[qq[obs][0][0]])
            for i in range(1, len(Observable)):
                # IdealObservable += SparsePauliOp([('I' * (backendqubitNum-2) + Observable[i])], coeffs=[qq[obs][0][i]])
                IdealObservable += SparsePauliOp([(Observable[i])], coeffs=[qq[obs][0][i]])
                IdealMeasurement[f"meas{obs}"] = IdealObservable
    else:
        Observable = ['II', 'XI', 'YI', 'ZI', 'IX', 'XX', 'YX', 'ZX', 'IY', 'XY', 'YY', 'ZY', 'IZ', 'XZ', 'YZ', 'ZZ']       
        for obs in Observable:
            IdealObservable = SparsePauliOp(['II'], coeffs=[qq[obs][0][0]])
            for i in range(1, len(Observable)):
                # IdealObservable += SparsePauliOp([('I' * (backendqubitNum-2) + Observable[i])], coeffs=[qq[obs][0][i]])
                IdealObservable += SparsePauliOp([(Observable[i])], coeffs=[qq[obs][0][i]])
                IdealMeasurement[f"meas{obs}"] = IdealObservable
    return IdealMeasurement

