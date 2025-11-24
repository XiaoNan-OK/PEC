from itertools import product
import os
import numpy as np
from numpy.linalg import inv
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator

def transpile_mode(circuit: QuantumCircuit, backend=None, policy: str = "lock_n") -> QuantumCircuit:
    """
    Transpile a circuit according to policy.
    - "lock_n": keep the circuit at its logical n qubits using backend's basis_gates
                and a coupling_map restricted to n physical qubits (no expansion).
    - "pad_to_backend": map to backend.num_qubits with trivial layout, no routing.
    """
    n = circuit.num_qubits

    if policy == "lock_n":
        # 用 backend 的 gate set；同時把耦合圖裁成「只含 n 顆、且 relabel 成 0..n-1」的子圖
        basis = None
        if backend is not None:
            try:
                basis = backend.configuration().basis_gates
            except Exception:
                basis = None

        coupling = None
        if backend is not None:
            try:
                full_edges = backend.configuration().coupling_map  # [[u,v], ...]
            except Exception:
                full_edges = None
            if full_edges:
                # 從 0 開始 BFS 撈到 n 顆（簡易連通子圖）；不追求最佳，只求不擴張
                from collections import deque
                adj = {}
                for u, v in full_edges:
                    adj.setdefault(u, set()).add(v)
                    adj.setdefault(v, set()).add(u)
                start = 0 if adj else 0
                picked, dq, seen = [], deque([start]), {start}
                while dq and len(picked) < n:
                    u = dq.popleft()
                    picked.append(u)
                    for v in sorted(adj.get(u, [])):
                        if v not in seen:
                            seen.add(v); dq.append(v)
                # 若撈不到 n 顆，就退而求其次用 0..n-1
                if len(picked) < n:
                    picked = list(range(n))
                # 只保留子圖邊，並 relabel → 0..n-1
                S = set(picked)
                sub_edges = [(u, v) for (u, v) in full_edges if u in S and v in S]
                remap = {p: i for i, p in enumerate(picked)}
                remapped = [(remap[u], remap[v]) for (u, v) in sub_edges]
                coupling = CouplingMap(couplinglist=remapped)
        # 關鍵：不傳 backend（避免被擴到 backend.num_qubits）
        tc = transpile(
            circuit,
            basis_gates=basis,
            coupling_map=coupling,          # 只用 n 顆的局部耦合圖
            initial_layout=list(range(n)),  # 邏輯 0..n-1 → 局部 0..n-1
            layout_method="trivial",
            routing_method="sabre",         # 或 "none" 若原本就相鄰
            optimization_level=0,
        )
        # 保命檢查：不應被擴張
        assert tc.num_qubits == n, f"Transpiled size changed to {tc.num_qubits}, expected {n}"
        return tc

    elif policy == "pad_to_backend":
        if backend is None:
            raise ValueError("pad_to_backend 需要提供 backend")
        try:
            m = backend.num_qubits
        except Exception:
            m = backend.configuration().num_qubits
        if m < n:
            raise ValueError(f"Backend qubits {m} < logical qubits {n}")
        return transpile(
            circuit,
            backend=backend,
            optimization_level=0,
            layout_method="trivial",
            routing_method="none",
            initial_layout=list(range(n))
        )

    else:
        raise ValueError("policy must be 'lock_n' or 'pad_to_backend'")
    
def _extract_evs_list(prim_result):
    """
    Robustly extract a flat list of EVs from EstimatorV2 PrimitiveResult.
    Works with Aer EstimatorV2 and falls back to legacy iterable behavior.
    """
    results_list = getattr(prim_result, "results", None)
    if results_list is not None:
        return [np.atleast_1d(r.data.evs).ravel()[0] for r in results_list]
    try:
        return [np.atleast_1d(r.data.evs).ravel()[0] for r in prim_result]
    except Exception as e:
        raise TypeError(f"Unsupported Estimator result format: {type(prim_result)}") from e

def run_measurements(circuits, measurements, backend=None, shots=1024, estimator=None, batch_size=256):
    """
    Submit all (circuit, measurement) pairs to the provided Estimator.
    ...
    """
    # Use the externally provided estimator if given; otherwise create a local Aer EstimatorV2.
    if estimator is None:
        try:
            est = Estimator(backend=backend, options={"run_options": {"shots": shots}})
        except TypeError:
            # 部分環境的 EstimatorV2 沒有 backend 參數
            est = Estimator(options={"run_options": {"shots": shots}})
    else:
        est = estimator

    if shots is not None:
        try:
            est.options.default_shots = shots
        except Exception:
            pass

    is_2D = isinstance(next(iter(circuits.values())), dict)
    pairs = []
    if is_2D:
        for _, circuit_set in circuits.items():
            for _, Measure in measurements.items():
                for _, Cir in circuit_set.items():
                    pairs.append((Cir, Measure))
    else:
        for _, Measure in measurements.items():
            for _, Cir in circuits.items():
                pairs.append((Cir, Measure))

    raw_circs = [c for (c, _) in pairs]
    if backend is not None:
        tcs = [transpile_mode(c, backend=backend, policy="lock_n") for c in raw_circs]
    else:
        tcs = raw_circs

    batched_evs = []
    def chunk(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    meas_list = [m for (_, m) in pairs]
    for idx_chunk in chunk(list(range(len(tcs))), batch_size):
        pubs_chunk = [(tcs[i], meas_list[i]) for i in idx_chunk]
        res = est.run(pubs_chunk).result()
        batched_evs.extend(_extract_evs_list(res))

    return np.asarray(batched_evs, dtype=float)

def collect_results(jobs, qubit_count=None):
    """
    Reshape into Gram matrix G of size (4^n × 4^n).
    """
    prim_result = jobs.result()
    flat_evs = _extract_evs_list(prim_result)

    if qubit_count is None:
        L = int(np.sqrt(len(flat_evs)))
        n_guess = int(round(np.log(L)/np.log(4)))
        dim = 4**n_guess
    else:
        dim = 4**qubit_count

    g = np.asarray(flat_evs, dtype=float).reshape(dim, dim)
    return g


def _pauli_labels_n(n: int, order: str = "LR"):
    """Return all length-n Pauli strings in the requested order.
    LR: rightmost index fastest (lexicographic: II, IX, IY, IZ, XI, ...)
    RL: leftmost index fastest  (your spec:      II, XI, YI, ZI, IX, ...)"""
    alpha = ['I','X','Y','Z']
    if order.upper() == "LR":
        return [''.join(t) for t in product(alpha, repeat=n)]
    elif order.upper() == "RL":
        # reverse each tuple so leftmost moves fastest
        return [''.join(reversed(t)) for t in product(alpha, repeat=n)]
    else:
        raise ValueError("order must be 'LR' or 'RL'")

def _single_qubit_A():
    """Single-qubit preparation matrix (your original definition)."""
    return np.array([[1,  1, 1, 1],
                     [0,  0, 1, 0],
                     [0,  0, 0, 1],
                     [1, -1, 0, 0]], dtype=float)

def _kron_power(mat: np.ndarray, n: int) -> np.ndarray:
    """n-fold Kronecker product of mat."""
    out = np.array([1.0])
    for _ in range(n):
        out = np.kron(out, mat)
    return out

def _ax_rows():
    """Single-qubit Pauli selector row vectors."""
    a_i = np.array([[1, 0, 0, 0]], dtype=float)
    a_x = np.array([[0, 1, 0, 0]], dtype=float)
    a_y = np.array([[0, 0, 1, 0]], dtype=float)
    a_z = np.array([[0, 0, 0, 1]], dtype=float)
    return {'I': a_i, 'X': a_x, 'Y': a_y, 'Z': a_z}

def build_initial_states(qubit_count=2):
    """
    Create all 4^n product-state preparations over {zero, one, plus, right}.
    Keys look like 'q0zero_q1plus_...'.
    """
    qreg = QuantumRegister(qubit_count, 'q')
    creg = ClassicalRegister(qubit_count, 'c')

    def prep_single(qi: int, label: str, base: QuantumCircuit) -> QuantumCircuit:
        qc = base.copy()
        if label == 'zero':
            pass
        elif label == 'one':
            qc.x(qi)
        elif label == 'plus':
            qc.h(qi)
        elif label == 'right':
            qc.h(qi); qc.s(qi)
        else:
            raise ValueError(f"Unknown single-qubit state: {label}")
        return qc

    labels = ['zero', 'one', 'plus', 'right']
    base = QuantumCircuit(qreg, creg)
    base.reset(range(qubit_count))

    initial_states = {}
    for choices in product(labels, repeat=qubit_count):
        qc = base.copy()
        for qi, lab in enumerate(choices):
            qc = prep_single(qi, lab, qc)
        key = "_".join([f"q{idx}{lab}" for idx, lab in enumerate(choices)])
        initial_states[key] = qc
    
    return initial_states

def build_measurement_pauli(qubit_count=2, observable_number=None, *, order: str = "RL"):
    """Use RL by default to match your column order in G."""
    if observable_number is None:
        observable_number = qubit_count
    if observable_number < qubit_count:
        raise ValueError("observable_number must be >= qubit_count")

    pad = 'I' * (observable_number - qubit_count)
    Measurement = {}
    for pstr in _pauli_labels_n(qubit_count, order=order):
        Measurement[f"meas{pstr}"] = [SparsePauliOp([pad + pstr], coeffs=[1.0])]
    return Measurement

def get_preparation_matrix(qubit_count=2):
    """A = A_single^{⊗ n}."""
    return _kron_power(_single_qubit_A(), qubit_count)

def build_corrected_observables(g, A, qubit_count=2):
    """
    Return:
      B = g A^{-1}
      qq: dict[str -> (1 × 4^n) weights], key is an n-length Pauli string.
    """
    B = np.matmul(g, inv(A))
    B_inv = inv(B)

    a1 = _ax_rows()
    labels = _pauli_labels_n(qubit_count)

    qq = {}
    for pstr in labels:
        a = a1[pstr[0]]
        for ch in pstr[1:]:
            a = np.kron(a, a1[ch])
        qq[pstr] = np.matmul(a, B_inv)  # shape (1, 4^n)
    return B, qq

def build_ideal_measurement(qq, qubit_count=2, observable_number=None):
    """
    Build ideal measurement operators from qq:
      'measp<target>' -> Σ_r qq[target][r] · (I^{⊗(m-n)} ⊗ r)
    """
    if observable_number is None:
        observable_number = qubit_count
    if observable_number < qubit_count:
        raise ValueError("observable_number must be >= qubit_count")

    pad = 'I' * (observable_number - qubit_count)
    labels = _pauli_labels_n(qubit_count)

    out = {}
    for p in labels:
        coeffs, paulis = [], []
        for i, r in enumerate(labels):
            coeffs.append(qq[p][0][i])
            paulis.append(pad + r)
        out[f"measp{p}"] = SparsePauliOp(paulis, coeffs=np.asarray(coeffs, dtype=float))
    return out

def readout_pec_from_circuit(circuit: QuantumCircuit, estimator, *, 
                             shots=None, transpile_backend=None, 
                             transpile_policy: str = "lock_n",   # "lock_n" 或 "pad_to_backend"
                             batch_size: int = 256):
    """
    One-call Readout PEC calibration driven only by (circuit, estimator).
    - Infers n = circuit.num_qubits.
    - Builds 4^n preparations and 4^n Pauli measurements.
    - Executes calibration jobs using the provided 'estimator'.
    - Returns B and qq (and also G, A, IdealMeasurement for convenience).
    """
    n = circuit.num_qubits
    if transpile_policy == "pad_to_backend" and transpile_backend is not None:
        try:
            m = transpile_backend.num_qubits
        except Exception:
            m = transpile_backend.configuration().num_qubits
    else:
        m = n  # ★ 預設 lock_n：observable 長度就維持 n

    inits = build_initial_states(qubit_count=n)
    meas  = build_measurement_pauli(qubit_count=n, observable_number=m, order="RL")

    # 內部 run_measurements 會用 policy="lock_n" + backend 的 n-子圖做轉譯（不擴張）
    evs = run_measurements(inits, meas, backend=transpile_backend, shots=shots, estimator=estimator, batch_size=batch_size)
    dim = 4**n
    G = np.asarray(evs, dtype=float).reshape(dim, dim)

    A = get_preparation_matrix(qubit_count=n)
    B, RD_weight = build_corrected_observables(G, A, qubit_count=n)
    IdealMeas = build_ideal_measurement(RD_weight, qubit_count=n, observable_number=n)

    return {
        "n": n,
        "G": G,
        "A": A,
        "B": B,
        "ReadoutWeight": RD_weight,
        "IdealMeasurement": IdealMeas,
    }