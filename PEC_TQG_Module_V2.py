import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from functools import reduce, lru_cache  
from itertools import product, islice
from typing import Dict, List, Tuple, Optional, Iterable

# --------- (1) laod circuit and get informations ----------
def _ensure_quantum_circuit(obj):
    """確保拿到的是 QuantumCircuit；否則拋出可讀錯誤。
       也容忍像 (circ, something) 這種 tuple，會自動取第 1 個。
    """
    if isinstance(obj, QuantumCircuit):
        return obj
    # 容忍包成 (circ, ...) 的情況
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], QuantumCircuit):
        return obj[0]
    # 盡量友善提示 CircuitInstruction 這種情況
    clsname = obj.__class__.__name__
    if clsname == "CircuitInstruction":
        raise TypeError("Got a CircuitInstruction (likely from circuit.data[k]). "
                        "Pass the full QuantumCircuit, not a single instruction.")
    raise TypeError(f"Expected QuantumCircuit, got {type(obj)} ({clsname}).")

def _used_qubit_indices(circ: QuantumCircuit) -> List[int]:
    """Return sorted list of qubit indices that are actually used by gates."""
    used = set()
    for inst in circ.data:
        try:
            op, qargs = inst.operation, inst.qubits
        except AttributeError:
            op, qargs, _ = inst
        for q in qargs:
            used.add(circ.find_bit(q).index)
    return sorted(used)

# ---- get circuit & cnot information ----
def cnot_pec(input_circuit: QuantumCircuit): 
    input_circuit = _ensure_quantum_circuit(input_circuit)
    n = input_circuit.num_qubits
    cnots = []
    for k, inst in enumerate(input_circuit.data):
        try:
            op = inst.operation; qargs = inst.qubits
        except AttributeError:
            op, qargs, _ = inst
        if op.name in ("cx", "cnot"):
            c = input_circuit.find_bit(qargs[0]).index
            t = input_circuit.find_bit(qargs[1]).index
            cnots.append({"data_idx": k, "control": c, "target": t})
    return {"num_qubits": n, "cnot_list": cnots}

def unique_cnot_keys(input_circuit: QuantumCircuit) -> List[Tuple[int,int]]:
    info = cnot_pec(input_circuit)
    seen, keys = set(), []
    for rec in info["cnot_list"]:
        key = (rec["control"], rec["target"])
        if key not in seen:
            seen.add(key); keys.append(key)
    return keys


# --------- (2) Construct Initial, observable set ----------
#---- Construct initial state labels：{|0>,|1>,|+>,|R>} ⊗ 4^n, from left ----
def enum_initial_labels(n: int) -> List[str]:
    return [''.join(p) for p in product(('0','1','+','R'), repeat=n)]

# ---- Construct observable labels：{I,X,Y,Z} ⊗ 4^n, from left----
def enum_pauli_labels(n: int) -> List[str]:
    return [''.join(p) for p in product(('I','X','Y','Z'), repeat=n)]

# ---- Construct initial circuits：{|0>,|1>,|+>,|R>} ⊗ 4^n ----
def init_circuit_from_label(n: int, label: str) -> QuantumCircuit:
    """
    key: e.g. '0+R' represented by q0=|0>, q1=|+>, q2=|R>
    """
    qc = QuantumCircuit(n, name=f"init_{label}")
    for q, ch in enumerate(label):
        if ch == '1': qc.x(q)
        elif ch == '+': qc.h(q)
        elif ch == 'R': qc.h(q); qc.s(q)
    return qc

#---- Construct observables：{I,X,Y,Z} ⊗ 4^n ----
def make_observable_map(labels: List[str]) -> Dict[str, SparsePauliOp]:
    # Change from starting from the left to starting from the right
    return {lab: SparsePauliOp.from_list([(''.join(lab[::-1]), 1.0)]) for lab in labels}


# --------- (3) Construct pauli layer set ----------
# ---- Calculate pauli after cnot gate ----
_letter_to_xz = {'I':(0,0), 'X':(1,0), 'Y':(1,1), 'Z':(0,1)}
_xz_to_letter = {(0,0):'I', (1,0):'X', (1,1):'Y', (0,1):'Z'}

def _label_to_xz(label: str):
    x=[]; z=[]
    for ch in label:
        xi, zi = _letter_to_xz[ch]
        x.append(xi); z.append(zi)
    return x, z

def _xz_to_label(x, z):
    return ''.join(_xz_to_letter[(xi, zi)] for xi, zi in zip(x, z))

def pauli_conj_by_cnot(label: str, c: int, t: int) -> str:
    x, z = _label_to_xz(label)
    # xt′​=xt​⊕xc ; ​zc′​=zc​⊕zt​ ; ​⊕: ^ in python
    x[t] = x[t] ^ x[c]
    z[c] = z[c] ^ z[t]
    return _xz_to_label(x, z)

# ---- Construct pauli layer：{I,X,Y,Z} ⊗ 4^n ----
def apply_pauli_layer(qc: QuantumCircuit, label: str):
    for q, ch in enumerate(label):
        if ch == 'X': qc.x(q)
        elif ch == 'Y': qc.y(q)
        elif ch == 'Z': qc.z(q)


# --------- (4) Construct Cnot circuit ----------
# ---- Construct init -> pre-Pauli -> CNOT(c,t) -> post-Pauli ----
def build_twirled_circuit(n: int, init_circ: QuantumCircuit, c: int, t: int, pre_label: str) -> QuantumCircuit:
    post_label = pauli_conj_by_cnot(pre_label, c, t)
    qc = QuantumCircuit(n, name=f"twirl(c{c},t{t})_init={init_circ.name[5:]}_pre={pre_label}")
    qc.compose(init_circ, inplace=True)
    apply_pauli_layer(qc, pre_label)
    qc.cx(c, t)
    apply_pauli_layer(qc, post_label)
    return qc


# --------- (5) Calculate average expectation matrix of each cnot ----------
# ---- calculate the expectation of single circuit in batch type ----
def _extract_evs_from_result(est_res):
    """
    Return a 1D np.array of expectation values, compatible with:
    - EstimatorV2 (PrimitiveResult[PubResult])
    - EstimatorV1 / BackendEstimator (EstimatorResult)
    """
        # --- EstimatorV2 path ---
    # 1) Preferred: result.records (some builds expose this)
    recs = getattr(est_res, "records", None)
    # 2) Fallback: PrimitiveResult is iterable over PubResult
    if recs is None and hasattr(est_res, "__iter__"):
        try:
            recs = list(est_res)
        except Exception:
            recs = None

    if recs is not None:
        vals = []
        for rec in recs:
            d = getattr(rec, "data", None)
            if d is None:
                continue
            # DataBin / attribute-style (你的環境)
            if hasattr(d, "evs"):
                vals.extend(np.asarray(d.evs, dtype=float).ravel().tolist())
            # dict-style（部分版本）
            elif isinstance(d, dict):
                if "evs" in d:
                    vals.extend(np.asarray(d["evs"], dtype=float).ravel().tolist())
        if vals:
            return np.asarray(vals, dtype=float).ravel()

    # --- EstimatorV1 / BackendEstimator fallbacks ---
    vals = getattr(est_res, "values", None)
    if vals is not None:
        return np.asarray(vals, dtype=float).ravel()

    data = getattr(est_res, "data", None)
    if data is not None and getattr(data, "evs", None) is not None:
        return np.asarray(data.evs, dtype=float).ravel()

    rs = getattr(est_res, "results", None)
    if rs and len(rs) > 0 and getattr(rs[0], "data", None) is not None:
        d = rs[0].data
        if getattr(d, "evs", None) is not None:
            return np.asarray(d.evs, dtype=float).ravel()

    raise TypeError(f"Unsupported EstimatorResult format: {type(est_res)}")

def _estimator_expect_many(estimator, circuit, observables):
    # v2 風格（publishables）
    try:
        pubs = [(circuit, list(observables), None)]
        res = estimator.run(pubs).result()
        return _extract_evs_from_result(res)

    except Exception:
        # v1 退路
        k = len(observables)
        res = estimator.run(circuits=[circuit]*k, observables=list(observables)).result()
        return _extract_evs_from_result(res)

def estimate_over_all_observables(estimator, circuit: QuantumCircuit, obs_labels: List[str], 
                                  obs_map: Dict[str, SparsePauliOp], obs_batch: int = 256) -> np.ndarray:
    out = np.zeros(len(obs_labels), dtype=float)
    for k0 in range(0, len(obs_labels), obs_batch):
        k1 = min(len(obs_labels), k0 + obs_batch)
        observables = [obs_map[lab] for lab in obs_labels[k0:k1]]
        evs = _estimator_expect_many(estimator, circuit, observables)
        if evs.shape[0] != (k1 - k0):
            raise ValueError(f"Estimator returned {evs.shape[0]} values but expected {k1-k0}")
        out[k0:k1] = evs
    return out

# ---- calculate the column elements of single CNOT, single initial state ----
def compute_column_for_init(estimator,
                            n: int,
                            c: int, t: int,
                            init_label: str,
                            obs_labels: List[str],
                            obs_map: Dict[str, SparsePauliOp],
                            pre_labels: Optional[List[str]] = None,
                            obs_batch: int = 256) -> np.ndarray:
    if pre_labels is None:
        pre_labels = enum_pauli_labels(n)
    init_circ = init_circuit_from_label(n, init_label)
    acc = np.zeros(len(obs_labels), dtype=float)

    for pre in pre_labels:
        qc = build_twirled_circuit(n, init_circ, c, t, pre)
        acc += estimate_over_all_observables(estimator, qc, obs_labels, obs_map, obs_batch=obs_batch)
        del qc  # 顯式釋放（可省略）

    return acc / float(len(pre_labels))

# ---- calculate Gram matrix of single CNOT ----
def compute_matrix_for_tqg(estimator,
                            input_circuit: QuantumCircuit,
                            key: Tuple[int,int],
                            obs_batch: int = 256) -> Tuple[np.ndarray, List[str], List[str]]:
    n = input_circuit.num_qubits
    c, t = key
    init_labels = enum_initial_labels(n)     # 4^n
    obs_labels  = enum_pauli_labels(n)       # 4^n
    obs_map     = make_observable_map(obs_labels)
    pre_labels  = enum_pauli_labels(n)       # 4^n

    M = np.zeros((len(obs_labels), len(init_labels)), dtype=float)
    for j, init_lab in enumerate(init_labels):
        col = compute_column_for_init(estimator, n, c, t, init_lab,
                                      obs_labels, obs_map,
                                      pre_labels=pre_labels, obs_batch=obs_batch)
        M[:, j] = col
        print(f"(c{c},t{t}) {j+1}/{len(init_labels)}")

    return M, obs_labels, init_labels

# ---- calculate average gram matrix with each cnot ----
def compute_tqg_matrices(estimator, input_circuit: QuantumCircuit, obs_batch: int = 256):
    """
    回傳：
      {
        'num_qubits': n,
        'row_labels': [obs_label...],   # 4^n
        'col_labels': [init_label...],  # 4^n
        'matrices': { (c,t): np.ndarray((4^n, 4^n)) }
      }
    """
    n = input_circuit.num_qubits
    matrices: Dict[Tuple[int,int], np.ndarray] = {}
    row_labels = col_labels = None

    for key in unique_cnot_keys(input_circuit):
        M, rlabs, clabs = compute_matrix_for_tqg(estimator, input_circuit, key, obs_batch=obs_batch)
        matrices[key] = M
        row_labels = rlabs; col_labels = clabs

    return {
        "num_qubits": n,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "matrices": matrices,
    }


# --------- (6) Calculate average expectation matrix of each cnot ----------
# ---- calculate cnot Pauli Twirling Matrix ----
try:
    from scipy.sparse import csr_matrix, identity as sp_identity
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def cnot_pauli_twirling_matrix(n: int,
                               c: Optional[int],
                               t: Optional[int],
                               *,
                               sparse: bool = True):
    
    """
    Return a PTM matrix 𝑀 of shape (4^𝑛, 4^𝑛) such that 𝑀[:, 𝑗] = 𝑒𝑓(𝑗);
    that is, the 𝑗-th Pauli label (an 𝑛-qubit string) is mapped, under conjugation by CNOT(𝑐, 𝑡), 
    to the position of its new label. All other qubits naturally tensor with the ideal identity.
    If 𝑐 or 𝑡 is None (i.e., no CNOT), return the identity PTM.
    Return (M, labels), where labels gives the row/column order of the basis {𝐼, 𝑋, 𝑌, 𝑍}^𝑛.
    """

    N = 4**n
    labels = enum_pauli_labels(n)

    if c is None or t is None:
        if sparse and _HAS_SCIPY:
            return sp_identity(N, format='csr'), labels
        return np.eye(N, dtype=float), labels

    idx = {lab: i for i, lab in enumerate(labels)}
    rows, cols = [], []

    for j, lab in enumerate(labels):
        lab2 = pauli_conj_by_cnot(lab, c, t)  # 
        i = idx[lab2]
        rows.append(i); cols.append(j)

    if sparse and _HAS_SCIPY:
        data = np.ones(len(rows), dtype=float)
        M = csr_matrix((data, (rows, cols)), shape=(N, N))
    else:
        M = np.zeros((N, N), dtype=float)
        M[rows, cols] = 1.0

    return M, labels

# ---- calculate Initial Matrix ----
_A_SINGLE = np.array([[1,  1, 1, 1],
                      [0,  0, 1, 0],
                      [0,  0, 0, 1],
                      [1, -1, 0, 0]], dtype=float)

# 可能有 scipy 就順帶支援 sparse；沒有也能跑 dense
try:
    from scipy.sparse import csr_matrix, kron as sp_kron, issparse
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    def issparse(_): return False

def _kron_power_dense(mat, n: int):
    out = mat
    for _ in range(n-1):
        out = np.kron(out, mat)
    return out

def _kron_power_sparse(mat, n: int):
    # sparse Kronecker power（需要 SciPy）
    A = csr_matrix(mat)
    for _ in range(n-1):
        A = sp_kron(A, csr_matrix(mat), format='csr')
    return A

# ---- calculate Average pauli twirling Matrix ----
def averaged_pauli_twirling_matrix(obj, B, *, n: int | None = None, c: int | None = None, t: int | None = None, cxm_sparse: bool = True, ainv_sparse: bool = False):
    """
    Apply a transformation to the output of `compute_tqg_matrices(...)`:
    M' = CXM^{-1} * B^{-1} * GM * A^{-1}
    Details:
        - CXM:
            Obtained from `cnot_pauli_twirling_matrix(n, c, t, sparse=cxm_sparse)`.
            Since the PTM of a CNOT is a permutation matrix, CXM^{-1} = CXM^T.
        - A^{-1}:
            First compute A_single^{-1} = inv(_A_SINGLE), then take the Kronecker power
            to n qubits.
        - B^{-1}:
            The inverse of the "Read out" matrix B, which converts a density matrix
    Parameters:
        cxm_sparse (bool): 
            Whether to use a sparse CXM (recommended: True).
        ainv_sparse (bool): 
            Whether to use a sparse A^{-1} (recommended: True for large n; requires SciPy).
    Returns:
        dict: Same structure as `tqg_pack`, but with the matrices inside 'matrices'
            replaced by M'.
    """
    A_single_inv = np.linalg.inv(_A_SINGLE)
    B_inv = np.linalg.inv(B)
    # --- Type1： dictionary ---
    if isinstance(obj, dict):
        pack = obj
        n = pack["num_qubits"]
        #  A^{-1}
        if ainv_sparse:
            if not _HAS_SCIPY:
                raise ImportError("ainv_sparse=True need scipy.sparse")
            Ainv = _kron_power_sparse(A_single_inv, n)
        else:
            Ainv = _kron_power_dense(A_single_inv, n)

        mats_out = {}
        for key, GM in pack["matrices"].items():
            if GM.ndim != 2 or GM.shape[0] != GM.shape[1]:
                raise ValueError(f"GM for key {key} must be square; got {GM.shape}")
            c_key, t_key = key
            CXM, _ = cnot_pauli_twirling_matrix(n, c_key, t_key, sparse=cxm_sparse)
            # CXM^{-1} = CXM^T
            if issparse(CXM):
                left = (CXM.T).dot(B_inv)  
            else: 
                left = CXM.T @ B_inv
            if ainv_sparse:
                right = GM.dot(Ainv)  
            else:
                right = GM @ Ainv
            Mprime = left@right
            mats_out[key] = Mprime

        return {
            "num_qubits": n,
            "row_labels": pack["row_labels"],
            "col_labels": pack["col_labels"],
            "matrices": mats_out,
        }

    # --- Type2： ndarray ---
    if not isinstance(obj, np.ndarray):
        raise TypeError("obj must be dict of compute_tqg_matrices or numpy.ndarray")

    M = obj
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square; got shape={M.shape}")
    dim = M.shape[0]
    if n is None:
        n_est = round(np.log(dim) / np.log(4))
        if 4**n_est != dim:
            raise ValueError(f"cannot infer n: size {dim} is not a power of 4")
        n = n_est
    if c is None or t is None:
        raise ValueError("ndarray need c & t -> control/target of CNOT")

    #  A^{-1}
    if ainv_sparse:
        if not _HAS_SCIPY:
            raise ImportError("ainv_sparse=True need scipy.sparse")
        Ainv = _kron_power_sparse(A_single_inv, n)
    else:
        Ainv = _kron_power_dense(A_single_inv, n)

    # premultiply by CXM^-1, and postmultiply by A^-1
    CXM, _ = cnot_pauli_twirling_matrix(n, c, t, sparse=cxm_sparse)
    left = (CXM.T).dot(M) if issparse(CXM) else CXM.T @ M
    Mprime = left.dot(Ainv) if ainv_sparse else left @ Ainv
    return Mprime

# ----- Calculate Weight ----- 
def _labels_to_xz(labels: List[str]):
    X = np.zeros((len(labels), len(labels[0])), dtype=np.uint8)
    Z = np.zeros_like(X)
    for i, lab in enumerate(labels):
        x, z = _label_to_xz(lab)
        X[i, :] = x
        Z[i, :] = z
    return X, Z

def build_commutation_transform(labels: List[str]) -> np.ndarray:
    """
    Given diag = diag(M_avg)（length：4^n）and labels（row_labels order)
    Using symplectic geometry：parity = x·z' + z·x' (mod 2)。
    """
    X, Z = _labels_to_xz(labels)         # [N, n]
    # parity[b,a] == 0 & 1 represent P_b and P_a are commute & anticommute
    parity = (X @ Z.T - Z @ X.T) & 1           # [N, N]，element ∈ {0,1}
    T = (1 - 2 * parity).astype(np.int8)       # 0→+1, 1→-1
    return T                            

def compute_weights_for_package(avg_pack: Dict) -> Dict[Tuple[int,int], np.ndarray]:
    """
    avg_pack -> {'num_qubits': n, 'row_labels': [...], 'col_labels': [...], 'matrices': {(c,t): M_avg, ...}}
    return：{(c, t): weights[np.ndarray length 4^n] }
    """
    labels = avg_pack["row_labels"]
    T = build_commutation_transform(labels)
    out: Dict[Tuple[int,int], np.ndarray] = {}
    for key, Mavg in avg_pack["matrices"].items():
        d = Mavg.diagonal() if hasattr(Mavg, "diagonal") else np.diag(Mavg)
        d = np.asarray(d, dtype=float).reshape(-1)
        out[key] = T @ d
    return out

def compute_inv_weights_for_package(avg_pack: Dict, eps: float = 0.0) -> Dict[Tuple[int,int], np.ndarray]:
    """
    return：{(1/a) = T^{-1} (1/c)，其中 c = diag(M_avg)}
    """
    n = avg_pack["num_qubits"]
    labels = avg_pack["row_labels"]
    T = build_commutation_transform (labels)
    T_inv = T.astype(float) / (4.0 ** n)         # T^{-1} = T / 4^n

    out: Dict[Tuple[int,int], np.ndarray] = {}
    for key, Mavg in avg_pack["matrices"].items():
        c = Mavg.diagonal() if hasattr(Mavg, "diagonal") else np.diag(Mavg)
        c = np.asarray(c, dtype=float).reshape(-1)
        if eps > 0:
            s = np.sign(c); s[s == 0.0] = 1.0
            c = s * np.maximum(np.abs(c), eps)
        inv_c = 1.0 / c
        out[key] = T_inv @ inv_c
    return out