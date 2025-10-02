import numpy as np
import os, json, time
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

def _compact_qubits(circ: QuantumCircuit) -> QuantumCircuit:
    """Return a copy of circ that keeps only used qubits, remapped densely to [0..k-1]."""
    used = _used_qubit_indices(circ)
    if len(used) == circ.num_qubits:
        return circ  # nothing to drop
    mapping = {old: new for new, old in enumerate(used)}
    new = QuantumCircuit(len(used), circ.num_clbits, name=circ.name)
    for inst in circ.data:
        try:
            op, qargs, cargs = inst.operation, inst.qubits, inst.clbits
        except AttributeError:
            op, qargs, cargs = inst
        new_q = []
        for q in qargs:
            old_idx = circ.find_bit(q).index
            if old_idx in mapping:
                new_q.append(new.qubits[mapping[old_idx]])
        new.append(op, new_q, cargs)
    return new

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
def transform_tqg_matrices_cxm_Ainv(obj,
                       *,
                       n: int | None = None,
                       c: int | None = None,
                       t: int | None = None,
                       cxm_sparse: bool = True,
                       ainv_sparse: bool = False):
    """
    Apply a transformation to the output of `compute_tqg_matrices(...)`:
    M' = CXM^{-1} * GM * A^{-1}
Details:
    - CXM:
        Obtained from `cnot_pauli_twirling_matrix(n, c, t, sparse=cxm_sparse)`.
        Since the PTM of a CNOT is a permutation matrix, CXM^{-1} = CXM^T.
    - A^{-1}:
        First compute A_single^{-1} = inv(_A_SINGLE), then take the Kronecker power
        to n qubits.
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
            left = (CXM.T).dot(GM) if issparse(CXM) else CXM.T @ GM
            Mprime = left.dot(Ainv) if ainv_sparse else left @ Ainv
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


# --------- (7) Calculate average expectation matrix of each cnot ----------
@lru_cache(maxsize=8)
def _get_pm(backend, opt_level: int):
    """
    依 backend 與 opt_level 產生並快取一個 preset pass manager。
    - opt_level 建議用 2~3；0 也可（幾乎不優化）
    - backend 用同一個（ex: AerSimulator.from_backend(device)）
    """
    try:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=opt_level)
        return pm
    except Exception:
        return None

# ---- Build a twirled variant that inserts Pauli layers for ALL CNOT instances ----
def _twirled_variant_for_all_cnot(orig: QuantumCircuit,
                                  cnot_list: List[Dict],
                                  label_strs: List[str]) -> QuantumCircuit:
    """
    Return a circuit where, for each CNOT instance (in the order of cnot_list),
    we insert a pre-Pauli before and a post-Pauli after that CNOT.
    - label_strs[j] is the n-qubit Pauli label (e.g., 'IXYZ...') assigned to cnot_list[j].
    - post-Pauli is computed via conjugation through that CNOT.
    """
    n = orig.num_qubits
    qc = QuantumCircuit(n, orig.num_clbits, name=f"{orig.name or 'circ'}|twirl_all")

    # Map data_idx -> (pre_label, post_label)
    idx2pair = {}
    for lab, rec in zip(label_strs, cnot_list):
        c = rec["control"]; t = rec["target"]
        post_lab = pauli_conj_by_cnot(lab, c, t)
        idx2pair[rec["data_idx"]] = (lab, post_lab)

    # Rebuild instruction-by-instruction, inserting Pauli layers around marked CNOTs
    for k, inst in enumerate(orig.data):
        try:
            op = inst.operation; qargs = inst.qubits; cargs = inst.clbits
        except AttributeError:
            op, qargs, cargs = inst

        pair = idx2pair.get(k)
        if pair is None:
            qc.append(op, qargs, cargs)
        else:
            pre_lab, post_lab = pair
            apply_pauli_layer(qc, pre_lab)
            qc.append(op, qargs, cargs)
            apply_pauli_layer(qc, post_lab)
    return qc


# ---- Correct TQG-PEC: joint enumeration over ALL CNOT Pauli assignments ----
def tqg_pec_single_circuit_joint(estimator,
                                 circuit: QuantumCircuit,
                                 observable: SparsePauliOp,
                                 weights_map: Dict[Tuple[int, int], np.ndarray],
                                 *,
                                 combo_batch_size: int = 64,
                                 t_backend=None,
                                 t_opt_level: int = 0,
                                 progress: bool = False) -> float:
    """
    Correct PEC estimator for a single circuit and observable by JOINT enumeration:
      PEC = sum_{(p_1,...,p_m)} [ prod_{i=1..m} w_i(p_i) ] * <O>_{U(p_1,...,p_m)},
    where m = number of CNOT instances in the circuit, and U(p_1,...,p_m) inserts
    pre/post Pauli layers around EVERY CNOT according to the chosen labels.

    - weights_map: {(c,t): weights} with length 4^n each, in the order of enum_pauli_labels(n).
    - combo_batch_size: how many joint combinations (circuits) to evaluate per estimator.run() call.
    """
    info = cnot_pec(circuit)   # {'num_qubits': n, 'cnot_list': [{'data_idx','control','target'}, ...]}
    n = info["num_qubits"]
    cnot_list = info["cnot_list"]
    m = len(cnot_list)

    # No CNOTs: just evaluate once
    if m == 0:
        cir = circuit
        if t_backend is not None:
            # lock layout to only the qubits actually used by the circuit
            layout = _used_qubit_indices(cir) or list(range(cir.num_qubits))
            cir = transpile(
                [cir],
                backend=t_backend,
                initial_layout=layout,
                layout_method="trivial",
                routing_method="none",
                optimization_level=t_opt_level,
            )[0]
            cir = _compact_qubits(cir)
        ev = _estimator_expect_many(estimator, cir, [observable])[0]
        return float(ev)

    pre_labels = enum_pauli_labels(n)   # length = 4^n
    N = len(pre_labels)

    # Sanity checks on weights
    for rec in cnot_list:
        key = (rec["control"], rec["target"])
        w = weights_map.get(key)
        if (w is None) or (len(w) != N):
            raise ValueError(f"weights_map key={key} missing or length != 4^n={N}")

    # Iterator over all joint assignments: each CNOT picks an index in [0..N-1]
    it = product(range(N), repeat=m)
    total_combos = N**m

    acc = 0.0
    processed = 0

    while True:
        # Take a batch of joint assignments
        batch_idx_tuples = list(islice(it, combo_batch_size))
        if not batch_idx_tuples:
            break

        circs = []
        weight_batch = []

        for idx_tuple in batch_idx_tuples:
            labels = [pre_labels[i] for i in idx_tuple]
            w_prod = 1.0
            for j, i_lab in enumerate(idx_tuple):
                rec = cnot_list[j]
                key = (rec["control"], rec["target"])
                w_prod *= float(weights_map[key][i_lab])
            tw = _twirled_variant_for_all_cnot(circuit, cnot_list, labels)
            circs.append(tw)
            weight_batch.append(w_prod)

        #  Process the entire batch at once
        if t_backend is not None:
            # derive a fixed layout from the original (pre-twirl) circuit’s used lines
            layout = _used_qubit_indices(circuit) or list(range(circuit.num_qubits))
            circs = transpile(
                circs,
                backend=t_backend,
                initial_layout=layout,
                layout_method="trivial",
                routing_method="none",
                optimization_level=t_opt_level,
            )
            # drop any idle lines the transpiler might have kept
            circs = [_compact_qubits(c) for c in circs]

        # sent the whole batch to Estimator at once
        pubs = [(cir, [observable], None) for cir in circs]
        res = estimator.run(pubs).result()

        # Extract expectation values in a compatible way (avoid using res[j])
        evs = _extract_evs_from_result(res)   # This returns something like [ev0, ev1, ...]
        # Safety check: some versions may only return the first result, so fall back to per-circuit extraction
        if evs.shape[0] != len(circs):
            # Fallback: evaluate circuit by circuit
            evs = []
            for cir in circs:
                evs.append(_estimator_expect_many(estimator, cir, [observable])[0])
            evs = np.asarray(evs, dtype=float)

        # Weighted accumulation
        acc += float(np.dot(weight_batch, evs))

        processed += len(circs)

        if progress and processed % (10 * combo_batch_size) == 0:
            pct = 100.0 * processed / total_combos
            print(f"[JOINT] {processed}/{total_combos} combos done ({pct:.2f}%)")
        # Free batch objects
        del circs, pubs, res, weight_batch

    return float(acc)

def _iter_nested_package(qc_bzls: Dict) -> Iterable[Tuple[str, int, int, int, QuantumCircuit]]: 
    """ 
    迭代你給的資料結構：Bz -> [time 列表] -> [carbon set 列表] -> [circuits 列表]。 
    逐一回傳 (Bz, t_idx, c_idx, circ_idx, circuit)。 
    """ 
    for Bz, t_lists in qc_bzls.items(): # 4 個 Bz 
        for t_idx, c_lists in enumerate(t_lists): # t 個時間點 
            for c_idx, circ in enumerate(c_lists): # c 組 carbon set 
                yield Bz, t_idx, c_idx, circ 
                
def _ckpt_key(Bz: str, t_idx: int, c_idx: int, obs_name: str) -> str: 
    return f"{Bz}|t{t_idx}|c{c_idx}|{obs_name}" 

def _load_done_set(ckpt_path: str) -> Dict[str, Dict]: 
    done = {} 
    if os.path.exists(ckpt_path): 
        with open(ckpt_path, "r") as f: 
            for line in f: 
                try: 
                    rec = json.loads(line) 
                    done[rec["key"]] = rec 
                except Exception: 
                    continue 
    return done

def run_tqg_pec_package(estimator,
                        qc_bzls: Dict,
                        observables: Dict[str, SparsePauliOp],
                        weights_map: Dict[Tuple[int, int], np.ndarray],
                        *,
                        combo_batch_size: int = 64,  # NOTE: joint-combo batch size
                        ckpt_path: Optional[str] = None,
                        resume: bool = True,
                        t_backend=None,
                        t_opt_level: int = 0,
                        verbose: bool = True) -> Dict[str, Dict]:
    """
    Run JOINT Pauli enumeration TQG-PEC over the entire dataset.
    - observables: e.g., {'X': measurements['measX'], 'Y': measurements['measY']}
    - weights_map: {(c,t): weights} with length 4^n, order matching enum_pauli_labels(n)
    - ckpt_path: JSONL; after finishing each (circuit, observable), append one line (resumable).
    """
    if ckpt_path is None:
        ckpt_path = f"tqg_pec_ckpt_{int(time.time())}.jsonl"

    done_map = _load_done_set(ckpt_path) if resume else {}
    results: Dict[str, Dict] = dict(done_map)  # preload completed entries

    total_units = sum(1 for _ in _iter_nested_package(qc_bzls)) * len(observables)
    finished = len(done_map)
    if verbose:
        print(f"[RUN] total tasks = {total_units}, already done = {finished}, resume={resume}, ckpt='{ckpt_path}'")

    mode = "a" if resume else "w"
    with open(ckpt_path, mode) as fout:
        unit = 0
        for Bz, t_idx, c_idx, circ in _iter_nested_package(qc_bzls):
            for obs_name, observable in observables.items():
                key = _ckpt_key(Bz, t_idx, c_idx, obs_name)
                unit += 1
                if resume and key in done_map:
                    if verbose and unit % 50 == 0:
                        print(f"[SKIP] {unit}/{total_units}  {key}")
                    continue

                try:
                    circ = _ensure_quantum_circuit(circ)
                except TypeError as e:
                    raise TypeError(f"[BAD ITEM] key={key}  {e}")

                # === The only line that changes: call the JOINT version ===
                val = tqg_pec_single_circuit_joint(
                    estimator, circ, observable, weights_map,
                    combo_batch_size=combo_batch_size,
                    t_backend=t_backend, t_opt_level=t_opt_level,
                    progress=True
                )

                rec = {"key": key, "Bz": Bz, "t": t_idx, "c": c_idx+1,
                       "obs": obs_name, "value": float(val)}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                results[key] = rec

                if verbose and unit % 10 == 0:
                    pct = 100.0 * unit / total_units
                    print(f"[PROGRESS] {unit}/{total_units} ({pct:5.1f}%)  {key}  → value={val:.6f}")

    if verbose:
        print("[DONE] results saved in", ckpt_path)
    return results
