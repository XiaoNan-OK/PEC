import numpy as np
from itertools import product, islice
from typing import Dict, List, Tuple, Optional
from itertools import product, islice
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp

# ----------------- 基本工具 -----------------
# 1-1
def _ensure_quantum_circuit(obj):
    if isinstance(obj, QuantumCircuit):
        return obj
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], QuantumCircuit):
        return obj[0]
    clsname = obj.__class__.__name__
    raise TypeError(f"Expected QuantumCircuit, got {type(obj)} ({clsname}).")

# 2-1
def circuit_information(circuit: QuantumCircuit):
    circ = _ensure_quantum_circuit(circuit)
    used = set()
    cnots = []
    for k, inst in enumerate(circ.data):
        op = getattr(inst, "operation", inst[0])
        qargs = getattr(inst, "qubits", inst[1])
        for q in qargs:
            used.add(circ.find_bit(q).index)
        if op.name in ("cx", "cnot"):
            c = circ.find_bit(qargs[0]).index
            t = circ.find_bit(qargs[1]).index
            cnots.append({"data_idx": k, "control": c, "target": t})
    active = sorted(used)
    phys2active = {q: i for i, q in enumerate(active)}
    return {
        "active_qubits": active,
        "cnot_list": cnots,
        "phys2active": phys2active,  # <--- 新增這個 mapping
    }

# 3-1
def enum_pauli_labels(n: int) -> List[str]:
    from itertools import product
    return [''.join(p) for p in product(('I','X','Y','Z'), repeat=n)]

# --------------- Pauli label 運算 ---------------

_letter_to_xz = {'I':(0,0), 'X':(1,0), 'Y':(1,1), 'Z':(0,1)}
_xz_to_letter = {(0,0):'I', (1,0):'X', (1,1):'Y', (0,1):'Z'}

def _label_to_xz(label: str):
    x,z = [],[]
    for ch in label:
        xi, zi = _letter_to_xz[ch]
        x.append(xi); z.append(zi)
    return x,z

def _xz_to_label(x,z):
    return ''.join(_xz_to_letter[(xi,zi)] for xi,zi in zip(x,z))

def pauli_conj_by_cnot(label: str, c: int, t: int,  active: Optional[Dict[int,int]]=None) -> str:
    x,z = _label_to_xz(label)
    if active is not None:
        if c not in active or t not in active: return label
        c = active[c]; t = active[t]
    x[t] ^= x[c]
    z[c] ^= z[t]
    return _xz_to_label(x,z)

def apply_pauli_layer(qc: QuantumCircuit, label: str, active_qubits: Optional[List[int]] = None):
    if active_qubits is None:
        active_qubits = list(range(len(label)))
    if len(active_qubits) != len(label):
        raise ValueError(f"label 長度={len(label)} 和 active_qubits 長度={len(active_qubits)} 不一致")
    
    for i, ch in enumerate(label):
        if ch == 'I': continue
        q = active_qubits[i]
        if ch == 'X': qc.x(q)
        elif ch == 'Y': qc.y(q)
        elif ch == 'Z': qc.z(q)

def _prepare_idx2pair(orig: QuantumCircuit,
                      cnot_list: List[Dict],
                      label_strs: List[str],
                      phys2active: Optional[Dict[int, int]]=None) -> QuantumCircuit:
    idx2pair = {}
    for lab, rec in zip(label_strs, cnot_list):
        c = rec["control"]; t = rec["target"]
        post_lab = pauli_conj_by_cnot(lab, c, t, active=phys2active)
        idx2pair[rec["data_idx"]] = (lab, post_lab)
    return idx2pair

# --------------- Sampler 測量期望值 ---------------

def _rotate_to_Z_for_label(qc: QuantumCircuit, q: int, ch: str):
    if ch == 'X': qc.h(q)
    elif ch == 'Y': qc.sdg(q); qc.h(q)

def _exp_from_distribution(dist) -> float:
    # dist 可能是 Counts 或 dict；normalize + parity
    try: items = dist.items()
    except Exception: dist = dict(dist); items = dist.items()
    total = float(sum(float(v) for _,v in items))
    if total <= 0: return 1.0
    exp = 0.0
    for key, v in items:
        p = float(v)/total
        if isinstance(key, str):
            ones = key.replace(' ','').count('1')
        else:
            ones = int(key).bit_count()
        exp += p * (-1.0 if (ones & 1) else 1.0)
    return float(exp)

def _extract_quasi_or_counts_list(res):
    d=getattr(res,"data",None)
    if d is not None:
        c=getattr(d,"c_meas",None)
        if c and hasattr(c,"get_counts"): return [c.get_counts()]
        if hasattr(d,"quasi_dists"): return list(d.quasi_dists)
        if hasattr(d,"meas") and hasattr(d.meas,"quasi_dists"): return list(d.meas.quasi_dists)
    rs=getattr(res,"results",None)
    if rs and hasattr(rs[0],"data"):
        dd=rs[0].data
        if hasattr(dd,"quasi_dists"): return list(dd.quasi_dists)
    try:
        out=[]
        for i in range(len(res)):
            out.extend(_extract_quasi_or_counts_list(res[i]))
        if out: return out
    except Exception: pass
    raise TypeError("Unsupported SamplerResult")

# --------------- Twirl & 測量建構 ---------------

def _twirled_variant_for_all_cnot(orig: QuantumCircuit,
                                  idx2pair: Optional[Dict[str, Tuple[str, str]]] = None,
                                  active_qubits: Optional[List[int]] = None,) -> QuantumCircuit:
    qc = QuantumCircuit(orig.num_qubits, orig.num_clbits, name=f"{orig.name or 'circ'}|twirl_all")
    for k, inst in enumerate(orig.data):
        op = getattr(inst, "operation", inst[0])
        qargs = getattr(inst, "qubits", inst[1])
        cargs = getattr(inst, "clbits", [])
        pair = idx2pair.get(k) if idx2pair is not None else None
        if pair is None:
            qc.append(op, qargs, cargs)
        else:
            pre_lab, post_lab = pair
            apply_pauli_layer(qc, pre_lab, active_qubits)
            qc.append(op, qargs, cargs)
            apply_pauli_layer(qc, post_lab, active_qubits)
    return qc

def _meas_circuit_for_observable(base: QuantumCircuit, obs: SparsePauliOp, active_qubits: Optional[List[int]] = None) -> Optional[QuantumCircuit]:
    """
    針對 obs（假設只有一項 Pauli）建立「旋到 Z + 量測」的電路。
    - 若 obs 是兩體（長度 2 的 label）且提供 meas_qubits=(qA,qB)：
        * 'II' → 回傳 None（代表期望值恆 +1）
        * 單邊 I → 只量測非 I 那條
        * 都非 I → 量測兩條
    - 否則 fallback：對所有非 I 的位旋轉並量測（動態配置 clbits）
    """
    # 從 SparsePauliOp 拿 label（假設只有一項）
    label=obs.to_list()[0][0]
    pauli=label[::-1]

    if active_qubits is None or len(pauli) == base.num_qubits:
        # 視為直接在物理 qubit 上
        qubit_map = list(range(len(pauli)))
    else:
        if len(pauli) != len(active_qubits):
            raise ValueError("observable label 長度和 active_qubits 長度不符")
        qubit_map = active_qubits

    act = [qubit_map[i] for i, ch in enumerate(pauli) if ch != 'I']
    if len(act) == 0: return None

    qc=base.copy()
    # 旋到 Z
    for i, ch in enumerate(pauli):
        if ch == 'I': continue
        q = qubit_map[i]
        _rotate_to_Z_for_label(qc, q, ch)
    # 加測量（把所有 non-I qubit 量測到連續的 classical bits 上）
    creg = ClassicalRegister(len(act), "c_meas")
    qc.add_register(creg)
    for ci, q in enumerate(act): qc.measure(q, creg[ci])
    return qc

# --------------- Sampler 版：單 observable 的 PEC 期望值 ---------------

def tqg_pec_batch_circuit_sampler(
    sampler,
    pending_circs: List[QuantumCircuit],
    pending_meta: List[Tuple[str, float]],
    results: Dict[str, float],
    *,
    backend=None,
    opt_level: int = 0,
    shots: int = 1024           
):
    """
    一次把 pending_circs 丟進 sampler.run，根據 meta 把結果加回 results。
    pending_meta: 每個元素是 (obs_name, w_prod)
    """
    if not pending_circs: return

    # 先 transpile（如果有 backend）
    if backend is not None: tr_circs = transpile(pending_circs, backend=backend, optimization_level=opt_level)
    else: tr_circs = pending_circs

    job = sampler.run(tr_circs, shots=shots)
    res = job.result()
    dists = _extract_quasi_or_counts_list(res)

    if len(dists) != len(pending_circs):
        # 和前面一樣，做 fallback：有些實作要 res[i] 取出才看得到 quasi_dists
        try:
            new_dists = []
            for i in range(len(pending_circs)):
                new_dists.extend(_extract_quasi_or_counts_list(res[i]))
            dists = new_dists
        except Exception:
            raise ValueError(f"Sampler returned {len(dists)} dists but expected {len(pending_circs)}")

    for dist, (obs_name, w_prod) in zip(dists, pending_meta):
        results[obs_name] += w_prod * _exp_from_distribution(dist)

    pending_circs.clear()
    pending_meta.clear()

# --------------- Sampler 版：批次跑多個 observable ---------------

def run_tqg_pec_package_sampler(
    sampler,
    qcircuit: QuantumCircuit,
    observables: Dict[str, SparsePauliOp],
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    backend=None,
    opt_level: int = 0,
    shots: int = 1024, #CPU 迴圈 chunk
    combo_batch_size: int = 64, #Sampler 真正 batch
    max_batch: int = 1024,
    verbose: bool = True
) -> Dict[str, Dict]:
    # --- 前置準備（不變） ---
    circ = _ensure_quantum_circuit(qcircuit)
    info = circuit_information(circ)

    active_qubits = info["active_qubits"]
    n_active = len(active_qubits)
    pre_labels = enum_pauli_labels(n_active)
    N = len(pre_labels)  
    n_cnot = len(info["cnot_list"])

    # 檢查 weights 維度
    for rec in info["cnot_list"]:
        key = (rec["control"], rec["target"])
        w = tqg_weights.get(key)
        if w is None or len(w) != N:
            raise ValueError(f"weights_map[{key}] length error.")

    # --- 主迴圈：把每一組 (idx_tuple, w_prod, idx2pair) 都算出來 ---
    results = {name: 0.0 for name in observables}
    total = N ** n_cnot
    done = 0
    it = product(range(N), repeat=n_cnot)

    # 用來累積「要一次丟進 sampler」的電路
    pending_circs: List[QuantumCircuit] = []
    pending_meta:  List[Tuple[str, float]] = []   # (obs_name, w_prod)

    while True:
        batch = list(islice(it, combo_batch_size))
        if not batch: break

        # 先算好這一批共用的 idx2pair & w_prod
        for idx_tuple in batch:
            labs = [pre_labels[i] for i in idx_tuple]
            w_prod = 1.0
            for j, i_lab in enumerate(idx_tuple):
                rec = info["cnot_list"][j]
                key = (rec["control"], rec["target"])
                w_prod *= float(tqg_weights[key][i_lab])
            idx2pair = _prepare_idx2pair(circ, info["cnot_list"], labs, phys2active=info["phys2active"])
            twirl_circ = _twirled_variant_for_all_cnot(circ, idx2pair=idx2pair, active_qubits=active_qubits)
            print(twirl_circ)

            for name, obs in observables.items():
                qc = _meas_circuit_for_observable(twirl_circ, obs, active_qubits)
                if qc is None:
                    # observable 是 I...，期望值恆為 +1
                    results[name] += w_prod
                    continue
                pending_circs.append(qc)
                pending_meta.append((name, w_prod))

                if len(pending_circs) >= max_batch:
                    tqg_pec_batch_circuit_sampler(sampler, pending_circs, pending_meta, results, backend=backend, opt_level=opt_level, shots=shots)

        done += len(batch)
        if verbose and done % (total // 10 or 1) == 0:
            print(f"{done}/{total} ({100*done/total:.1f}%)")

    # 迴圈跑完後，如果還有殘留沒 flush 的，就最後再跑一次
    if pending_circs:
        tqg_pec_batch_circuit_sampler(sampler, pending_circs, pending_meta, results, backend=backend, opt_level=opt_level, shots=shots)
    
    return {name: {"obs": name, "value": float(val)} for name, val in results.items()}
