import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import SparsePauliOp
from itertools import product
from typing import Dict, List, Tuple, Optional

# ==============================================================
# (1) Utility: 確認電路與使用 qubit
# ==============================================================

def _ensure_quantum_circuit(obj):
    if isinstance(obj, QuantumCircuit):
        return obj
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], QuantumCircuit):
        return obj[0]
    clsname = obj.__class__.__name__
    raise TypeError(f"Expected QuantumCircuit, got {type(obj)} ({clsname}).")

def _used_qubit_indices(circ: QuantumCircuit) -> List[int]:
    """取得整個電路中實際用到的 qubit 索引"""
    used = set()
    for inst in circ.data:
        qargs = getattr(inst, "qubits", inst[1])
        for q in qargs:
            used.add(circ.find_bit(q).index)
    return sorted(used)

def cnot_pec(input_circuit: QuantumCircuit):
    circ = _ensure_quantum_circuit(input_circuit)
    cnots = []
    for k, inst in enumerate(circ.data):
        op = getattr(inst, "operation", inst[0])
        qargs = getattr(inst, "qubits", inst[1])
        if op.name in ("cx", "cnot"):
            c = circ.find_bit(qargs[0]).index
            t = circ.find_bit(qargs[1]).index
            cnots.append({"data_idx": k, "control": c, "target": t})
    return {"num_qubits": circ.num_qubits, "cnot_list": cnots}

def unique_cnot_keys(input_circuit: QuantumCircuit):
    info = cnot_pec(input_circuit)
    seen, out = set(), []
    for rec in info["cnot_list"]:
        key = (rec["control"], rec["target"])
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out

# ==============================================================
# (2) 基底與 Observable
# ==============================================================

def enum_initial_labels(n: int) -> List[str]:
    return [''.join(p) for p in product(('0','1','+','R'), repeat=n)]

def enum_pauli_labels(n: int) -> List[str]:
    return [''.join(p) for p in product(('I','X','Y','Z'), repeat=n)]

def init_circuit_from_label(n: int, label: str) -> QuantumCircuit:
    qc = QuantumCircuit(n, name=f"init_{label}")
    for q, ch in enumerate(label):
        if ch == '1': qc.x(q)
        elif ch == '+': qc.h(q)
        elif ch == 'R': qc.h(q); qc.s(q)
    return qc

def make_observable_map(labels: List[str]) -> Dict[str, SparsePauliOp]:
    return {lab: SparsePauliOp.from_list([(''.join(lab[::-1]), 1.0)]) for lab in labels}

# ==============================================================
# (3) Pauli label operations
# ==============================================================

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

def pauli_conj_by_cnot(label: str, c: int, t: int) -> str:
    x,z = _label_to_xz(label)
    x[t] ^= x[c]
    z[c] ^= z[t]
    return _xz_to_label(x,z)

def apply_pauli_layer(qc: QuantumCircuit, label: str):
    for q, ch in enumerate(label):
        if ch == 'I': continue
        if ch == 'X': qc.x(q)
        elif ch == 'Y': qc.y(q)
        elif ch == 'Z': qc.z(q)

# ==============================================================
# (4) Sampler expectation helpers
# ==============================================================

def _rotate_to_Z_for_label(circ: QuantumCircuit, label: str):
    for q, ch in enumerate(label):
        if ch == 'X': circ.h(q)
        elif ch == 'Y': circ.sdg(q); circ.h(q)

def _attach_measure_active_only(circ: QuantumCircuit, active: List[int]):
    if len(active)==0:
        circ.add_register(ClassicalRegister(0,"c"))
        return
    creg = ClassicalRegister(len(active),"c")
    circ.add_register(creg)
    for ci,q in enumerate(active):
        circ.measure(q,creg[ci])

def _extract_quasi_or_counts_list(samp_res):
    """
    盡可能涵蓋各種 Sampler 版本結構，回傳 [dist, dist, ...]。
    每個 dist 是 dict：bitstring -> prob 或 count（若是 count，我們會在 _exp_from_distribution 規一化）。
    支援：
      - 你的環境：res = sampler.run(circs).result()[i]; res.data.c.get_counts()
      - V2 PrimitiveResult：for rec in res: rec.data.meas.quasi_dists 或 rec.data.quasi_dists
      - V1：res.quasi_dists / res.quasi_distributions
      - Runtime：res.results[...].data.quasi_dists
    """
    # ---- case A: 你現在的寫法：已經先 [i] 取出一筆 PubResult 類型 ----
    # 例如：res = sampler.run([...]).result()[0]
    #       res.data.c.get_counts()
    data = getattr(samp_res, "data", None)
    if data is not None:
        # 先試 counts：data 可能有一個或多個 classical register（常見名稱 'c'）
        # 優先嘗試你示例的 .data.c.get_counts()
        cobj = getattr(data, "c", None)
        if cobj is not None and hasattr(cobj, "get_counts"):
            return [cobj.get_counts()]

        # 其次嘗試把 data 當作 dict-like 的多暫存器
        # e.g. data.<any_reg>.get_counts()
        for name in dir(data):
            if name.startswith("_") or name in ("meas", "quasi_dists"):
                continue
            reg = getattr(data, name, None)
            if hasattr(reg, "get_counts"):
                try:
                    cnts = reg.get_counts()
                    return [cnts]
                except Exception:
                    pass

        # 再來試 quasi_dists（V2）
        if hasattr(data, "quasi_dists"):
            return list(data.quasi_dists)
        if hasattr(data, "meas") and hasattr(data.meas, "quasi_dists"):
            return list(data.meas.quasi_dists)

        # dict 風格
        if isinstance(data, dict):
            if "quasi_dists" in data:
                return list(data["quasi_dists"])
            if "meas" in data and isinstance(data["meas"], dict) and "quasi_dists" in data["meas"]:
                return list(data["meas"]["quasi_dists"])

    # ---- case B: 結果是 iterable（多筆 PubResult）----
    try:
        if hasattr(samp_res, "__iter__") and not isinstance(samp_res, (dict, str)):
            outs = []
            for rec in list(samp_res):
                outs.extend(_extract_quasi_or_counts_list(rec))
            if outs:
                return outs
    except Exception:
        pass

    # ---- case C: V1 SamplerResult ----
    for attr in ("quasi_dists", "quasi_distributions"):
        qd = getattr(samp_res, attr, None)
        if qd is not None:
            return list(qd)

    # ---- case D: Runtime 結構 ----
    rs = getattr(samp_res, "results", None)
    if rs and hasattr(rs[0], "data"):
        d = rs[0].data
        if hasattr(d, "quasi_dists"):
            return list(d.quasi_dists)
        if hasattr(d, "meas") and hasattr(d.meas, "quasi_dists"):
            return list(d.meas.quasi_dists)

    raise TypeError(f"Unsupported SamplerResult format: {type(samp_res)}")

def _exp_from_distribution(dist) -> float:
    """
    dist: dict -> {bitstring (str or int): prob 或 count}
    自動正規化（如果是 counts），再用奇偶數規則算 E[Z^{⊗k}].
    """
    total = sum(dist.values())
    if total <= 0:
        return 1.0
    exp = 0.0
    for key, v in dist.items():
        # 規一化
        p = v / total
        # bitstring 可能是 '0110' 或 int
        if isinstance(key, str):
            ones = key.count('1')
        else:
            ones = int(key).bit_count()
        ev = -1.0 if (ones & 1) else 1.0
        exp += p * ev
    return float(exp)

def _exp_from_quasi_dist(qd) -> float:
    val=0.0
    for k,p in qd.items():
        ones = k.count('1') if isinstance(k,str) else bin(int(k)).count('1')
        val += p * ((-1)**ones)
    return float(val)

def _extract_quasi_dists(samp_res):
    for attr in ("records","__iter__"):
        if hasattr(samp_res,attr):
            try:
                recs = list(getattr(samp_res,"records",samp_res))
            except Exception:
                recs=None
            if recs:
                out=[]
                for r in recs:
                    d = getattr(r,"data",None)
                    if hasattr(d,"quasi_dists"): out.extend(list(d.quasi_dists))
                if out: return out
    for attr in ("quasi_dists","quasi_distributions"):
        qd=getattr(samp_res,attr,None)
        if qd is not None: return list(qd)
    rs=getattr(samp_res,"results",None)
    if rs and hasattr(rs[0],"data") and getattr(rs[0].data,"quasi_dists",None):
        return list(rs[0].data.quasi_dists)
    raise TypeError("Unsupported SamplerResult")

# ==============================================================
# (5) Active-label <-> n-label 擴張工具
# ==============================================================

def _expand_label_to_n(label_m: str, active: list[int], n: int, fill: str) -> str:
    out=[fill]*n
    for ch,q in zip(label_m,active): out[q]=ch
    return ''.join(out)

def _local_indices(active: list[int], c: int, t: int):
    to_loc={g:i for i,g in enumerate(active)}
    return to_loc[c],to_loc[t]

# ==============================================================
# (6) Build twirled circuit on n lines (繞過未參與線)
# ==============================================================

def build_twirled_circuit_bypass_n(n: int, init_label_m: str, pre_label_m: str,
                                   active: list[int], c: int, t: int) -> QuantumCircuit:
    c_loc,t_loc=_local_indices(active,c,t)
    init_label_n=_expand_label_to_n(init_label_m,active,n,fill='0')
    pre_label_n=_expand_label_to_n(pre_label_m,active,n,fill='I')
    post_label_m=pauli_conj_by_cnot(pre_label_m,c_loc,t_loc)
    post_label_n=_expand_label_to_n(post_label_m,active,n,fill='I')

    qc=QuantumCircuit(n,name=f"twirl(c{c},t{t})")
    qc.compose(init_circuit_from_label(n,init_label_n),inplace=True)
    apply_pauli_layer(qc,pre_label_n)
    qc.cx(c,t)
    apply_pauli_layer(qc,post_label_n)
    return qc

# ==============================================================
# (7) Sampler expectation on active labels
# ==============================================================

def _build_meas_circuit_for_observable_n(base_n: QuantumCircuit, obs_label_m: str, active: list[int]):
    n=base_n.num_qubits
    obs_label_n=_expand_label_to_n(obs_label_m,active,n,fill='I')
    qc=QuantumCircuit(n,name=f"{base_n.name}__meas_{obs_label_m}")
    qc.compose(base_n,inplace=True)
    _rotate_to_Z_for_label(qc,obs_label_n)
    act_meas=[q for q in active if obs_label_n[q]!='I']
    _attach_measure_active_only(qc,act_meas)
    return qc

def _flush_sampler_batch(
    sampler,
    circs: list[QuantumCircuit],
    meta: list[Tuple[int,int]],
    acc: np.ndarray,
    shots: int
):
    """
    sampler.run(circs) 一次跑掉，根據 meta 把期望值加回 acc。
    meta 裡面每個元素是 (pre_idx, obs_idx)，這裡實際上只用到 obs_idx 來累加。
    """
    if not circs:
        return

    res = sampler.run(circs, shots=shots).result()
    dists = _extract_quasi_or_counts_list(res)

    if len(dists) != len(circs):
        # 有些 Sampler 結構要先對 result 做 [i] 再解 quasi_dists / counts
        try:
            new_dists = []
            for i in range(len(circs)):
                new_dists.extend(_extract_quasi_or_counts_list(res[i]))
            dists = new_dists
        except Exception:
            raise ValueError(f"Sampler returned {len(dists)} distributions but expected {len(circs)}")

    # 將每一個 distribution 轉成期望值，累加到對應的 obs_idx
    for qd, (pre_idx, obs_idx) in zip(dists, meta):
        val = _exp_from_distribution(qd)
        acc[obs_idx] += val

    circs.clear()
    meta.clear()

# ==============================================================
# (8) Compute matrix for each CNOT
# ==============================================================

def compute_matrix_for_tqg(
    sampler,
    input_circuit: QuantumCircuit,
    key: Tuple[int,int],     # 這個就可以先閒置不用，或之後拿掉
    shots: int = 1024,
    max_batch: int = 1024      # 新增：一次 sampler.run 最多丟幾個電路
):
    n = input_circuit.num_qubits
    print("num_qubits:", n)
    active = _used_qubit_indices(input_circuit)
    m = len(active)
    c, t = key

    init_labels_m = enum_initial_labels(m)
    obs_labels_m  = enum_pauli_labels(m)
    pre_labels_m  = enum_pauli_labels(m)

    M = np.zeros((len(obs_labels_m), len(init_labels_m)), dtype=float)
    for j, init_lab_m in enumerate(init_labels_m):
        acc = np.zeros(len(obs_labels_m), dtype=float)
        pending_circs: List[QuantumCircuit] = []
        pending_meta:  List[Tuple[int,int]] = []   # (pre_idx, obs_idx)
        # 掃過所有 pre_label
        for pre_idx, pre_m in enumerate(pre_labels_m):
            # 先建這個 init/pre 對應的 twirled base circuit（尚未加測量）
            qc_base = build_twirled_circuit_bypass_n(n, init_lab_m, pre_m, active, c, t)
            # 再掃過所有 observable label
            for obs_idx, obs_lab_m in enumerate(obs_labels_m):
                # 如果觀察算子是全 I，對任何 pre 的期望值都是 1
                if all(ch == 'I' for ch in obs_lab_m):
                    acc[obs_idx] += 1.0
                    continue
                # 否則要為這個 (init, pre, obs) 建測量電路
                qc_meas = _build_meas_circuit_for_observable_n(qc_base, obs_lab_m, active)
                pending_circs.append(qc_meas)
                pending_meta.append((pre_idx, obs_idx))

                # 如果這一批達到 max_batch，就一次送 sampler 跑掉
                if len(pending_circs) >= max_batch: _flush_sampler_batch(sampler, pending_circs, pending_meta, acc, shots=shots)

        # 跑完所有 pre / obs 組合後，如果還有剩的電路，最後 flush 一次
        if pending_circs: _flush_sampler_batch(sampler, pending_circs, pending_meta, acc, shots=shots)

        # 對所有 pre_label 取平均
        M[:, j] = acc / float(len(pre_labels_m))

        print(f"(c{c},t{t}) {j+1}/{len(init_labels_m)} (active={active})")

    return M, obs_labels_m, init_labels_m

def compute_tqg_matrices(sampler, input_circuit: QuantumCircuit, obs_batch:int=256, shots:int=1024):
    active=_used_qubit_indices(input_circuit)
    m=len(active)
    mats={}
    row_labels=col_labels=None
    for key in unique_cnot_keys(input_circuit):
        M,rlabs,clabs=compute_matrix_for_tqg(sampler,input_circuit,key,obs_batch,shots)
        mats[key]=M
        row_labels, col_labels = rlabs, clabs
    return {"num_qubits":m,"row_labels":row_labels,"col_labels":col_labels,"matrices":mats, "active_qubits":active}

# ==============================================================
# (9) Pauli twirling matrix + A/B inverses + weight calculation
# ==============================================================

_A_SINGLE=np.array([[1,1,1,1],[0,0,1,0],[0,0,0,1],[1,-1,0,0]],float)

def cnot_pauli_twirling_matrix(n:int,c_loc:int,t_loc:int):
    N=4**n
    labels=enum_pauli_labels(n)
    idx={lab:i for i,lab in enumerate(labels)}
    rows,cols=[],[]
    for j,lab in enumerate(labels):
        label2=pauli_conj_by_cnot(lab,c_loc,t_loc)
        i=idx[label2]; rows.append(i); cols.append(j)
    M=np.zeros((N,N)); M[rows,cols]=1.0
    return M, labels

def _kron_power_dense(mat,n:int):
    out=mat
    for _ in range(n-1): out=np.kron(out,mat)
    return out

def _to_local_indices(active: list[int], c: int, t: int) -> tuple[int,int]:
    to_loc = {g:i for i,g in enumerate(active)}
    return to_loc[c], to_loc[t]

def averaged_pauli_twirling_matrix(pack: dict, B: np.ndarray, *, active_qubit=None):
    """
    給 compute_tqg_matrices(...) 的輸出 pack（其中包含 active_qubits）與 B 矩陣，
    回傳：M' = CXM_active^{-1} * B^{-1} * G * A^{-1}
    每個 CNOT key 都是一個 4^m × 4^m 矩陣。
    """
    # ---- 取出基本資訊 ----
    m = pack["num_qubits"]
    if active_qubit is None:
        active = pack.get("active_qubits")
        if active is None:
            raise ValueError("pack 缺少 'active_qubits' 欄位")
    # ---- 建立 A⁻¹ 與 B⁻¹ ----
    Ainv_1q = np.linalg.inv(_A_SINGLE)
    Ainv = _kron_power_dense(Ainv_1q, m)
    Binv = np.linalg.inv(B)
    # ---- 主運算 ----
    out = {}
    for (c, t), G in pack["matrices"].items():
        # 全域索引 → 局部索引
        c_loc, t_loc = _to_local_indices(active, c, t)
        CXM_active, _ = cnot_pauli_twirling_matrix(m, c_loc, t_loc)
        # M' = CXM^T * B^-1 * G * A^-1
        Mavg = CXM_active.T @ Binv @ G @ Ainv
        out[(c, t)] = np.asarray(Mavg, dtype=float)
    return {
        "num_qubits": m,
        "active_qubits": active,
        "row_labels": pack["row_labels"],
        "col_labels": pack["col_labels"],
        "matrices": out,
    }

def _labels_to_xz(labels):
    X=np.zeros((len(labels),len(labels[0])),np.uint8)
    Z=np.zeros_like(X)
    for i,lab in enumerate(labels):
        x,z=_label_to_xz(lab); X[i,:]=x; Z[i,:]=z
    return X,Z

def build_commutation_transform(labels: List[str]):
    X,Z=_labels_to_xz(labels)
    parity=(X@Z.T - Z@X.T) & 1
    return (1-2*parity).astype(np.int8)

def compute_weights_for_package(avg_pack: Dict):
    T=build_commutation_transform(avg_pack["row_labels"])
    out={}
    for key,M in avg_pack["matrices"].items():
        d=np.diag(M)
        out[key]=T@d
    return out

def compute_inv_weights_for_package(avg_pack: Dict, eps:float=0.0):
    n=avg_pack["num_qubits"]
    T=build_commutation_transform(avg_pack["row_labels"])
    Tinv=T.astype(float)/(4.0**n)
    out={}
    for key,M in avg_pack["matrices"].items():
        c=np.diag(M)
        if eps>0:
            s=np.sign(c); s[s==0]=1; c=s*np.maximum(np.abs(c),eps)
        out[key]=Tinv@(1.0/c)
    return out