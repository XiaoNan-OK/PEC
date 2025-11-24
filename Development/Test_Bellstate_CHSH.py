# ==============================================================
#  chsh_tools_sampler.py
#  - 把 main 的 CHSH/權重/比較工具抽出來（Sampler 版）
#  - 正確處理 II/IX/IY/.../ZZ 全 16 個項目（含 I 的不會被量錯）
# ==============================================================
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from Test_PEC_sampler import run_tqg_pec_package_sampler
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

# 你原本的固定順序（請勿改）
ORDER = ["II","XI","YI","ZI","IX","XX","YX","ZX","IY","XY","YY","ZY","IZ","XZ","YZ","ZZ"]
# ORDER = ["II","IX","IY","IZ","XI","XX","XY","XZ","YI","YX","YY","YZ","ZI","ZX","ZY","ZZ"]

# ---------------- 基本工具 ----------------
def obs(label2):
    # label2 like 'XX'; SparsePauliOp 需要右到左
    return SparsePauliOp.from_list([(label2[::-1], 1.0)])

def infer_meas_qubits(qc: QuantumCircuit) -> Tuple[int, int]:
    """
    自動推定 CHSH 測量的兩條 qubit：
    - 找出第一個兩體 CNOT 或 CZ 閘。
    - 取它的兩個 qubit index 為 Bell pair。
    - 若電路中有多個 entangler，就取最早出現的那對。
    """
    for inst in qc.data:
        op = getattr(inst, "operation", inst[0])
        qargs = getattr(inst, "qubits", inst[1])
        if op.name in ("cx", "cz"):
            q0 = qc.find_bit(qargs[0]).index
            q1 = qc.find_bit(qargs[1]).index
            return (q0, q1)
    # fallback：若沒 entangler，就取前兩條線
    return (0, 1)

def meas_dict_to_vec(meas_dict, order=ORDER, prefix="meas"):
    return np.array([float(meas_dict[f"{prefix}{lbl}"]["value"]) for lbl in order], dtype=float)

def expect_from_weights(weights_dict, label, m):
    w = np.asarray(weights_dict[label], dtype=float).reshape(-1)
    return float(w @ m)

# 這裡使用你最初的索引：XX=5, ZX=7, XZ=13, ZZ=15
def chsh_from_meas_vector(m):
    XX = float(m[5]);  ZX = float(m[7]);  XZ = float(m[13]);  ZZ = float(m[15])
    rt2 = np.sqrt(2.0)
    E00 = (ZZ + ZX)/rt2
    E01 = (ZZ - ZX)/rt2
    E10 = (XZ + XX)/rt2
    E11 = (XZ - XX)/rt2
    S = E00 + E01 + E10 - E11
    return dict(E00=float(E00), E01=float(E01), E10=float(E10), E11=float(E11), S=float(S))

def chsh_from_pec_results(meas_dict, weights_dict):
    m = meas_dict_to_vec(meas_dict, ORDER)
    XX = expect_from_weights(weights_dict, "XX", m)
    ZX = expect_from_weights(weights_dict, "ZX", m)
    XZ = expect_from_weights(weights_dict, "XZ", m)
    ZZ = expect_from_weights(weights_dict, "ZZ", m)
    rt2 = np.sqrt(2.0)
    E00 = (ZZ + ZX)/rt2
    E01 = (ZZ - ZX)/rt2
    E10 = (XZ + XX)/rt2
    E11 = (XZ - XX)/rt2
    S = E00 + E01 + E10 - E11
    return dict(E00=float(E00), E01=float(E01), E10=float(E10), E11=float(E11), S=float(S))

# ---------------- Sampler：建立 2Q 量測程式（支援 I） ----------------

def _rot_one(qc: QuantumCircuit, q: int, ch: str):
    if ch == 'X':
        qc.h(q)
    elif ch == 'Y':
        qc.sdg(q); qc.h(q)
    # 'I' 或 'Z' 不用轉

def build_meas_program_2q(base: QuantumCircuit, meas_qubits=(0,1)):
    """
    依 ORDER 產生：
      - circuits: 需要送 Sampler 的測量電路（只為非 'II' 的項目建立）
      - plan: 對應回長度16向量的位置填法：
          ('const', idx, +1.0)  -> 直接填 1（II）
          ('run',   idx, run_id) -> 用第 run_id 個 sampler 輸出計期望值後填入
    """
    qA, qB = meas_qubits
    circuits = []
    plan = []

    for idx, lab in enumerate(ORDER):
        a, b = lab[1], lab[0]
        need = (a != 'I') + (b != 'I')
        if need == 0:
            # II：不跑 Sampler，值恆為 +1
            plan.append(('const', idx, 1.0))
            continue
        qc = base.copy()
        ci = 0
        if a != 'I':
            _rot_one(qc, qA, a)
            qc.measure(qA, ci); ci += 1
        if b != 'I':
            _rot_one(qc, qB, b)
            qc.measure(qB, ci)
        circuits.append(qc)
        run_id = len(circuits) - 1
        plan.append(('run', idx, run_id))

    return circuits, plan

def _exp_from_dist(dist) -> float:
    # parity：奇數個 '1' → -1，偶數個 → +1
    try:
        items = dist.items()
    except Exception:
        dist = dict(dist); items = dist.items()
    tot = float(sum(float(v) for _, v in items))
    if tot <= 0: return 1.0
    s = 0.0
    for bitstr, v in items:
        p = float(v) / tot
        if isinstance(bitstr, str):
            ones = bitstr.replace(' ', '').count('1')
        else:
            ones = int(bitstr).bit_count()
        s += p * (-1.0 if (ones & 1) else 1.0)
    return float(s)

def _extract_quasi_or_counts_list(res):
    data = getattr(res, "data", None)
    if data is not None:
        cobj = getattr(data, "c", None)
        if cobj is not None and hasattr(cobj, "get_counts"):
            return [cobj.get_counts()]
        if hasattr(data, "quasi_dists"):
            return list(data.quasi_dists)
        if hasattr(data, "meas") and hasattr(data.meas, "quasi_dists"):
            return list(data.meas.quasi_dists)
    rs = getattr(res, "results", None)
    if rs and hasattr(rs[0], "data"):
        d = rs[0].data
        if hasattr(d, "quasi_dists"): return list(d.quasi_dists)
        if hasattr(d, "meas") and hasattr(d.meas, "quasi_dists"): return list(d.meas.quasi_dists)
    # 逐一索引回退
    out = []
    try:
        for i in range(len(res)):
            out.extend(_extract_quasi_or_counts_list(res[i]))
        if out:
            return out
    except Exception:
        pass
    raise TypeError("Unsupported SamplerResult format")

def build_meas_vector_nopec_sampler(sampler, base: QuantumCircuit, backend, *, opt_level=0, shots=4096, meas_qubits=(0,1)):
    """
    回傳長度16的 m 向量（依 ORDER），包含 II/IX/IY/.../ZZ。
    - II 直接填 1
    - 單 I：只量測另一條
    - 都非 I：兩條都量測
    """
    circs, plan = build_meas_program_2q(base, meas_qubits=meas_qubits)
    if backend is not None:
        circs = [transpile([circ], backend=backend, optimization_level=opt_level)[0] for circ in circs]
    # 跑需要的那些電路
    dists = []
    if len(circs) > 0:
        res = sampler.run(circs, shots=shots).result()
        dists = _extract_quasi_or_counts_list(res)

    m = np.zeros(len(ORDER), dtype=float)
    for kind, idx, val in plan:
        if kind == 'const':
            m[idx] = float(val)  # II -> +1
        else:
            run_id = int(val)
            m[idx] = _exp_from_dist(dists[run_id])
    print(m)
    return m

# ---------------- 無 PEC / PEC 單次試驗 ----------------

def one_trial_nopec_S_sampler(
    sampler,
    base_circuit: QuantumCircuit,
    *,
    backend=None,
    opt_level: int = 0,
    shots: int = 4096,
    meas_qubits=(0,1),
):
    """
    Sampler：為 2-qubit 的 16 個 Pauli⊗Pauli（含 I）建立測量程式，組成 m 向量，再算 CHSH。
    """
    circ = base_circuit
    m = build_meas_vector_nopec_sampler(sampler, circ, backend, opt_level=opt_level, shots=shots, meas_qubits=meas_qubits)
    cs = chsh_from_meas_vector(m)
    meas_dict = {f"meas{lbl}": {"value": float(m[i])} for i, lbl in enumerate(ORDER)}
    return float(cs["S"]), cs, meas_dict

def one_trial_pec_S_sampler(
    sampler,
    base_circuit: QuantumCircuit,
    observables: Dict[str, SparsePauliOp],
    *,
    tqg_weights: Dict[Tuple[int,int], np.ndarray],
    readout_weights: Dict[str, np.ndarray],
    backend=None,
    opt_level: int = 0,
    shots: int = 4096,
    combo_batch_size: int = 256,
    max_batch: int = 1024
):
    """
    Sampler：用 TQG-PEC 重建 {'measXX':{'value':...}}，再用 readout weights 算 CHSH。
    """
    circ = base_circuit
    results = run_tqg_pec_package_sampler(
        sampler=sampler,
        qcircuit=circ,
        observables=observables,            # {'measXX': SparsePauliOp(...), ...}
        tqg_weights=tqg_weights,
        backend=backend,
        opt_level=opt_level,
        shots=shots,
        combo_batch_size=combo_batch_size,
        max_batch=max_batch,
        verbose=True,
    )
    print(results)
    cs = chsh_from_pec_results(results, readout_weights)
    return float(cs["S"]), cs, results

# ---------------- 多次 trial 對照 ----------------

def compare_pec_vs_nopec_sampler(
    sampler,
    qcircuit: QuantumCircuit,
    tqg_weights: Dict[Tuple[int,int], np.ndarray],
    readout_weights: Dict[str, np.ndarray],
    *,
    n_trials: int = 5,
    backend=None,
    opt_level: int = 0,
    shots: int = 4096,
    combo_batch_size: int = 256,
    max_batch: int = 1024
):
    S_pec, S_nopec = [], []
    details_pec, details_nopec = [], []
    meas_pec_records, meas_nopec_records = [], []
    meas_qubits=infer_meas_qubits(qcircuit)
    observables = {f"meas{lbl}": obs(lbl) for lbl in ORDER}

    for i in range(1, n_trials+1):
        s_nopec, cs_nopec, meas_nopec = one_trial_nopec_S_sampler(
            sampler, qcircuit, backend=backend, opt_level=opt_level, shots=shots,
            meas_qubits=meas_qubits
        )
        s_pec, cs_pec, meas_pec = one_trial_pec_S_sampler(
            sampler, qcircuit, observables,
            tqg_weights=tqg_weights, readout_weights=readout_weights,
            backend=backend, opt_level=opt_level, shots=shots, 
            combo_batch_size=combo_batch_size, max_batch=max_batch
        )
        S_nopec.append(s_nopec); details_nopec.append(cs_nopec); meas_nopec_records.append(meas_nopec)
        S_pec.append(s_pec); details_pec.append(cs_pec); meas_pec_records.append(meas_pec)
        print(f"[{i}/{n_trials}]  S(PEC)={s_pec:.8f}   S(noPEC)={s_nopec:.8f}")

    S_pec = np.asarray(S_pec, dtype=float)
    S_nopec = np.asarray(S_nopec, dtype=float)
    summary = {
        "PEC_mean": float(S_pec.mean()),
        "PEC_std": float(S_pec.std(ddof=1)) if len(S_pec)>1 else 0.0,
        "noPEC_mean": float(S_nopec.mean()),
        "noPEC_std": float(S_nopec.std(ddof=1)) if len(S_nopec)>1 else 0.0,
    }
    return S_pec, S_nopec, summary, details_pec, details_nopec, meas_pec_records, meas_nopec_records
