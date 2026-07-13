# ==============================================================
#  chsh_tools_sampler.py
#  - 把 main 的 CHSH/權重/比較工具抽出來（Sampler 版）
#  - 正確處理 II/IX/IY/.../ZZ 全 16 個項目（含 I 的不會被量錯）
# ==============================================================
from __future__ import annotations
import os, json, time
import numpy as np
from itertools import product
from typing import Dict, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

import sys
from pathlib import Path
ROOT = Path().resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from PEC_sampler_package_teleportation import tele_PEC_sampler as pec

# 你原本的固定順序（請勿改）
Tele_ORDER = ["XII", "YII","ZII"]

# ---------------- 紀錄工具 ----------------
def _json_safe(x):
    """Convert numpy / complex / arrays / dicts into JSON-serializable objects."""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, complex):
        # 你 state 是 complex tuple，必要時會用到
        return {"re": float(np.real(x)), "im": float(np.imag(x))}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    # fallback：至少不要炸
    return str(x)

def _append_jsonl(path: str, record: dict):
    """Append one JSON record to JSONL file, flush+fsync for immediate persistence."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(_json_safe(record), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())

# ---------------- 基本工具 ----------------
def pauli_labels_k(k: int):
    alpha = ['I','X','Y','Z']
    return [''.join(t) for t in product(alpha, repeat=k)]

def obs(label2):
    # label2 like 'XX'; SparsePauliOp 需要右到左
    return SparsePauliOp.from_list([(label2[::-1], 1.0)])

def teleportation_circuit(state: Tuple[complex, complex]) -> QuantumCircuit:
    """
    建立 teleportation 電路，輸入態為 state = (a, b) 尙未正規化。
    回傳 QuantumCircuit 物件。
    """
    a, b = state
    norm = np.sqrt(abs(a)**2 + abs(b)**2)
    a /= norm; b /= norm

    qc = QuantumCircuit(3, 1)
    qc.initialize([a, b], 0)
    # construct Bell pair at q1, q2
    qc.h(1); qc.cx(1, 2); qc.barrier()
    # Teleportation circuit
    qc.cx(0, 1); qc.h(0); qc.barrier()
    # cx identify
    qc.cx(1, 2); 
    # cz identify
    qc.h(2); qc.cx(0, 2); qc.h(2)
    return qc

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

def meas_dict_to_vec(meas_dict, order=Tele_ORDER, *, prefix="meas"):
    return np.array([float(meas_dict[f"{prefix}{lbl}"]["value"]) for lbl in order], dtype=float)

def expect_from_weights(weights_dict, label, m):
    w = np.asarray(weights_dict[label], dtype=float).reshape(-1)
    return float(w @ m)

def expct_from_pec_results(meas_dict, weights_dict):
    m = meas_dict_to_vec(meas_dict, pauli_labels_k(3))
    XII = expect_from_weights(weights_dict, "XII", m)
    YII = expect_from_weights(weights_dict, "YII", m)
    ZII = expect_from_weights(weights_dict, "ZII", m)
    expectation = {"XII": XII, "YII": YII, "ZII": ZII}
    return expectation

def state_fidelity(init_state: Tuple[complex, complex], final_state: Tuple[float, float, float]) -> float:
    norm = np.sqrt(abs(init_state[0])**2 + abs(init_state[1])**2)
    na, nb = init_state[0]/norm, init_state[1]/norm
    # Bloch vector of |psi>
    sx = 2*np.real(np.conj(na)*nb)
    sy = 2*np.imag(np.conj(na)*nb)
    sz = abs(na)**2 - abs(nb)**2
    F = (1 + sx*final_state["XII"] + sy*final_state["YII"] + sz*final_state["ZII"])/2
    return F


# ---------------- Sampler：建立 量測程式（支援 I） ----------------

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

    for idx, lab in enumerate(Tele_ORDER):
        a, b, c =lab[2], lab[1], lab[0]
        need = (a != 'I') + (b != 'I') + (c != 'I')
        if need == 0:
            # III：不跑 Sampler，值恆為 +1
            plan.append(('const', idx, 1.0))
            continue
        qc = base.copy()
        ci = 0
        if a != 'I':
            _rot_one(qc, qA, a)
            qc.measure(qA, ci); ci += 1
        if b != 'I':
            _rot_one(qc, qB, b)
            qc.measure(qB, ci); ci += 1
        if c != 'I':
            _rot_one(qc, 2, c)
            qc.measure(2, ci)
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
    回傳長度16的 m 向量（依 ORDER），包含 XII/YII/ZII。
    - III 直接填 1
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

    m = np.zeros(len(Tele_ORDER), dtype=float)
    for kind, idx, val in plan:
        if kind == 'const':
            m[idx] = float(val)  # III -> +1
        else:
            run_id = int(val)
            m[idx] = _exp_from_dist(dists[run_id])
    return m

# ---------------- select pec running set ----------------
def _analyze_from_wvecs(
    wvecs: list[np.ndarray],
    N: int,
    *,
    threshold: float = 1,
    verbose: bool = True,
):
    n_cnot = len(wvecs)
    idx_tuples = []
    w_prods = []
    for idx_tuple in product(range(N), repeat=n_cnot):
        w = 1.0
        for j, i_lab in enumerate(idx_tuple):
            w *= wvecs[j][i_lab]
        if w != 0.0:
            idx_tuples.append(idx_tuple)
            w_prods.append(w)

    w_prods = np.asarray(w_prods, dtype=float)
    abs_w = np.abs(w_prods)
    order = np.argsort(abs_w)[::-1]
    abs_w = abs_w[order]
    w_sorted = w_prods[order]
    idx_sorted = [idx_tuples[i] for i in order]

    gamma_total = float(abs_w.sum())
    cum_ratio = np.cumsum(abs_w) / gamma_total
    K = int(np.searchsorted(cum_ratio, threshold) + 1)
    k_idx = K - 1
    signed_sum_kept = float(w_sorted[:K].sum())

    diag = {
        "twirled_cnot_count": n_cnot,
        "nonzero_combos": len(idx_sorted),
        "gamma_total": gamma_total,
        "threshold": threshold,
        "keep_K": K,
        "kept_ratio": float(cum_ratio[K - 1]),
        "plot_abs_w_sorted": abs_w,
        "plot_signed_w_sorted": w_sorted,
        "plot_cum_ratio": cum_ratio,
        "kept_idx_tuples": idx_sorted[:K],
        "kept_w_prods": w_sorted[:K]/signed_sum_kept,
    }

    if verbose:
        print("\n[ Phase-0 PEC truncation analysis ]")
        print(f"  twirled CNOT count : {n_cnot}")
        print(f"  nonzero combos     : {len(idx_sorted)}")
        print(f"  gamma_total = Σ|w| : {gamma_total:.6e}")
        print(f"  cutoff |w|         : {abs_w[k_idx]}")
        print(f"  cutoff signed w    : {w_sorted[k_idx]}")
        print(f"  cumulative ratio at cutoff : {cum_ratio[k_idx]}")
        print(f"  keep K             : {K}")
        print(f"  kept ratio         : {cum_ratio[K-1]:.6f}")

    return diag


def prepare_tqg_pec_plan(
    qcircuit: QuantumCircuit,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    segment_id: int = 0,
    segment_count: int = 1,
    trunc_threshold: float | None = None,
    verbose: bool = True,
):
    # 只做一次 setup
    circ = pec._ensure_quantum_circuit(qcircuit)
    info = pec.circuit_information(circ)

    active_qubits = info["active_qubits"]
    n_active = len(active_qubits)
    pre_labels = pec.enum_pauli_labels(n_active)
    N = len(pre_labels)

    # teleportation: skip first CNOT
    full_cnot_list = info["cnot_list"]
    twirl_cnot_list = full_cnot_list[0:]
    n_cnot = len(twirl_cnot_list)

    # wvecs + dim check
    wvecs = []
    for rec in twirl_cnot_list:
        key = (rec["control"], rec["target"])
        w = tqg_weights.get(key)
        if w is None or len(w) != N:
            raise ValueError(f"weights_map[{key}] length error.")
        wvecs.append(np.asarray(w, dtype=float))

    diag = None

    if trunc_threshold is None:
        total_all = N ** n_cnot
        seg_size = total_all // segment_count
        start = segment_id * seg_size
        end = (segment_id + 1) * seg_size if segment_id < segment_count - 1 else total_all

        it_all = product(range(N), repeat=n_cnot)
        it = pec.islice(it_all, start, end)
        it_pairs = [(idx_tuple, None) for idx_tuple in it]

        if verbose:
            print(f"Segment {segment_id+1}/{segment_count}: full-set {start}..{end-1} (total {len(it_pairs)})")

    else:
        diag = _analyze_from_wvecs(wvecs, N, threshold=trunc_threshold, verbose=verbose)

        kept_idx_tuples = diag["kept_idx_tuples"]
        kept_w_prods = diag["kept_w_prods"]

        total_kept = len(kept_idx_tuples)
        seg_size = total_kept // segment_count
        start = segment_id * seg_size
        end = (segment_id + 1) * seg_size if segment_id < segment_count - 1 else total_kept

        it_pairs = list(zip(kept_idx_tuples[start:end], kept_w_prods[start:end]))

        if verbose:
            print(f"Segment {segment_id+1}/{segment_count}: kept-set {start}..{end-1} (total {len(it_pairs)})")

    return {
        "circ": circ,
        "info": info,
        "active_qubits": active_qubits,
        "pre_labels": pre_labels,
        "twirl_cnot_list": twirl_cnot_list,
        "N": N,
        "n_cnot": n_cnot,
        "trunc_threshold": trunc_threshold,
        "it_pairs": it_pairs,
        "diag": diag,
    }

# ---------------- 無 PEC / PEC 單次試驗 ----------------
def one_trial_nopec_S_sampler(
    sampler, state,
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
    expct = {"XII": m[0], "YII": m[1], "ZII": m[2]}
    fid = state_fidelity(state, expct)
    return fid, expct

def one_trial_pec_S_sampler(
    sampler, 
    state, 
    base_circuit,
    observables: Dict[str, SparsePauliOp],
    *,
    backend=None,
    tqg_weights: Dict[Tuple[int,int], np.ndarray],
    readout_weights: Dict[str, np.ndarray],
    opt_level: int = 0,
    shots: int = 4096,
    combo_batch_size: int = 256,
    max_batch: int = 1024,
    plan: dict | None = None, 
):
    """
    Sampler：用 TQG-PEC 重建 {'measXX':{'value':...}}，再用 readout weights 算 CHSH。
    """
    circ = base_circuit
    results = pec.run_tqg_pec_package_sampler(
        sampler=sampler,
        qcircuit=circ,
        observables=observables,            # {'measXX': SparsePauliOp(...), ...}
        tqg_weights=tqg_weights,
        backend=backend,
        opt_level=opt_level,
        shots=shots,
        combo_batch_size=combo_batch_size,
        max_batch=max_batch,
        plan=plan,  
        verbose=True,
    )
    print(results)
    diag = results.get("_trunc_diag", None)
    expct = expct_from_pec_results(results, readout_weights)
    fid = state_fidelity(state, expct)
    return fid, expct, diag

# ---------------- 多次 trial 對照 ----------------
def compare_pec_vs_nopec_sampler(
    sampler,
    state: Tuple[complex, complex],
    tqg_weights: Dict[Tuple[int,int], np.ndarray],
    readout_weights: Dict[str, np.ndarray],
    *,
    n_trials: int = 5,
    backend=None,
    opt_level: int = 0,
    shots: int = 4096,
    combo_batch_size: int = 256,
    max_batch: int = 1024,
    segment_id: int = 0,   # 第幾段：0 或 1
    segment_count: int = 1,
    trunc_threshold: float = 1,
    log_path: str = "./logs/tele_fidelity_pec_trials.jsonl"
):
    qcircuit = teleportation_circuit(state) 
    meas_qubits=infer_meas_qubits(qcircuit)
    observables = {f"meas{lbl}": obs(lbl) for lbl in pauli_labels_k(3)}

    fidelity_pec, fidelity_nopec = [], []
    meas_pec_records, meas_nopec_records = [], []
    diag_records = []
    plan = prepare_tqg_pec_plan(qcircuit=qcircuit, tqg_weights=tqg_weights, 
                                segment_id=segment_id, segment_count=segment_count, 
                                trunc_threshold=trunc_threshold, verbose=True)
    
    _append_jsonl(log_path, {
        "type": "run_header",
        "ts": time.time(),
        "n_trials": n_trials,
        "shots": shots,
        "combo_batch_size": combo_batch_size,
        "max_batch": max_batch,
        "segment_id": segment_id,
        "segment_count": segment_count,
        "trunc_threshold": trunc_threshold,
        "meas_qubits": meas_qubits,
        "plan_diag": plan.get("diag", None),
    })

    for i in range(1, n_trials+1):
        fid_nopec, meas_nopec = one_trial_nopec_S_sampler(
            sampler, state, qcircuit, backend=backend, 
            opt_level=opt_level, shots=shots,
            meas_qubits=meas_qubits
        )
        fidelity_nopec.append(fid_nopec); meas_nopec_records.append(meas_nopec)
        print(f"[{i}/{n_trials}]  S(noPEC)={fid_nopec:.8f}")
    for i in range(1, n_trials+1):      
        t0 = time.perf_counter()
        fid_pec, meas_pec, diag = one_trial_pec_S_sampler(
            sampler, state, qcircuit, observables, backend=backend,
            tqg_weights=tqg_weights, readout_weights=readout_weights,
            opt_level=opt_level, shots=shots, combo_batch_size=combo_batch_size, 
            max_batch=max_batch, plan=plan
        )
        dt = time.perf_counter() - t0
        _append_jsonl(log_path, {
            "type": "trial",
            "ts": time.time(),
            "trial": i,
            "fid_pec": float(fid_pec),
            "meas_pec": meas_pec,   # dict 也會被 _json_safe 處理
            "diag": diag,           # dict/ndarray 都 ok
            "elapsed_s": dt,
        })
        fidelity_pec.append(fid_pec); meas_pec_records.append(meas_pec); diag_records.append(diag)
        print(f"[{i}/{n_trials}]  S(PEC)={fid_pec:.8f}")

    fidelity_pec = np.asarray(fidelity_pec, dtype=float)
    fidelity_nopec = np.asarray(fidelity_nopec, dtype=float)
    summary = {
        "PEC_mean": float(fidelity_pec.mean()),
        "PEC_std": float(fidelity_pec.std(ddof=1)) if len(fidelity_pec)>1 else 0.0,
        "noPEC_mean": float(fidelity_nopec.mean()),
        "noPEC_std": float(fidelity_nopec.std(ddof=1)) if len(fidelity_nopec)>1 else 0.0,
    }
    return fidelity_pec, fidelity_nopec, summary, meas_pec_records, meas_nopec_records, diag_records

def analyze_tqg_weights_cumulative(
    qcircuit: QuantumCircuit,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    threshold: float = 0.99,
    verbose: bool = True,
):
    """
    分析 TQG PEC 的 |w_prod| 累積分布（variance / overhead 導向）。
    不跑模擬，只算 weights。
    第一個 CNOT 不納入 product。
    """
    circ = pec._ensure_quantum_circuit(qcircuit)
    info = pec.circuit_information(circ)

    active_qubits = info["active_qubits"]
    n_active = len(active_qubits)
    pre_labels = pec.enum_pauli_labels(n_active)
    N = len(pre_labels)

    # ====== discard first CNOT ======
    full_cnot_list = info["cnot_list"]
    twirl_cnot_list = full_cnot_list[-3:]
    n_cnot = len(twirl_cnot_list)

    # 檢查權重維度
    for rec in info["cnot_list"]:
        key = (rec["control"], rec["target"])
        if key not in tqg_weights:
            raise ValueError(f"Missing weights for CNOT {key}.")
        if len(tqg_weights[key]) != N:
            raise ValueError(f"weights[{key}] length mismatch.")

    # ====== enumerate all w_prod ======
    abs_w_list = []

    for idx_tuple in product(range(N), repeat=n_cnot):
        w_prod = 1.0
        for j, i_lab in enumerate(idx_tuple):
            rec = twirl_cnot_list[j]
            key = (rec["control"], rec["target"])
            w_prod *= float(tqg_weights[key][i_lab])

        if w_prod != 0.0:
            abs_w_list.append(abs(w_prod))

    abs_w = np.asarray(abs_w_list)
    abs_w.sort()
    abs_w = abs_w[::-1]   # descending |w|

    # ====== cumulative |w| ======
    gamma_total = abs_w.sum()
    cum_abs = np.cumsum(abs_w)
    cum_ratio = cum_abs / gamma_total

    cutoff_idx = int(np.searchsorted(cum_ratio, threshold))

    # ====== print summary ======
    if verbose:
        print("\n[ PEC |w_prod| cumulative analysis ]")
        print(f"  twirled CNOT count : {n_cnot}")
        print(f"  nonzero combos     : {len(abs_w)}")
        print(f"  gamma_total = Σ|w| : {gamma_total:.6e}")
        print(f"  threshold          : {threshold:.3f}")
        print(f"  keep K             : {cutoff_idx + 1}")
        print(f"  kept ratio         : {cum_ratio[cutoff_idx]:.6f}")
        print(f"  eliminated ratio   : {1 - cum_ratio[cutoff_idx]:.6f}")

    return abs_w, cum_ratio, cutoff_idx
