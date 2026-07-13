from __future__ import annotations
import json
import time
from collections import deque
import numpy as np
from itertools import product, islice
from collections import Counter
from typing import Dict, List, Tuple, Optional, Any
from itertools import product, islice
from concurrent.futures import ProcessPoolExecutor
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
                      phys2active: Optional[Dict[int, int]]=None) -> Dict[int, Tuple[str, str]]:
    idx2pair = {}
    for lab, rec in zip(label_strs, cnot_list):
        c = rec["control"]; t = rec["target"]
        post_lab = pauli_conj_by_cnot(lab, c, t, active=phys2active)
        idx2pair[rec["data_idx"]] = (lab, post_lab)
    return idx2pair

# --------------- Weight Sorting + threshold set---------------
def build_threshold_combo_set_by_abs(
    qcircuit: QuantumCircuit,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    discard_first_cnot: bool = True,
    threshold: float = 0.99,
    verbose: bool = True,
):
    """
    先取 abs() 看 amplitude：
      - 計算所有 w_prod（signed）
      - 以 abs(w_prod) 排序
      - 取 cumulative abs 超過 threshold 的最小 set
    回傳：
      twirl_cnot_list, pre_labels,
      kept_idx_tuples(list[tuple[int]]), kept_w_prod(np.ndarray),
      diag(dict)
    """
    circ = _ensure_quantum_circuit(qcircuit)
    info = circuit_information(circ)

    active_qubits = info["active_qubits"]
    n_active = len(active_qubits)
    pre_labels = enum_pauli_labels(n_active)
    N = len(pre_labels)

    full_cnot_list = info["cnot_list"]
    if discard_first_cnot:
        twirl_cnot_list = full_cnot_list[1:]   # 只 twirl 後面三個
    else:
        twirl_cnot_list = full_cnot_list

    n_cnot = len(twirl_cnot_list)
    if n_cnot <= 0:
        raise ValueError("No CNOTs selected for twirling.")

    # 檢查 weights 維度（照你原本寫法）
    for rec in info["cnot_list"]:
        key = (rec["control"], rec["target"])
        if key not in tqg_weights:
            raise ValueError(f"Missing weights for CNOT {key}.")
        if len(tqg_weights[key]) != N:
            raise ValueError(f"weights[{key}] length mismatch: {len(tqg_weights[key])} != {N}")

    # ====== enumerate all combos & compute w_prod ======
    idx_tuples = []
    w_prods = []

    # 為了加速：先把每個 twirled CNOT 對應的 weight vector 抽出來
    wvecs = []
    for rec in twirl_cnot_list:
        key = (rec["control"], rec["target"])
        wvecs.append(np.asarray(tqg_weights[key], dtype=float))

    for idx_tuple in product(range(N), repeat=n_cnot):
        w_prod = 1.0
        for j, i_lab in enumerate(idx_tuple):
            w_prod *= float(wvecs[j][i_lab])
        if w_prod != 0.0:
            idx_tuples.append(idx_tuple)
            w_prods.append(w_prod)

    w_prods = np.asarray(w_prods, dtype=float)
    abs_w = np.abs(w_prods)

    # ====== sort by abs(w_prod) descending ======
    order = np.argsort(abs_w)[::-1]
    abs_w_sorted = abs_w[order]
    w_sorted = w_prods[order]              # 保留 signed（後面跑 PEC 就用它）
    idx_sorted = [idx_tuples[i] for i in order]

    # ====== find minimal set reaching threshold in cumulative abs ======
    gamma_total = float(abs_w_sorted.sum())
    if gamma_total <= 0:
        raise ValueError("gamma_total is 0. Check weights.")
    cum_ratio = np.cumsum(abs_w_sorted) / gamma_total
    K = int(np.searchsorted(cum_ratio, threshold, side="left") + 1)
    K = min(K, len(idx_sorted))

    kept_idx = idx_sorted[:K]
    kept_w = w_sorted[:K]
    kept_abs = abs_w_sorted[:K]
    kept_ratio = float(cum_ratio[K-1]) if K > 0 else 0.0

    diag = {
        "twirled_cnot_count": n_cnot,
        "nonzero_combos": int(len(idx_sorted)),
        "gamma_total": gamma_total,            # Σ|w|
        "threshold": float(threshold),
        "keep_K": int(K),
        "kept_ratio": kept_ratio,
        "eliminated_ratio": float(1.0 - kept_ratio),
        # 給你作圖用（signed 會震盪沒關係）
        "plot_signed_w_sorted": w_sorted,      # 全部排序後的 signed w
        "plot_abs_w_sorted": abs_w_sorted,     # 全部排序後的 |w|
        "plot_cum_ratio": cum_ratio,           # 全部累積比
    }

    if verbose:
        print("\n[ PEC |w_prod| cumulative analysis ]")
        print(f"  twirled CNOT count : {n_cnot}")
        print(f"  nonzero combos     : {len(idx_sorted)}")
        print(f"  gamma_total = Σ|w| : {gamma_total:.6e}")
        print(f"  threshold          : {threshold:.3f}")
        print(f"  keep K             : {K}")
        print(f"  kept ratio         : {kept_ratio:.6f}")
        print(f"  eliminated ratio   : {1.0 - kept_ratio:.6f}")

    return twirl_cnot_list, pre_labels, kept_idx, kept_w, diag


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
                                  idx2pair: Optional[Dict[int, Tuple[str, str]]] = None,
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

# -------------------------------------------------
#  Plan builder（讓 main code 決定要不要做）
# -------------------------------------------------
def _make_truncation_pairs(
    *,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    twirl_cnot_list: List[Dict[str, int]],
    N: int,
    trunc_threshold: float = 0.99,
    precompute_wprod: bool = True,
    verbose: bool = True,
):
    """
    Truncation rule:
      - enumerate all idx_tuple in product(range(N), repeat=n_cnot)
      - compute w_prod = Π_j w_j[idx_j]
      - keep only nonzero w_prod
      - sort by |w_prod| descending
      - choose minimal K such that (Σ_{k<=K} |w_k|) / (Σ_all |w|) >= trunc_threshold
      - return it_pairs of length K:
          * (idx_tuple, w_prod) if precompute_wprod=True
          * (idx_tuple, None)   if precompute_wprod=False  (runner will recompute)

    Returns:
      it_pairs, diag
    """
    n_cnot = len(twirl_cnot_list)

    # Build per-CNOT weight vectors (and dimension check)
    wvecs: List[np.ndarray] = []
    for rec in twirl_cnot_list:
        key = (rec["control"], rec["target"])
        w = tqg_weights.get(key)
        if w is None or len(w) != N:
            raise ValueError(f"weights_map[{key}] length error: expect {N}.")
        wvecs.append(np.asarray(w, dtype=float))

    idx_tuples: List[Tuple[int, ...]] = []
    w_prods: List[float] = []

    # Enumerate all combinations and compute w_prod
    for idx_tuple in product(range(N), repeat=n_cnot):
        w = 1.0
        for j, i_lab in enumerate(idx_tuple):
            w *= float(wvecs[j][i_lab])
        if w != 0.0:
            idx_tuples.append(idx_tuple)
            w_prods.append(w)

    w_prods = np.asarray(w_prods, dtype=float)

    # Edge case: all weights zero => nothing to run
    if w_prods.size == 0:
        diag = {
            "twirled_cnot_count": n_cnot,
            "nonzero_combos": 0,
            "gamma_total": 0.0,
            "threshold": trunc_threshold,
            "keep_K": 0,
            "kept_ratio": 0.0,
            "plot_abs_w_sorted": np.array([]),
            "plot_signed_w_sorted": np.array([]),
            "plot_cum_ratio": np.array([]),
            "kept_idx_tuples": [],
            "kept_w_prods": [],
        }
        if verbose:
            print("[trunc] all w_prod are zero -> keep K=0")
        return [], diag

    abs_w = np.abs(w_prods)
    order = np.argsort(abs_w)[::-1]

    abs_w_sorted = abs_w[order]
    w_sorted = w_prods[order]
    idx_sorted = [idx_tuples[i] for i in order]

    gamma_total = float(abs_w_sorted.sum())
    cum_ratio = np.cumsum(abs_w_sorted) / gamma_total

    K = int(np.searchsorted(cum_ratio, trunc_threshold) + 1)
    K = min(K, len(idx_sorted))

    kept_idx = idx_sorted[:K]
    kept_w = w_sorted[:K]

    if precompute_wprod:
        it_pairs = list(zip(kept_idx, map(float, kept_w)))
    else:
        it_pairs = [(t, None) for t in kept_idx]

    diag = {
        "twirled_cnot_count": n_cnot,
        "nonzero_combos": len(idx_sorted),
        "gamma_total": gamma_total,
        "threshold": trunc_threshold,
        "keep_K": K,
        "kept_ratio": float(cum_ratio[K - 1]) if K > 0 else 0.0,
        # These are useful for plotting / debugging:
        "plot_abs_w_sorted": abs_w_sorted,
        "plot_signed_w_sorted": w_sorted,
        "plot_cum_ratio": cum_ratio,
        "kept_idx_tuples": kept_idx,
        "kept_w_prods": kept_w,
    }

    if verbose:
        print("\n[ PEC truncation analysis ]")
        print(f"  twirled CNOT count : {n_cnot}")
        print(f"  nonzero combos     : {len(idx_sorted)}")
        print(f"  gamma_total = Σ|w| : {gamma_total:.6e}")
        print(f"  threshold          : {trunc_threshold}")
        print(f"  keep K             : {K}")
        print(f"  kept ratio         : {diag['kept_ratio']:.6f}")

    return it_pairs, diag

def build_tqg_pec_plan(
    qcircuit,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    trunc_threshold: float = None,
    precompute_wprod: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    建立可重用的 plan：
    - enable_trunc=False：仍然可以做「預先準備」(circ/info/active_qubits/pre_labels/...) 讓每次 trial 少做重複工作
    - enable_trunc=True ：額外產出 it_pairs 與 diag（truncation 用）

    回傳的 plan 可直接丟入 run_tqg_pec_package_sampler(plan=plan)
    """
    circ = _ensure_quantum_circuit(qcircuit)
    info = circuit_information(circ)

    active_qubits = info["active_qubits"]
    n_active = len(active_qubits)
    pre_labels = enum_pauli_labels(n_active)
    N = len(pre_labels)

    # cnot 清單：你原本叫 cnot_list（全 twirl）或 twirl_cnot_list（部分 twirl）
    # 這裡先用「預設：全 twirl」
    twirl_cnot_list = info["cnot_list"]
    n_cnot = len(twirl_cnot_list)

    # 檢查 weights 維度
    for rec in twirl_cnot_list:
        key = (rec["control"], rec["target"])
        w = tqg_weights.get(key)
        if w is None or len(w) != N:
            raise ValueError(f"weights_map[{key}] length error: expect {N}.")

    plan: Dict[str, Any] = {
        "circ": circ,
        "info": info,
        "active_qubits": active_qubits,
        "pre_labels": pre_labels,
        "twirl_cnot_list": twirl_cnot_list,
        "N": N,
        "n_cnot": n_cnot,
        "diag": None,
        "it_pairs": None,  # list[(idx_tuple, w_prod_or_None)]
        "precompute_wprod": precompute_wprod,
    }

    if trunc_threshold is not None:
        it_pairs, diag = _make_truncation_pairs(
            tqg_weights=tqg_weights,
            twirl_cnot_list=twirl_cnot_list,
            N=N,
            trunc_threshold=trunc_threshold,
            precompute_wprod=precompute_wprod,
            verbose=verbose,
        )
        plan["it_pairs"] = it_pairs
        plan["diag"] = diag

    return plan

# ----------------------------
#  Sampler 批次執行（保持不變）
# ----------------------------
def tqg_pec_batch_circuit_sampler(
    sampler,
    pending_circs: List,
    pending_meta: List[Tuple[str, float]],
    results: Dict[str, float],
    *,
    backend=None,
    opt_level: int = 0,
    shots: int = 1024,
):
    """
    一次把 pending_circs 丟進 sampler.run，根據 meta 把結果加回 results。
    pending_meta: 每個元素是 (obs_name, w_prod)
    """
    if not pending_circs:
        return

    if backend is not None:
        tr_circs = transpile(pending_circs, backend=backend, optimization_level=opt_level)
    else:
        tr_circs = pending_circs

    job = sampler.run(tr_circs, shots=shots)
    res = job.result()
    dists = _extract_quasi_or_counts_list(res)

    if len(dists) != len(pending_circs):
        # fallback：某些實作要 res[i] 才看得到 quasi_dists
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

def _pec_worker_task(idx_batch, plan, tqg_weights, observables):
    """
    此函式由 ProcessPoolExecutor 呼叫，在獨立 CPU 核心執行。
    """
    circ = plan["circ"]
    info = plan["info"]
    active_qubits = plan["active_qubits"]
    pre_labels = plan["pre_labels"]
    twirl_cnot_list = plan["twirl_cnot_list"]
    
    local_tasks = []
    local_direct_sum = {name: 0.0 for name in observables}

    for idx_tuple in idx_batch:
        # 計算權重
        w_prod = 1.0
        for j, i_lab in enumerate(idx_tuple):
            rec = twirl_cnot_list[j]
            key = (rec["control"], rec["target"])
            w_prod *= float(tqg_weights[key][i_lab])
        
        if w_prod == 0: continue

        # 生成電路（呼叫你原本檔案裡的工具）
        labs = [pre_labels[i] for i in idx_tuple]
        idx2pair = _prepare_idx2pair(circ, twirl_cnot_list, labs, phys2active=info["phys2active"])
        base_twirl = _twirled_variant_for_all_cnot(circ, idx2pair=idx2pair, active_qubits=active_qubits)

        for name, obs in observables.items():
            qc = _meas_circuit_for_observable(base_twirl, obs, active_qubits)
            if qc is None:
                local_direct_sum[name] += w_prod
            else:
                # 這裡就是關鍵標籤化：將 (電路, 觀測量名, 權重) 綁在一起
                local_tasks.append({
                    "qc": qc,
                    "obs_name": name,
                    "weight": w_prod
                })
    return local_tasks, local_direct_sum

# -------------------------------------------------
#  統一版 runner：plan 有/無、trunc 有/無，都走同一套
# -------------------------------------------------
def run_tqg_pec_package_sampler(
    sampler,
    qcircuit,
    observables: Dict[str, Any],
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    backend=None,
    opt_level: int = 0,
    shots: int = 1024,
    num_workers: int = 64,  # 新增：指定要用多少核
    max_batch: int = 2048,   # 批次提交的大小
    plan: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Dict]:
    """
    並行化加速版：取代原本緩慢的單執行緒 while 迴圈。
    """
    t_start = time.perf_counter()

    # 如果沒傳 plan，就現場建一個
    if plan is None:
        plan = build_tqg_pec_plan(qcircuit, tqg_weights, verbose=verbose)

    # 決定迭代索引
    if plan.get("it_pairs") is not None:
        all_idx = [p[0] for p in plan["it_pairs"]]
    else:
        # 若沒 truncation，則遍歷所有組合
        N, n_cnot = plan["N"], plan["n_cnot"]
        all_idx = list(product(range(N), repeat=n_cnot))

    results = {name: 0.0 for name in observables}
    master_task_list = []

    # --- Phase 1: 並行生成 (Parallel Build) ---
    if verbose:
        print(f"[Parallel] Using {num_workers} workers to build {len(all_idx)} combinations...")

    # 切分任務給各個核心
    chunks = np.array_split(all_idx, num_workers)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_pec_worker_task, chunk, plan, tqg_weights, observables) for chunk in chunks]
        for f in futures:
            tasks, direct_sum = f.result()
            master_task_list.extend(tasks)
            for name, val in direct_sum.items():
                results[name] += val

    # --- Phase 2: 批次執行 (Batch Run) ---
    if verbose:
        print(f"[Sampler] Submitting {len(master_task_list)} circuits in batches of {max_batch}...")

    for i in range(0, len(master_task_list), max_batch):
        batch = master_task_list[i : i + max_batch]
        batch_qcs = [t["qc"] for t in batch]
        
        # 轉譯與執行
        if backend is not None:
            batch_qcs = transpile(batch_qcs, backend=backend, optimization_level=opt_level)
        
        job = sampler.run(batch_qcs, shots=shots)
        dists = _extract_quasi_or_counts_list(job.result())

        # 精確累加
        for task, dist in zip(batch, dists):
            exp_val = _exp_from_distribution(dist)
            results[task["obs_name"]] += task["weight"] * exp_val
        
        if verbose and (i // max_batch) % 5 == 0:
            print(f"  Progress: {min(i+max_batch, len(master_task_list))}/{len(master_task_list)}")

    t_end = time.perf_counter()
    if verbose: print(f"PEC completed in {t_end - t_start:.2f} seconds.")

    # 格式化輸出以兼容你原本的 main code
    out = {name: {"obs": name, "value": float(val)} for name, val in results.items()}
    if plan.get("diag") is not None:
        out["_trunc_diag"] = plan["diag"]
    return out
