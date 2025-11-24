import time, os
import numpy as np
from numpy.linalg import inv
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as Sampler

# ============================================================
#  Part 1.  前處理：產生初態、Pauli 測量
# ============================================================
def find_active_qubits(circ):
    """回傳實際被非量測指令使用到的 qubit 索引（升冪）。相容新版 Qiskit。"""
    act = set()
    for inst, qargs, _ in circ.data:
        if inst.name == "measure":
            continue
        for q in qargs:
            # 新版：用 find_bit 取得全域 index；舊版退回 qubits.index()
            try:
                idx = circ.find_bit(q).index
            except Exception:
                idx = circ.qubits.index(q)
            act.add(idx)
    return sorted(act)

def local_index_map(active: list[int]) -> dict[int,int]:
    """global qubit idx -> local [0..k-1] 映射（依 active 排序）。"""
    return {q:i for i,q in enumerate(active)}

def pauli_labels_k(k: int):
    alpha = ['I','X','Y','Z']
    return [''.join(t) for t in product(alpha, repeat=k)]

def build_initial_states_on(active: list[int]):
    """
    只在 active qubits 上建立 4^k 種 product 初態（zero, one, plus, right）。
    回傳 dict[str -> CircuitBuilder]，它是「一個能在 base 電路前綴準備態」的 lambda。
    """
    labels = ['zero', 'one', 'plus', 'right']
    k = len(active)
    init_specs = {}

    for choices in product(labels, repeat=k):
        key = "_".join([f"q{active[i]}{choices[i]}" for i in range(k)])
        def builder_factory(choices_tuple):
            def add_prep(base: QuantumCircuit):
                for i, lab in enumerate(choices_tuple):
                    q = active[i]
                    if lab == 'one':
                        base.x(q)
                    elif lab == 'plus':
                        base.h(q)
                    elif lab == 'right':
                        base.h(q); base.s(q)
                return base
            return add_prep
        init_specs[key] = builder_factory(tuple(choices))
    return init_specs

def build_measurement_pauli_on(active: list[int]):
    """
    僅對 active qubits 產生所有 Pauli 測量（以 Pauli 字串表示，長度 = len(active)）。
    這裡回傳的是 Pauli 標籤字串；真正執行時會把它轉成旋到 Z 的基底操作。
    """
    k = len(active)
    meas = {}
    for p in pauli_labels_k(k):
        meas[f"meas{p}"] = p  # 先存字串；用時再做基底旋轉
    return meas

# ============================================================
#  Part 2. Sampler 計算：對所有 (state, measurement) 組合取樣
# ============================================================

def apply_measurement_for_pauli(circ: QuantumCircuit, active: list[int], pauli: str):
    """
    根據 pauli（長度 = len(active)）：
      - 若是 'X'：加 H 旋轉並量測
      - 若是 'Y'：加 Sdg+H 旋轉並量測
      - 若是 'Z'：直接量測
      - 若是 'I'：不旋轉、不量測
    classical bits 對應非 'I' qubit，由小到大排列。
    """
    # 篩選出要量測的 qubit
    measure_qubits = [active[i] for i, ch in enumerate(pauli) if ch != 'I']
    k = len(measure_qubits)
    if k == 0:
        return  # 全 I 無需量測

    creg = ClassicalRegister(k, 'c_meas')
    circ.add_register(creg)

    for i, ch in enumerate(pauli):
        q = active[i]
        if ch == 'I':
            continue
        elif ch == 'X':
            circ.h(q)
        elif ch == 'Y':
            circ.sdg(q)
            circ.h(q)
        # 'Z' 不旋轉

        circ.measure(q, i)

def expectation_from_counts_Z_only(counts: dict, pauli: str) -> float:
    """
    在 Z 基底下，針對長度=len(bitstring) 的 pauli（只含 I/Z）計算期望值。
    此函式用於已經做過基底旋轉的情況：此時只需用 Z/I 規則估算。
    """
    total = sum(counts.values())
    probs = {k: v/total for k,v in counts.items()}
    n = len(next(iter(probs)))  # bitstring 長度 = active qubits 個數
    # pauli 長度不足則左側補 I
    if len(pauli) < n: pauli = "I"*(n-len(pauli)) + pauli
    exp = 0.0
    for bs, p in probs.items():
        val = 1.0
        bits = bs[::-1]  # 右邊是 qubit0（對應 active[0]）
        for i in range(n):
            op = pauli[::-1][i]
            if op == 'Z':
                val *= 1.0 if bits[i]=='0' else -1.0
        exp += p*val
    return float(exp)

def run_sampler_on_subsystem(base_circuit: QuantumCircuit, 
                             sampler=None, backend=None, shots=1024,
                             batch_mode: bool = True, 
                             backend_options=None):
    """
    給一個「原始電路」（可能有很多 qubits 但只用其中幾個），
    1) 擷取 active qubits
    2) 只在 active 上建立 4^k 初態 × 4^k Pauli 測量
    3) Sampler 取樣，且只量 active
       - batch_mode=True: 每個 Pauli 一次丟一批所有 initial states 進 sampler
       - use_multiprocessing=True: 再把不同 Pauli 切給多個 process 跑
    4) 回傳 ev 向量、以及索引對照
    """
    active = find_active_qubits(base_circuit)
    if not active:
        raise ValueError("No active qubits detected.")
    qreg, creg = active[-1]+1, len(active)   
    k = len(active)

    inits = build_initial_states_on(active)
    meas  = build_measurement_pauli_on(active)
    init_keys = list(inits.keys())
    meas_keys  = list(meas.keys())

    # 如果 caller 有給 sampler/ backend，就只在「單 process 模式」使用
    # 多 process 模式裡每個 worker 自己 new，避免 pickling / thread 問題
    if not batch_mode:
        print("No acceleration: single process, single circuit per run.")
        if sampler is None: sampler = Sampler(options={"backend_options": backend_options,"run_options": {"shots": shots}})
        if backend is None: backend = AerSimulator()
        sampler.mode = backend
        
        evs = []
        for mk in meas_keys:
            pauli = meas[mk]  # 長度=k 的 Pauli 字串
            for ik in init_keys:           
                # 準備一份工作電路：先複製 base，再在 active 上加初態、旋基、量測
                qc = QuantumCircuit(qreg, creg)
                inits[ik](qc)                       # 在 active qubits 準備狀態
                apply_measurement_for_pauli(qc, active, pauli)
                qt = transpile(qc, backend=backend, optimization_level=0)
                res = sampler.run([qt]).result()[0]
                counts = res.data.c.get_counts()
                # 旋到 Z 後，期望值就是對 "Z...Z"(長度=k) 的 Z-only 規則
                ev = expectation_from_counts_Z_only(counts, "Z"*k)
                evs.append(ev)
            print(f"{pauli}:", evs[-16:-1])

        return np.asarray(evs, dtype=float), active, init_keys, meas_keys
    
    # =============== 單 process + batch 模式 ===============
    print("Batch mode: single process, batch circuits per measurement.")
    # 用同一個 backend/sampler，對每個 Pauli 一次丟一整批 init
    if sampler is None: sampler = Sampler(options={"backend_options": backend_options, "run_options": {"shots": shots}})
    if backend is None: 
        if backend_options is not None: backend = AerSimulator(noise_model=backend_options["noise_model"], basis_gates=backend_options["basis_gates"], coupling_map=["coupling_map"])
        else: backend = AerSimulator()
    sampler.mode = backend

    evs = []
    for mk in meas_keys:
        pauli = meas[mk]
        circuits_batch = []
        for ik in init_keys:
            qc = QuantumCircuit(qreg, creg)
            inits[ik](qc)
            apply_measurement_for_pauli(qc, active, pauli)
            qt = transpile(qc, backend=backend, optimization_level=0)
            circuits_batch.append(qt)

        job = sampler.run(circuits_batch)
        result = job.result()

        evs_for_this_meas = []
        for out in result:
            counts = out.data.c.get_counts()
            ev = expectation_from_counts_Z_only(counts, "Z" * k)
            evs_for_this_meas.append(ev)

        evs.extend(evs_for_this_meas)
        print(f"{pauli}:", evs_for_this_meas[-len(init_keys):-1])

    return np.asarray(evs, dtype=float), active, init_keys, meas_keys

# ============================================================
#  Part 3.  計算 G, A, B, qq (Readout Weights)
# ============================================================

def prep_matrix_kronA(k: int) -> np.ndarray:
    A1 = np.array([[1, 1, 1, 1],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1,-1, 0, 0]], dtype=float)
    out = np.array([1.0])
    for _ in range(k):
        out = np.kron(out, A1)
    return out

def build_corrected_observables_from_G(G: np.ndarray, k: int):
    """
    給子系統 k qubits 的 G，算 A、B、B^{-1}，並構建每個 Pauli 的 readout weight qq。
    回傳：
      A, B, qq（dict: pstr_k -> row vector shape (1, 4^k)）
    """
    A = prep_matrix_kronA(k)
    B = G @ inv(A)
    B_inv = inv(B)

    amap = {
        'I': np.array([[1,0,0,0]], dtype=float),
        'X': np.array([[0,1,0,0]], dtype=float),
        'Y': np.array([[0,0,1,0]], dtype=float),
        'Z': np.array([[0,0,0,1]], dtype=float),
    }
    qq = {}
    for p in pauli_labels_k(k):
        a = amap[p[0]]
        for ch in p[1:]:
            a = np.kron(a, amap[ch])
        qq[p] = a @ B_inv    # shape (1, 4^k)
    return A, B, qq

def readout_pec_sampler_subsystem(circuit: QuantumCircuit, 
                                  sampler=None, backend=None, shots=1024, 
                                  batch_mode: bool = True,
                                  backend_options=None):
    """
    ★ 主入口（子系統版）：
       自動偵測 active qubits（例如 {1,3}），
       只在這 k 個 qubit 上做 4^k×4^k 的 readout 校正，
       回傳 G/A/B/qq 與 active 索引。
    """
    evs, active, init_keys, meas_keys = run_sampler_on_subsystem(circuit, sampler=sampler, backend=backend, shots=shots, 
                                                                 batch_mode=batch_mode, backend_options=backend_options)
    k = len(active)
    dim = 4**k
    G = evs.reshape(dim, dim)
    A, B, qq = build_corrected_observables_from_G(G, k)
    return {
        "active_qubits": active,  # 例如 [1,3]
        "G": G,                   # (4^k, 4^k)
        "A": A,                   # (4^k, 4^k)
        "B": B,                   # (4^k, 4^k)
        "ReadoutWeight": qq,      # dict: p_k -> (1,4^k)
        "init_keys": init_keys,
        "meas_keys": meas_keys,
    }