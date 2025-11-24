import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import time
import numpy as np
import os, json, time
import concurrent.futures
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from functools import lru_cache  
from itertools import product, islice
from typing import Dict, List, Tuple, Optional
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2

# --------- laod circuit and get informations ----------
def _ensure_quantum_circuit(obj):
    """確保拿到的是 QuantumCircuit；否則拋出可讀錯誤。
       也容忍像 (circ, something) 這種 tuple，會自動取第 1 個。
    """
    if isinstance(obj, QuantumCircuit):
        return obj
    # 容忍包成 (circ, ...) 的情況
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], QuantumCircuit):
        return obj[0]
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
        new_c = [new.clbits[circ.find_bit(c).index] for c in cargs] if cargs else []
        new.append(op, new_q, new_c)
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

# ---- Construct observable labels：{I,X,Y,Z} ⊗ 4^n, from left----
def enum_pauli_labels(n: int) -> List[str]:
    return [''.join(p) for p in product(('I','X','Y','Z'), repeat=n)]


# --------- Construct pauli layer set ----------
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

# --------- Construct Cnot circuit ----------
# ---- Construct init -> pre-Pauli -> CNOT(c,t) -> post-Pauli ----
def build_twirled_circuit(n: int, init_circ: QuantumCircuit, c: int, t: int, pre_label: str) -> QuantumCircuit:
    post_label = pauli_conj_by_cnot(pre_label, c, t)
    qc = QuantumCircuit(n, name=f"twirl(c{c},t{t})_init={init_circ.name[5:]}_pre={pre_label}")
    qc.compose(init_circ, inplace=True)
    apply_pauli_layer(qc, pre_label)
    qc.cx(c, t)
    apply_pauli_layer(qc, post_label)
    return qc

def _extract_evs_from_result(est_res):
    # This function is compatible with EstimatorV2 results
    if hasattr(est_res, "__iter__"):
        vals = []
        for pub_res in est_res:
            # For EstimatorV2, each pub_res has a data field with evs
            if hasattr(pub_res.data, "evs"):
                vals.extend(np.asarray(pub_res.data.evs).ravel())
        if vals:
            return np.asarray(vals, dtype=float)
    raise TypeError(f"Unsupported result format: {type(est_res)}")

def _estimator_expect_many(estimator, circuit, observable):
    # This now assumes V2-style run
    pubs = [(circuit, observable)]
    if not pubs: return np.array([])
    job = estimator.run(pubs)
    result = job.result()
    return _extract_evs_from_result(result)

# --------- Calculate average expectation matrix of each cnot ----------
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
        # Find the corresponding qubits in the NEW circuit `qc` by their index
        new_qargs = [qc.qubits[orig.find_bit(q).index] for q in qargs]
        new_cargs = [qc.clbits[orig.find_bit(c).index] for c in cargs] if cargs else []

        pair = idx2pair.get(k)
        if pair is None:
            qc.append(op, new_qargs, new_cargs)
        else:
            pre_lab, post_lab = pair
            apply_pauli_layer(qc, pre_lab)
            qc.append(op, new_qargs, new_cargs)
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
    Correct PEC estimator for a single circuit and observable by JOINT enumeration...
    (docstring remains the same)
    """
    info = cnot_pec(circuit)
    n = info["num_qubits"]
    cnot_list = info["cnot_list"]
    m = len(cnot_list)

    if m == 0:
        # This part runs only once, so we won't add detailed timers here.
        cir = circuit
        if t_backend is not None:
            layout = _used_qubit_indices(cir) or list(range(cir.num_qubits))
            cir = transpile(
                [cir], backend=t_backend, initial_layout=layout,
                layout_method="trivial", routing_method="none",
                optimization_level=t_opt_level
            )[0]
            cir = _compact_qubits(cir)
        ev = estimator.run([(cir, observable)]).result()[0].data.evs
        return float(ev),  {"generation": 0, "transpilation": 0, "execution": 0, "processing": 0}

    pre_labels = enum_pauli_labels(n)
    N = len(pre_labels)
    it = product(range(N), repeat=m)
    total_combos = N**m
    acc = 0.0
    processed = 0
    timings = {"generation": 0.0, "transpilation": 0.0, "execution": 0.0, "processing": 0.0}
    
    # --- Start of main loop ---
    batch_num = 0 # Add a counter for batches
    while True:
        # === TIMER START: Overall loop timer ===
        t_loop_start = time.time()
        batch_num += 1

        batch_idx_tuples = list(islice(it, combo_batch_size))
        if not batch_idx_tuples:
            break

        # === TIMER START: Circuit Generation ===
        t1 = time.time()       
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

        t2 = time.time()
        timings["generation"] += (t2 - t1)
        # === TIMER END: Circuit Generation ===

        
        # === TIMER START: Transpilation ===
        t3 = time.time()     
        if t_backend is not None:
            layout = _used_qubit_indices(circuit) or list(range(circuit.num_qubits))
            circs = transpile(
                circs,
                backend=t_backend,
                initial_layout=layout,
                layout_method="trivial",
                routing_method="none",
                optimization_level=t_opt_level,
            )
            circs = [_compact_qubits(c) for c in circs]
        t4 = time.time()
        timings["transpilation"] += (t4 - t3)
        # === TIMER END: Transpilation ===

        # === TIMER START: Estimator Execution ===
        t5 = time.time()
        pubs = [(cir, [observable]) for cir in circs]
        res = estimator.run(pubs).result()
        t6 = time.time()
        timings["execution"] += (t6 - t5)
        # === TIMER END: Estimator Execution ===
        
        # === TIMER START: Result Processing ===
        t7 = time.time()    
        evs = _extract_evs_from_result(res)
        if evs.shape[0] != len(circs):
            evs = []
            for cir in circs:
                evs.append(_estimator_expect_many(estimator, cir, [observable])[0])
            evs = np.asarray(evs, dtype=float)      
        acc += float(np.dot(weight_batch, evs))
        t8 = time.time()
        timings["processing"] += (t8 - t7)
        # === TIMER END: Result Processing ===

        processed += len(circs)
        if progress and processed % (10 * combo_batch_size) == 0:
            pct = 100.0 * processed / total_combos
            print(f"[JOINT] {processed}/{total_combos} combos done ({pct:.2f}%)")       
        del circs, pubs, res, weight_batch

        # === TIMER END: Overall loop timer ===
        t_loop_end = time.time()
        print(f"[TIMER] Total Batch Time:     {t_loop_end - t_loop_start:.4f}s")
        print("---------------------------------")

    return float(acc), timings
                
def _ckpt_key(obs_name: str) -> str: 
    return f"{obs_name}" 

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

def run_single_observable_task(task_args: Tuple) -> Tuple[str, float]:
    (obs_name, observable, qcircuit, weights_map, backend_options_dict,
     combo_batch_size, t_opt_level) = task_args

    # CRITICAL: Create a new Backend and Estimator inside the worker process.
    # Force the backend to use only ONE CPU thread.
    worker_options = backend_options_dict.copy()
    worker_backend_options = worker_options.get("backend_options", {}).copy()
    worker_backend_options["max_parallel_threads"] = 1
    # worker_options["backend_options"] = worker_backend_options
    
    # # Recreate the EstimatorV2
    # local_estimator = AerEstimatorV2(options=worker_options)
    # est0 = AerEstimatorV2(options={"backend_options": worker_options, "run_options": {"shots": 0}})
    est1 = AerEstimatorV2(options={"backend_options": worker_backend_options, "run_options": {"shots": 1024}})


    # We need a backend object for transpilation
    t_backend = AerSimulator()
    t_backend.set_options(**worker_backend_options)

    # Call the calculation function
    value, timings = tqg_pec_single_circuit_joint(
        estimator=est1,
        circuit=qcircuit,
        observable=observable,
        weights_map=weights_map,
        combo_batch_size=combo_batch_size,
        t_backend=t_backend, # Use the local backend for transpilation
        t_opt_level=t_opt_level,
        progress=True
    )
    return obs_name, value, timings

def run_tqg_pec_package(estimator: AerEstimatorV2,
                        qcircuit: QuantumCircuit,
                        observables: Dict[str, SparsePauliOp],
                        weights_map: Dict[Tuple[int, int], np.ndarray],
                        *,
                        max_workers: Optional[int] = None,
                        combo_batch_size: int = 64,
                        ckpt_path: Optional[str] = None,
                        resume: bool = True,
                        t_backend=None, # This is now mainly for reference
                        t_opt_level: int = 0,
                        verbose: bool = True) -> Dict[str, Dict]:
    """
    Run JOINT Pauli enumeration TQG-PEC over the entire dataset.
    - observables: e.g., {'X': measurements['measX'], 'Y': measurements['measY']}
    - weights_map: {(c,t): weights} with length 4^n, order matching enum_pauli_labels(n)
    - ckpt_path: JSONL; after finishing each (circuit, observable), append one line (resumable).
    """
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if ckpt_path is None:
        ckpt_path = f"tqg_pec_ckpt_{int(time.time())}.jsonl"

    done_map = _load_done_set(ckpt_path) if resume else {}
    results = dict(done_map)
    tasks_to_run = {k: v for k, v in observables.items() if _ckpt_key(k) not in done_map}

    # Extract the backend options from the provided EstimatorV2 instance
    # This is the key change for V2 compatibility
    if t_backend is None:
        raise ValueError("t_backend (the backend instance) must be provided for parallel execution.")
    backend_options_dict = estimator.options.__dict__

    task_packages = [
        (obs_name, observable, qcircuit, weights_map, backend_options_dict,
         combo_batch_size, t_opt_level)
        for obs_name, observable in tasks_to_run.items()
    ]
    if not task_packages:
        if verbose: print("[RUN] All tasks are already completed.")
        return results

    if verbose:
        print(f"[RUN] Total tasks = {len(observables)}, To Do = {len(task_packages)}, Parallel Workers = {max_workers}")

    mode = "a" if resume else "w"
    with open(ckpt_path, mode) as fout, concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_observable_task, pkg): pkg[0] for pkg in task_packages}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            obs_name_done = futures[future]
            try:
                _, val, timings = future.result()
                key = _ckpt_key(obs_name_done)
                rec = {"obs": obs_name_done, "value": val, "key": key}
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                results[key] = rec
                if verbose:
                    total_time = sum(timings.values())
                    pct = 100.0 * (len(done_map) + i + 1) / len(observables)

                    print(f"[PROGRESS] {len(done_map) + i + 1}/{len(observables)} ({pct:5.1f}%) '{obs_name_done}' -> value={val:.6f}")
                    print(f"\n--- Task {len(done_map) + i + 1}/{len(observables)} Completed ({pct:.1f}%) ---")
                    print(f"  Observable: '{obs_name_done}' -> value={val:.6f}")
                    print(f"  Total Time: {total_time:.4f}s")
                    print("  Timings:")
                    print(f"    - Generation:    {timings['generation']:.4f}s")
                    print(f"    - Transpilation: {timings['transpilation']:.4f}s")
                    print(f"    - Execution:     {timings['execution']:.4f}s")
                    print(f"    - Processing:    {timings['processing']:.4f}s")
                    print("-----------------------------------------")
            except Exception as exc:
                print(f"\n[ERROR] Task '{obs_name_done}' failed with exception: {exc}\n")

    if verbose:
        print("[DONE] All parallel tasks completed. Results saved in", ckpt_path)
    return results

def linear_combo_targets(e_vec, readout_weight_dict, targets=("ZZ","IX")):
    out = {}
    for tgt in targets:
        w = np.asarray(readout_weight_dict[tgt]).ravel()  # shape (16,)
        assert w.shape[0] == len(BASE_LABELS_2Q)
        out[tgt] = float(w @ e_vec)
    return out