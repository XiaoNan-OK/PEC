from __future__ import annotations

import json
import os
import sys
import time
from itertools import islice, product
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp

ROOT = Path().resolve().parents[1]
sys.path.append(str(ROOT / "PEC_Class" / "src"))

from mitigation.PEC_Readout import ReadoutQEM
from mitigation.PEC_Sampliing import PauliSamplingPEC
from mitigation.PEC_Twirling import PauliTwirlingPEC

TELE_ORDER = ["XII", "YII", "ZII"]


def _json_safe(x):
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, complex):
        return {"re": float(np.real(x)), "im": float(np.imag(x))}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    return str(x)


def _append_jsonl(path: str, record: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(_json_safe(record), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def pauli_labels_k(k: int):
    alpha = ["I", "X", "Y", "Z"]
    return ["".join(t) for t in product(alpha, repeat=k)]


def obs(label: str):
    return SparsePauliOp.from_list([(label[::-1], 1.0)])


def teleportation_circuit(state: Tuple[complex, complex]) -> QuantumCircuit:
    a, b = state
    norm = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
    a /= norm
    b /= norm

    qc = QuantumCircuit(3, 1)
    qc.initialize([a, b], 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()
    qc.cx(0, 1)
    qc.h(0)
    qc.barrier()
    qc.cx(1, 2)
    qc.h(2)
    qc.cx(0, 2)
    qc.h(2)
    return qc


def infer_meas_qubits(qc: QuantumCircuit) -> Tuple[int, int]:
    for inst in qc.data:
        op = getattr(inst, "operation", inst[0])
        qargs = getattr(inst, "qubits", inst[1])
        if op.name in ("cx", "cz"):
            q0 = qc.find_bit(qargs[0]).index
            q1 = qc.find_bit(qargs[1]).index
            return (q0, q1)
    return (0, 1)


def meas_dict_to_vec(meas_dict, order=TELE_ORDER, *, prefix="meas"):
    return np.array([float(meas_dict[f"{prefix}{label}"]["value"]) for label in order], dtype=float)


def expect_from_weights(weights_dict, label, meas_vec):
    weights = np.asarray(weights_dict[label], dtype=float).reshape(-1)
    return float(weights @ meas_vec)


def expct_from_pec_results(meas_dict, weights_dict):
    meas_vec = meas_dict_to_vec(meas_dict, pauli_labels_k(3))
    return {
        "XII": expect_from_weights(weights_dict, "XII", meas_vec),
        "YII": expect_from_weights(weights_dict, "YII", meas_vec),
        "ZII": expect_from_weights(weights_dict, "ZII", meas_vec),
    }


def state_fidelity(init_state: Tuple[complex, complex], final_state: Dict[str, float]) -> float:
    norm = np.sqrt(abs(init_state[0]) ** 2 + abs(init_state[1]) ** 2)
    na = init_state[0] / norm
    nb = init_state[1] / norm
    sx = 2 * np.real(np.conj(na) * nb)
    sy = 2 * np.imag(np.conj(na) * nb)
    sz = abs(na) ** 2 - abs(nb) ** 2
    return float((1 + sx * final_state["XII"] + sy * final_state["YII"] + sz * final_state["ZII"]) / 2)


def _rot_one(qc: QuantumCircuit, q: int, ch: str):
    if ch == "X":
        qc.h(q)
    elif ch == "Y":
        qc.sdg(q)
        qc.h(q)


def build_meas_program_tele(base: QuantumCircuit, meas_qubits=(0, 1)):
    q_a, q_b = meas_qubits
    circuits = []
    plan = []

    for idx, label in enumerate(TELE_ORDER):
        a, b, c = label[2], label[1], label[0]
        need = (a != "I") + (b != "I") + (c != "I")
        if need == 0:
            plan.append(("const", idx, 1.0))
            continue

        qc = base.copy()
        cidx = 0
        if a != "I":
            _rot_one(qc, q_a, a)
            qc.measure(q_a, cidx)
            cidx += 1
        if b != "I":
            _rot_one(qc, q_b, b)
            qc.measure(q_b, cidx)
            cidx += 1
        if c != "I":
            _rot_one(qc, 2, c)
            qc.measure(2, cidx)
        circuits.append(qc)
        plan.append(("run", idx, len(circuits) - 1))

    return circuits, plan


def _exp_from_dist(dist) -> float:
    try:
        items = dist.items()
    except Exception:
        dist = dict(dist)
        items = dist.items()
    total = float(sum(float(v) for _, v in items))
    if total <= 0:
        return 1.0
    out = 0.0
    for bitstr, value in items:
        prob = float(value) / total
        if isinstance(bitstr, str):
            ones = bitstr.replace(" ", "").count("1")
        else:
            ones = int(bitstr).bit_count()
        out += prob * (-1.0 if (ones & 1) else 1.0)
    return float(out)


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
        inner = rs[0].data
        if hasattr(inner, "quasi_dists"):
            return list(inner.quasi_dists)
        if hasattr(inner, "meas") and hasattr(inner.meas, "quasi_dists"):
            return list(inner.meas.quasi_dists)

    out = []
    try:
        for i in range(len(res)):
            out.extend(_extract_quasi_or_counts_list(res[i]))
        if out:
            return out
    except Exception:
        pass
    raise TypeError("Unsupported SamplerResult format")


def build_meas_vector_nopec_sampler(
    sampler,
    base: QuantumCircuit,
    backend,
    *,
    opt_level: int = 0,
    shots: int = 4096,
    meas_qubits=(0, 1),
):
    circuits, plan = build_meas_program_tele(base, meas_qubits=meas_qubits)
    if backend is not None:
        circuits = [transpile([circ], backend=backend, optimization_level=opt_level)[0] for circ in circuits]

    dists = []
    if circuits:
        dists = _extract_quasi_or_counts_list(sampler.run(circuits, shots=shots).result())

    meas_vec = np.zeros(len(TELE_ORDER), dtype=float)
    for kind, idx, value in plan:
        if kind == "const":
            meas_vec[idx] = float(value)
        else:
            meas_vec[idx] = _exp_from_dist(dists[int(value)])
    return meas_vec


def build_readout_weights_class(
    qcircuit: QuantumCircuit,
    *,
    sampler=None,
    backend=None,
    shots: int = 4096,
    batch_mode: bool = True,
    verbose: bool = False,
):
    readout = ReadoutQEM(backend=backend, shots=shots, verbose=verbose)
    return readout.calibrate(qcircuit, sampler=sampler, backend=backend, shots=shots, batch_mode=batch_mode)


def build_tqg_weights_class(
    qcircuit: QuantumCircuit,
    readout_result: dict,
    *,
    sampler,
    backend=None,
    shots: int = 4096,
    obs_batch: int = 256,
    parallel: bool = False,
    max_workers=None,
    verbose: bool = False,
    inverse: bool = True,
):
    twirling = PauliTwirlingPEC(backend=backend, shots=shots, verbose=verbose)
    twirl_pack = twirling.compute_tqg_matrices(
        sampler=sampler,
        input_circuit=qcircuit,
        shots=shots,
        obs_batch=obs_batch,
        parallel=parallel,
        max_workers=max_workers,
    )
    avg_pack = twirling.averaged_pauli_twirling_matrix(
        twirl_pack,
        readout_result["B"],
        active_qubits=readout_result["active_qubits"],
    )
    if inverse:
        return twirling.compute_inv_weights_for_package(avg_pack)
    return twirling.compute_weights_for_package(avg_pack)


def prepare_tqg_pec_plan_class(
    qcircuit: QuantumCircuit,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    segment_id: int = 0,
    segment_count: int = 1,
    trunc_threshold: Optional[float] = 0.99,
    verbose: bool = True,
    discard_first_cnot: bool = False,
    sampling_pec: Optional[PauliSamplingPEC] = None,
):
    sampling_pec = sampling_pec or PauliSamplingPEC(verbose=verbose)
    circ = sampling_pec._ensure_quantum_circuit(qcircuit)
    info = sampling_pec.circuit_information(circ)

    active_qubits = info["active_qubits"]
    pre_labels = sampling_pec.enum_pauli_labels(len(active_qubits))
    full_cnot_list = info["cnot_list"]
    twirl_cnot_list = full_cnot_list[1:] if discard_first_cnot else full_cnot_list
    n_labels = len(pre_labels)
    n_cnot = len(twirl_cnot_list)

    for rec in twirl_cnot_list:
        key = (rec["control"], rec["target"])
        weights = tqg_weights.get(key)
        if weights is None or len(weights) != n_labels:
            raise ValueError(f"weights_map[{key}] length error.")

    diag = None
    if trunc_threshold is None:
        total_all = n_labels ** n_cnot
        seg_size = total_all // segment_count
        start = segment_id * seg_size
        end = (segment_id + 1) * seg_size if segment_id < segment_count - 1 else total_all
        it_pairs = [(idx_tuple, None) for idx_tuple in islice(product(range(n_labels), repeat=n_cnot), start, end)]
        if verbose:
            print(f"Segment {segment_id + 1}/{segment_count}: full-set {start}..{end - 1} (total {len(it_pairs)})")
    else:
        it_pairs_all, diag = sampling_pec._make_truncation_pairs(
            tqg_weights=tqg_weights,
            twirl_cnot_list=twirl_cnot_list,
            n_labels=n_labels,
            trunc_threshold=trunc_threshold,
            precompute_wprod=True,
        )
        total_kept = len(it_pairs_all)
        seg_size = total_kept // segment_count
        start = segment_id * seg_size
        end = (segment_id + 1) * seg_size if segment_id < segment_count - 1 else total_kept
        it_pairs = it_pairs_all[start:end]
        if verbose:
            print(f"Segment {segment_id + 1}/{segment_count}: kept-set {start}..{end - 1} (total {len(it_pairs)})")

    return {
        "circ": circ,
        "info": info,
        "active_qubits": active_qubits,
        "pre_labels": pre_labels,
        "twirl_cnot_list": twirl_cnot_list,
        "N": n_labels,
        "n_cnot": n_cnot,
        "trunc_threshold": trunc_threshold,
        "it_pairs": it_pairs,
        "diag": diag,
    }


def one_trial_nopec_sampler(
    sampler,
    state,
    base_circuit: QuantumCircuit,
    *,
    backend=None,
    opt_level: int = 0,
    shots: int = 4096,
    meas_qubits=(0, 1),
):
    meas_vec = build_meas_vector_nopec_sampler(
        sampler,
        base_circuit,
        backend,
        opt_level=opt_level,
        shots=shots,
        meas_qubits=meas_qubits,
    )
    expectation = {"XII": meas_vec[0], "YII": meas_vec[1], "ZII": meas_vec[2]}
    fidelity = state_fidelity(state, expectation)
    return fidelity, expectation


def one_trial_pec_sampler(
    sampler,
    state,
    base_circuit: QuantumCircuit,
    observables: Dict[str, SparsePauliOp],
    *,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    readout_weights: Dict[str, np.ndarray],
    backend=None,
    opt_level: int = 0,
    shots: int = 4096,
    combo_batch_size: int = 256,
    max_batch: int = 1024,
    plan: Optional[dict] = None,
    verbose: bool = True,
    sampling_pec: Optional[PauliSamplingPEC] = None,
):
    sampling_pec = sampling_pec or PauliSamplingPEC(backend=backend, shots=shots, verbose=verbose)
    results = sampling_pec.run_tqg_pec_package_sampler(
        sampler=sampler,
        qcircuit=base_circuit,
        observables=observables,
        tqg_weights=tqg_weights,
        backend=backend,
        opt_level=opt_level,
        shots=shots,
        combo_batch_size=combo_batch_size,
        max_batch=max_batch,
        plan=plan,
        verbose=verbose,
    )
    diag = results.get("_trunc_diag", None)
    expectation = expct_from_pec_results(results, readout_weights)
    fidelity = state_fidelity(state, expectation)
    return fidelity, expectation, results, diag


def compare_pec_vs_nopec_sampler(
    sampler,
    state: Tuple[complex, complex],
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    readout_weights: Dict[str, np.ndarray],
    *,
    n_trials: int = 5,
    backend=None,
    opt_level: int = 0,
    shots: int = 4096,
    combo_batch_size: int = 256,
    max_batch: int = 1024,
    segment_id: int = 0,
    segment_count: int = 1,
    trunc_threshold: Optional[float] = 0.99,
    discard_first_cnot: bool = False,
    log_path: str = "./logs/tele_fidelity_pec_class_trials.jsonl",
    verbose: bool = True,
):
    qcircuit = teleportation_circuit(state)
    meas_qubits = infer_meas_qubits(qcircuit)
    observables = {f"meas{label}": obs(label) for label in pauli_labels_k(3)}

    sampling_pec = PauliSamplingPEC(backend=backend, shots=shots, verbose=verbose)
    plan = prepare_tqg_pec_plan_class(
        qcircuit=qcircuit,
        tqg_weights=tqg_weights,
        segment_id=segment_id,
        segment_count=segment_count,
        trunc_threshold=trunc_threshold,
        verbose=verbose,
        discard_first_cnot=discard_first_cnot,
        sampling_pec=sampling_pec,
    )

    fidelity_pec = []
    fidelity_nopec = []
    meas_pec_records = []
    meas_nopec_records = []
    diag_records = []

    _append_jsonl(
        log_path,
        {
            "type": "run_header",
            "ts": time.time(),
            "n_trials": n_trials,
            "shots": shots,
            "combo_batch_size": combo_batch_size,
            "max_batch": max_batch,
            "segment_id": segment_id,
            "segment_count": segment_count,
            "trunc_threshold": trunc_threshold,
            "discard_first_cnot": discard_first_cnot,
            "meas_qubits": meas_qubits,
            "plan_diag": plan.get("diag", None),
        },
    )

    for i in range(1, n_trials + 1):
        fid_nopec, meas_nopec = one_trial_nopec_sampler(
            sampler,
            state,
            qcircuit,
            backend=backend,
            opt_level=opt_level,
            shots=shots,
            meas_qubits=meas_qubits,
        )
        fidelity_nopec.append(fid_nopec)
        meas_nopec_records.append(meas_nopec)
        if verbose:
            print(f"[{i}/{n_trials}] F(noPEC)={fid_nopec:.8f}")

    for i in range(1, n_trials + 1):
        t0 = time.perf_counter()
        fid_pec, _, meas_pec, diag = one_trial_pec_sampler(
            sampler,
            state,
            qcircuit,
            observables,
            tqg_weights=tqg_weights,
            readout_weights=readout_weights,
            backend=backend,
            opt_level=opt_level,
            shots=shots,
            combo_batch_size=combo_batch_size,
            max_batch=max_batch,
            plan=plan,
            verbose=verbose,
            sampling_pec=sampling_pec,
        )
        elapsed = time.perf_counter() - t0
        fidelity_pec.append(fid_pec)
        meas_pec_records.append(meas_pec)
        diag_records.append(diag)
        _append_jsonl(
            log_path,
            {
                "type": "trial",
                "ts": time.time(),
                "trial": i,
                "fid_pec": float(fid_pec),
                "meas_pec": meas_pec,
                "diag": diag,
                "elapsed_s": elapsed,
            },
        )
        if verbose:
            print(f"[{i}/{n_trials}] F(PEC)={fid_pec:.8f}")

    fidelity_pec = np.asarray(fidelity_pec, dtype=float)
    fidelity_nopec = np.asarray(fidelity_nopec, dtype=float)
    summary = {
        "PEC_mean": float(fidelity_pec.mean()) if len(fidelity_pec) else 0.0,
        "PEC_std": float(fidelity_pec.std(ddof=1)) if len(fidelity_pec) > 1 else 0.0,
        "noPEC_mean": float(fidelity_nopec.mean()) if len(fidelity_nopec) else 0.0,
        "noPEC_std": float(fidelity_nopec.std(ddof=1)) if len(fidelity_nopec) > 1 else 0.0,
    }
    return fidelity_pec, fidelity_nopec, summary, meas_pec_records, meas_nopec_records, diag_records
