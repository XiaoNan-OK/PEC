from __future__ import annotations

import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp


class PauliSamplingPEC:
    def __init__(self, backend=None, shots: int = 1024, verbose: bool = False):
        self.backend = backend
        self.shots = shots
        self.verbose = verbose
        self.pauli_alphabet = ["I", "X", "Y", "Z"]
        self._letter_to_xz = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
        self._xz_to_letter = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}

    def _ensure_quantum_circuit(self, obj) -> QuantumCircuit:
        if isinstance(obj, QuantumCircuit):
            return obj
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], QuantumCircuit):
            return obj[0]
        clsname = obj.__class__.__name__
        raise TypeError(f"Expected QuantumCircuit, got {type(obj)} ({clsname}).")

    def circuit_information(self, circuit: QuantumCircuit) -> Dict[str, Any]:
        circ = self._ensure_quantum_circuit(circuit)
        active = set()
        cnots = []
        for data_idx, inst in enumerate(circ.data):
            op = getattr(inst, "operation", inst[0])
            qargs = getattr(inst, "qubits", inst[1])
            if op.name != "measure":
                for qubit in qargs:
                    active.add(circ.find_bit(qubit).index)
            if op.name in ("cx", "cnot"):
                control = circ.find_bit(qargs[0]).index
                target = circ.find_bit(qargs[1]).index
                cnots.append({"data_idx": data_idx, "control": control, "target": target})
        active_qubits = sorted(active)
        return {
            "active_qubits": active_qubits,
            "cnot_list": cnots,
            "phys2active": {q: i for i, q in enumerate(active_qubits)},
        }

    def unique_cnot_keys(self, circuit: QuantumCircuit) -> List[Tuple[int, int]]:
        seen = set()
        keys = []
        for rec in self.circuit_information(circuit)["cnot_list"]:
            key = (rec["control"], rec["target"])
            if key not in seen:
                seen.add(key)
                keys.append(key)
        return keys

    def enum_pauli_labels(self, n: int) -> List[str]:
        return ["".join(p) for p in product(self.pauli_alphabet, repeat=n)]

    def _label_to_xz(self, label: str) -> Tuple[List[int], List[int]]:
        x_bits = []
        z_bits = []
        for char in label:
            x_val, z_val = self._letter_to_xz[char]
            x_bits.append(x_val)
            z_bits.append(z_val)
        return x_bits, z_bits

    def _xz_to_label(self, x_bits: List[int], z_bits: List[int]) -> str:
        return "".join(self._xz_to_letter[(x, z)] for x, z in zip(x_bits, z_bits))

    def pauli_conj_by_cnot(
        self,
        label: str,
        control: int,
        target: int,
        active_map: Optional[Dict[int, int]] = None,
    ) -> str:
        x_bits, z_bits = self._label_to_xz(label)
        if active_map is not None:
            if control not in active_map or target not in active_map:
                return label
            control = active_map[control]
            target = active_map[target]
        x_bits[target] ^= x_bits[control]
        z_bits[control] ^= z_bits[target]
        return self._xz_to_label(x_bits, z_bits)

    def apply_pauli_layer(
        self,
        qc: QuantumCircuit,
        label: str,
        active_qubits: Optional[List[int]] = None,
    ) -> None:
        if active_qubits is None:
            active_qubits = list(range(len(label)))
        if len(active_qubits) != len(label):
            raise ValueError("Length mismatch between label and active_qubits.")

        for idx, char in enumerate(label):
            qubit = active_qubits[idx]
            if char == "I":
                continue
            if char == "X":
                qc.x(qubit)
            elif char == "Y":
                qc.y(qubit)
            elif char == "Z":
                qc.z(qubit)

    def _prepare_idx2pair(
        self,
        cnot_list: List[Dict[str, int]],
        label_strs: List[str],
        phys2active: Optional[Dict[int, int]] = None,
    ) -> Dict[int, Tuple[str, str]]:
        idx2pair = {}
        for label, rec in zip(label_strs, cnot_list):
            post_label = self.pauli_conj_by_cnot(
                label,
                rec["control"],
                rec["target"],
                active_map=phys2active,
            )
            idx2pair[rec["data_idx"]] = (label, post_label)
        return idx2pair

    def build_threshold_combo_set_by_abs(
        self,
        qcircuit: QuantumCircuit,
        tqg_weights: Dict[Tuple[int, int], np.ndarray],
        *,
        discard_first_cnot: bool = True,
        threshold: float = 0.99,
    ):
        info = self.circuit_information(qcircuit)
        active_qubits = info["active_qubits"]
        pre_labels = self.enum_pauli_labels(len(active_qubits))
        full_cnot_list = info["cnot_list"]
        twirl_cnot_list = full_cnot_list[1:] if discard_first_cnot else full_cnot_list
        n_cnot = len(twirl_cnot_list)
        if n_cnot <= 0:
            raise ValueError("No CNOTs selected for twirling.")

        n_labels = len(pre_labels)
        wvecs = []
        for rec in twirl_cnot_list:
            key = (rec["control"], rec["target"])
            if key not in tqg_weights:
                raise ValueError(f"Missing weights for CNOT {key}.")
            weight_vec = np.asarray(tqg_weights[key], dtype=float)
            if len(weight_vec) != n_labels:
                raise ValueError(f"weights[{key}] length mismatch: {len(weight_vec)} != {n_labels}")
            wvecs.append(weight_vec)

        idx_tuples = []
        w_prods = []
        for idx_tuple in product(range(n_labels), repeat=n_cnot):
            w_prod = 1.0
            for cnot_idx, label_idx in enumerate(idx_tuple):
                w_prod *= float(wvecs[cnot_idx][label_idx])
            if w_prod != 0.0:
                idx_tuples.append(idx_tuple)
                w_prods.append(w_prod)

        w_prods = np.asarray(w_prods, dtype=float)
        abs_w = np.abs(w_prods)
        order = np.argsort(abs_w)[::-1]
        idx_sorted = [idx_tuples[i] for i in order]
        abs_w_sorted = abs_w[order]
        w_sorted = w_prods[order]

        gamma_total = float(abs_w_sorted.sum())
        if gamma_total <= 0:
            raise ValueError("gamma_total is 0. Check weights.")
        cum_ratio = np.cumsum(abs_w_sorted) / gamma_total
        keep_k = int(np.searchsorted(cum_ratio, threshold, side="left") + 1)
        keep_k = min(keep_k, len(idx_sorted))

        diag = {
            "twirled_cnot_count": n_cnot,
            "nonzero_combos": int(len(idx_sorted)),
            "gamma_total": gamma_total,
            "threshold": float(threshold),
            "keep_K": int(keep_k),
            "kept_ratio": float(cum_ratio[keep_k - 1]) if keep_k > 0 else 0.0,
            "eliminated_ratio": float(1.0 - cum_ratio[keep_k - 1]) if keep_k > 0 else 1.0,
            "plot_signed_w_sorted": w_sorted,
            "plot_abs_w_sorted": abs_w_sorted,
            "plot_cum_ratio": cum_ratio,
        }

        return twirl_cnot_list, pre_labels, idx_sorted[:keep_k], w_sorted[:keep_k], diag

    def _rotate_to_z_for_label(self, qc: QuantumCircuit, qubit: int, char: str) -> None:
        if char == "X":
            qc.h(qubit)
        elif char == "Y":
            qc.sdg(qubit)
            qc.h(qubit)

    def _attach_measure_active_only(self, qc: QuantumCircuit, active_qubits: List[int]) -> None:
        if not active_qubits:
            return
        creg = ClassicalRegister(len(active_qubits), "c_meas")
        qc.add_register(creg)
        for classical_idx, qubit in enumerate(active_qubits):
            qc.measure(qubit, creg[classical_idx])

    def _extract_quasi_or_counts_list(self, res) -> List[dict]:
        data = getattr(res, "data", None)
        if data is not None:
            for reg_name in ("c_meas", "c"):
                reg = getattr(data, reg_name, None)
                if reg is not None and hasattr(reg, "get_counts"):
                    return [reg.get_counts()]
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

        try:
            out = []
            for i in range(len(res)):
                out.extend(self._extract_quasi_or_counts_list(res[i]))
            if out:
                return out
        except Exception:
            pass

        raise TypeError(f"Unsupported SamplerResult format: {type(res)}")

    def _bind_sampler_to_backend(self, sampler, backend):
        if backend is None:
            return sampler
        if hasattr(sampler, "mode"):
            try:
                sampler.mode = backend
            except Exception:
                pass
        return sampler

    def _strip_empty_classical_registers(self, circuit: QuantumCircuit) -> QuantumCircuit:
        used_clbits = []
        seen = set()
        for inst in circuit.data:
            cargs = getattr(inst, "clbits", inst[2] if len(inst) > 2 else [])
            for cbit in cargs:
                key = id(cbit)
                if key not in seen:
                    seen.add(key)
                    used_clbits.append(cbit)

        needs_rebuild = any(creg.size == 0 for creg in circuit.cregs) or len(used_clbits) != len(circuit.clbits)
        if not needs_rebuild:
            return circuit

        cleaned = QuantumCircuit(circuit.num_qubits, len(used_clbits), name=circuit.name)
        cleaned.global_phase = circuit.global_phase
        cleaned.metadata = dict(circuit.metadata) if circuit.metadata is not None else None

        qubit_map = {old_bit: cleaned.qubits[idx] for idx, old_bit in enumerate(circuit.qubits)}
        clbit_map = {old_bit: cleaned.clbits[idx] for idx, old_bit in enumerate(used_clbits)}

        for inst in circuit.data:
            op = getattr(inst, "operation", inst[0])
            qargs = getattr(inst, "qubits", inst[1])
            cargs = getattr(inst, "clbits", inst[2] if len(inst) > 2 else [])
            cleaned.append(
                op,
                [qubit_map[qarg] for qarg in qargs],
                [clbit_map[carg] for carg in cargs if carg in clbit_map],
            )

        return cleaned

    def _sanitize_circuits_for_runtime(self, circuits: List[QuantumCircuit]) -> List[QuantumCircuit]:
        return [self._strip_empty_classical_registers(circuit) for circuit in circuits]

    def _run_sampler_job(self, sampler, circuits, shots):
        sanitized_circuits = self._sanitize_circuits_for_runtime(circuits)
        return sampler.run(sanitized_circuits, shots=shots)

    def _exp_from_distribution(self, dist) -> float:
        try:
            items = dist.items()
        except Exception:
            dist = dict(dist)
            items = dist.items()
        total = float(sum(float(value) for _, value in items))
        if total <= 0:
            return 1.0

        exp_value = 0.0
        for key, value in dist.items():
            prob = float(value) / total
            if isinstance(key, str):
                ones = key.replace(" ", "").count("1")
            else:
                ones = int(key).bit_count()
            exp_value += prob * (-1.0 if (ones & 1) else 1.0)
        return float(exp_value)

    def _twirled_variant_for_all_cnot(
        self,
        orig: QuantumCircuit,
        idx2pair: Dict[int, Tuple[str, str]],
        active_qubits: List[int],
    ) -> QuantumCircuit:
        qc = QuantumCircuit(orig.num_qubits, orig.num_clbits, name=f"{orig.name or 'circ'}|twirl_all")
        for data_idx, inst in enumerate(orig.data):
            op = getattr(inst, "operation", inst[0])
            qargs = getattr(inst, "qubits", inst[1])
            cargs = getattr(inst, "clbits", [])
            pair = idx2pair.get(data_idx)
            if pair is None:
                qc.append(op, qargs, cargs)
                continue
            pre_label, post_label = pair
            self.apply_pauli_layer(qc, pre_label, active_qubits)
            qc.append(op, qargs, cargs)
            self.apply_pauli_layer(qc, post_label, active_qubits)
        return qc

    def _meas_circuit_for_observable(
        self,
        base: QuantumCircuit,
        obs: SparsePauliOp,
        active_qubits: Optional[List[int]] = None,
    ) -> Optional[QuantumCircuit]:
        label = obs.to_list()[0][0][::-1]
        if active_qubits is None or len(label) == base.num_qubits:
            qubit_map = list(range(len(label)))
        else:
            if len(label) != len(active_qubits):
                raise ValueError("Observable label length does not match active_qubits.")
            qubit_map = active_qubits

        meas_qubits = [qubit_map[i] for i, char in enumerate(label) if char != "I"]
        if not meas_qubits:
            return None

        qc = base.copy()
        for idx, char in enumerate(label):
            if char != "I":
                self._rotate_to_z_for_label(qc, qubit_map[idx], char)
        self._attach_measure_active_only(qc, meas_qubits)
        return qc

    def _make_truncation_pairs(
        self,
        *,
        tqg_weights: Dict[Tuple[int, int], np.ndarray],
        twirl_cnot_list: List[Dict[str, int]],
        n_labels: int,
        trunc_threshold: float = 0.99,
        precompute_wprod: bool = True,
    ):
        idx_tuples = []
        w_prods = []
        for idx_tuple in product(range(n_labels), repeat=len(twirl_cnot_list)):
            weight = 1.0
            for cnot_idx, label_idx in enumerate(idx_tuple):
                rec = twirl_cnot_list[cnot_idx]
                key = (rec["control"], rec["target"])
                weight *= float(tqg_weights[key][label_idx])
            if weight != 0.0:
                idx_tuples.append(idx_tuple)
                w_prods.append(weight)

        if not w_prods:
            diag = {
                "twirled_cnot_count": len(twirl_cnot_list),
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
            return [], diag

        w_prods = np.asarray(w_prods, dtype=float)
        abs_w = np.abs(w_prods)
        order = np.argsort(abs_w)[::-1]
        abs_w_sorted = abs_w[order]
        w_sorted = w_prods[order]
        idx_sorted = [idx_tuples[i] for i in order]
        gamma_total = float(abs_w_sorted.sum())
        cum_ratio = np.cumsum(abs_w_sorted) / gamma_total
        keep_k = int(np.searchsorted(cum_ratio, trunc_threshold, side="left") + 1)
        keep_k = min(keep_k, len(idx_sorted))

        kept_idx = idx_sorted[:keep_k]
        kept_w = w_sorted[:keep_k]
        if precompute_wprod:
            it_pairs = list(zip(kept_idx, map(float, kept_w)))
        else:
            it_pairs = [(idx_tuple, None) for idx_tuple in kept_idx]

        diag = {
            "twirled_cnot_count": len(twirl_cnot_list),
            "nonzero_combos": len(idx_sorted),
            "gamma_total": gamma_total,
            "threshold": trunc_threshold,
            "keep_K": keep_k,
            "kept_ratio": float(cum_ratio[keep_k - 1]) if keep_k > 0 else 0.0,
            "plot_abs_w_sorted": abs_w_sorted,
            "plot_signed_w_sorted": w_sorted,
            "plot_cum_ratio": cum_ratio,
            "kept_idx_tuples": kept_idx,
            "kept_w_prods": kept_w,
        }
        return it_pairs, diag

    def build_tqg_pec_plan(
        self,
        qcircuit,
        tqg_weights: Dict[Tuple[int, int], np.ndarray],
        *,
        trunc_threshold: Optional[float] = None,
        precompute_wprod: bool = True,
    ) -> Dict[str, Any]:
        circ = self._ensure_quantum_circuit(qcircuit)
        info = self.circuit_information(circ)
        active_qubits = info["active_qubits"]
        pre_labels = self.enum_pauli_labels(len(active_qubits))
        twirl_cnot_list = info["cnot_list"]
        n_labels = len(pre_labels)

        for rec in twirl_cnot_list:
            key = (rec["control"], rec["target"])
            weight_vec = tqg_weights.get(key)
            if weight_vec is None or len(weight_vec) != n_labels:
                raise ValueError(f"weights_map[{key}] length error: expect {n_labels}.")

        plan = {
            "circ": circ,
            "info": info,
            "active_qubits": active_qubits,
            "pre_labels": pre_labels,
            "twirl_cnot_list": twirl_cnot_list,
            "N": n_labels,
            "n_cnot": len(twirl_cnot_list),
            "diag": None,
            "it_pairs": None,
            "precompute_wprod": precompute_wprod,
        }

        if trunc_threshold is not None:
            it_pairs, diag = self._make_truncation_pairs(
                tqg_weights=tqg_weights,
                twirl_cnot_list=twirl_cnot_list,
                n_labels=n_labels,
                trunc_threshold=trunc_threshold,
                precompute_wprod=precompute_wprod,
            )
            plan["it_pairs"] = it_pairs
            plan["diag"] = diag

        return plan

    def tqg_pec_batch_circuit_sampler(
        self,
        sampler,
        pending_circs: List[QuantumCircuit],
        pending_meta: List[Tuple[str, float]],
        results: Dict[str, float],
        *,
        backend=None,
        opt_level: int = 0,
        shots: Optional[int] = None,
        jobs: Optional[List[Any]] = None,
    ) -> None:
        if not pending_circs:
            return
        run_backend = self.backend if backend is None else backend
        run_shots = self.shots if shots is None else shots
        sampler = self._bind_sampler_to_backend(sampler, run_backend)
        sanitized_circs = self._sanitize_circuits_for_runtime(pending_circs)
        tr_circs = (
            transpile(sanitized_circs, backend=run_backend, optimization_level=opt_level)
            if run_backend is not None
            else sanitized_circs
        )
        job = self._run_sampler_job(sampler, tr_circs, run_shots)
        if jobs is not None:
            jobs.append(job)
        primitive_result = job.result()
        dists = self._extract_quasi_or_counts_list(primitive_result)
        if len(dists) != len(pending_circs):
            raise ValueError(f"Sampler returned {len(dists)} dists but expected {len(pending_circs)}")

        for dist, (obs_name, weight) in zip(dists, pending_meta):
            results[obs_name] += weight * self._exp_from_distribution(dist)

        pending_circs.clear()
        pending_meta.clear()

    def run_tqg_pec_package_sampler(
        self,
        sampler,
        qcircuit,
        observables: Dict[str, Any],
        tqg_weights: Dict[Tuple[int, int], np.ndarray],
        *,
        backend=None,
        opt_level: int = 0,
        shots: Optional[int] = None,
        combo_batch_size: int = 64,
        max_batch: int = 1024,
        plan: Optional[Dict[str, Any]] = None,
        verbose: Optional[bool] = None,
        return_jobs: bool = False,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        del combo_batch_size, kwargs
        run_verbose = self.verbose if verbose is None else verbose
        run_shots = self.shots if shots is None else shots
        run_backend = self.backend if backend is None else backend

        if plan is None:
            plan = self.build_tqg_pec_plan(qcircuit, tqg_weights, precompute_wprod=False)

        circ = plan["circ"]
        info = plan["info"]
        active_qubits = plan["active_qubits"]
        pre_labels = plan["pre_labels"]
        twirl_cnot_list = plan["twirl_cnot_list"]
        n_labels = plan["N"]

        results = {name: 0.0 for name in observables}
        pending_circs: List[QuantumCircuit] = []
        pending_meta: List[Tuple[str, float]] = []
        jobs: List[Any] = []
        executed = 0
        start_time = time.perf_counter()

        if plan.get("it_pairs") is not None:
            it_pairs = plan["it_pairs"]
        else:
            it_pairs = [(idx_tuple, None) for idx_tuple in product(range(n_labels), repeat=len(twirl_cnot_list))]

        total_terms = len(it_pairs)
        for term_idx, (idx_tuple, cached_weight) in enumerate(it_pairs, start=1):
            labels = [pre_labels[label_idx] for label_idx in idx_tuple]
            if cached_weight is None:
                weight = 1.0
                for cnot_idx, label_idx in enumerate(idx_tuple):
                    rec = twirl_cnot_list[cnot_idx]
                    key = (rec["control"], rec["target"])
                    weight *= float(tqg_weights[key][label_idx])
            else:
                weight = float(cached_weight)

            idx2pair = self._prepare_idx2pair(twirl_cnot_list, labels, phys2active=info["phys2active"])
            base_twirl = self._twirled_variant_for_all_cnot(circ, idx2pair, active_qubits)

            for name, obs in observables.items():
                meas_circuit = self._meas_circuit_for_observable(base_twirl, obs, active_qubits)
                if meas_circuit is None:
                    results[name] += weight
                    continue
                pending_circs.append(meas_circuit)
                pending_meta.append((name, weight))
                if len(pending_circs) >= max_batch:
                    self.tqg_pec_batch_circuit_sampler(
                        sampler,
                        pending_circs,
                        pending_meta,
                        results,
                        backend=run_backend,
                        opt_level=opt_level,
                        shots=run_shots,
                        jobs=jobs if return_jobs else None,
                    )
                    executed += max_batch

            if run_verbose and term_idx % 50 == 0:
                elapsed = time.perf_counter() - start_time
                print(f"[PEC Sampling] processed {term_idx}/{total_terms} terms in {elapsed:.2f}s", flush=True)

        if pending_circs:
            executed += len(pending_circs)
            self.tqg_pec_batch_circuit_sampler(
                sampler,
                pending_circs,
                pending_meta,
                results,
                backend=run_backend,
                opt_level=opt_level,
                shots=run_shots,
                jobs=jobs if return_jobs else None,
            )

        if run_verbose:
            elapsed = time.perf_counter() - start_time
            print(f"[PEC Sampling] completed {total_terms} terms, ran {executed} circuits in {elapsed:.2f}s", flush=True)

        out = {name: {"obs": name, "value": float(value)} for name, value in results.items()}
        if plan.get("diag") is not None:
            out["_trunc_diag"] = plan["diag"]
        if return_jobs:
            out["_jobs"] = jobs
            out["_job_ids"] = [job.job_id() for job in jobs if hasattr(job, "job_id")]
        return out


_DEFAULT_PEC_SAMPLER = PauliSamplingPEC()


def _resolve_sampling_pec(backend=None, shots: int = 1024, verbose: bool = False) -> PauliSamplingPEC:
    return PauliSamplingPEC(backend=backend, shots=shots, verbose=verbose)


def _ensure_quantum_circuit(obj):
    return _DEFAULT_PEC_SAMPLER._ensure_quantum_circuit(obj)


def circuit_information(circuit: QuantumCircuit):
    return _DEFAULT_PEC_SAMPLER.circuit_information(circuit)


def enum_pauli_labels(n: int) -> List[str]:
    return _DEFAULT_PEC_SAMPLER.enum_pauli_labels(n)


def pauli_conj_by_cnot(label: str, c: int, t: int, active: Optional[Dict[int, int]] = None) -> str:
    return _DEFAULT_PEC_SAMPLER.pauli_conj_by_cnot(label, c, t, active_map=active)


def apply_pauli_layer(qc: QuantumCircuit, label: str, active_qubits: Optional[List[int]] = None) -> None:
    _DEFAULT_PEC_SAMPLER.apply_pauli_layer(qc, label, active_qubits)


def unique_cnot_keys(input_circuit: QuantumCircuit):
    return _DEFAULT_PEC_SAMPLER.unique_cnot_keys(input_circuit)


def build_threshold_combo_set_by_abs(
    qcircuit: QuantumCircuit,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    discard_first_cnot: bool = True,
    threshold: float = 0.99,
    verbose: bool = True,
):
    pec = _resolve_sampling_pec(verbose=verbose)
    return pec.build_threshold_combo_set_by_abs(
        qcircuit,
        tqg_weights,
        discard_first_cnot=discard_first_cnot,
        threshold=threshold,
    )


def build_tqg_pec_plan(
    qcircuit,
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    trunc_threshold: Optional[float] = None,
    precompute_wprod: bool = True,
    verbose: bool = False,
):
    pec = _resolve_sampling_pec(verbose=verbose)
    return pec.build_tqg_pec_plan(
        qcircuit,
        tqg_weights,
        trunc_threshold=trunc_threshold,
        precompute_wprod=precompute_wprod,
    )


def tqg_pec_batch_circuit_sampler(
    sampler,
    pending_circs: List[QuantumCircuit],
    pending_meta: List[Tuple[str, float]],
    results: Dict[str, float],
    *,
    backend=None,
    opt_level: int = 0,
    shots: int = 1024,
    jobs: Optional[List[Any]] = None,
):
    pec = _resolve_sampling_pec(backend=backend, shots=shots)
    return pec.tqg_pec_batch_circuit_sampler(
        sampler,
        pending_circs,
        pending_meta,
        results,
        backend=backend,
        opt_level=opt_level,
        shots=shots,
        jobs=jobs
    )


def run_tqg_pec_package_sampler(
    sampler,
    qcircuit,
    observables: Dict[str, Any],
    tqg_weights: Dict[Tuple[int, int], np.ndarray],
    *,
    backend=None,
    opt_level: int = 0,
    shots: int = 1024,
    combo_batch_size: int = 64,
    max_batch: int = 1024,
    plan: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    return_jobs: bool = False,
    **kwargs,
):
    pec = _resolve_sampling_pec(backend=backend, shots=shots, verbose=verbose)
    return pec.run_tqg_pec_package_sampler(
        sampler,
        qcircuit,
        observables,
        tqg_weights,
        backend=backend,
        opt_level=opt_level,
        shots=shots,
        combo_batch_size=combo_batch_size,
        max_batch=max_batch,
        plan=plan,
        verbose=verbose,
        return_jobs=return_jobs,
        **kwargs,
    )
