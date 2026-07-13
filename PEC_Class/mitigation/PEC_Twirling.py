import numpy as np
from numpy.linalg import inv
from itertools import product
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_aer import AerSimulator

class PauliTwirlingPEC:
    def __init__(self, backend=None, shots=1024, verbose=False):
        self.backend = backend
        self.shots = shots
        self.verbose = verbose
        self.pauli_alphabet = ['I', 'X', 'Y', 'Z']
        self.prep_labels = ['0', '1', '+', 'R']
        self._letter_to_xz = {'I': (0, 0), 'X': (1, 0), 'Y': (1, 1), 'Z': (0, 1)}
        self._xz_to_letter = {(0, 0): 'I', (1, 0): 'X', (1, 1): 'Y', (0, 1): 'Z'}
        self._a_single = np.array(
            [[1, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [1, -1, 0, 0]],
            dtype=float,
        )

    def _ensure_quantum_circuit(self, obj) -> QuantumCircuit:
        if isinstance(obj, QuantumCircuit):
            return obj
        if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], QuantumCircuit):
            return obj[0]
        clsname = obj.__class__.__name__
        raise TypeError(f"Expected QuantumCircuit, got {type(obj)} ({clsname}).")

    def _get_active_qubits(self, circ: QuantumCircuit) -> list[int]:
        active = set()
        for inst in circ.data:
            operation = getattr(inst, "operation", inst[0])
            qubits = getattr(inst, "qubits", inst[1])
            if operation.name == "measure":
                continue
            for qubit in qubits:
                try:
                    idx = circ.find_bit(qubit).index
                except Exception:
                    idx = circ.qubits.index(qubit)
                active.add(idx)
        return sorted(active)

    def _find_unique_cnot_keys(self, circ: QuantumCircuit) -> list[tuple[int, int]]:
        seen = set()
        out = []
        for inst in circ.data:
            operation = getattr(inst, "operation", inst[0])
            qubits = getattr(inst, "qubits", inst[1])
            if operation.name not in ("cx", "cnot"):
                continue
            control = circ.find_bit(qubits[0]).index
            target = circ.find_bit(qubits[1]).index
            key = (control, target)
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def _enum_initial_labels(self, n: int) -> list[str]:
        return [''.join(p) for p in product(self.prep_labels, repeat=n)]

    def _enum_pauli_labels(self, n: int) -> list[str]:
        return [''.join(p) for p in product(self.pauli_alphabet, repeat=n)]

    def _init_circuit_from_label(self, n: int, label: str) -> QuantumCircuit:
        qc = QuantumCircuit(n, name=f"init_{label}")
        for qubit, char in enumerate(label):
            if char == '1':
                qc.x(qubit)
            elif char == '+':
                qc.h(qubit)
            elif char == 'R':
                qc.h(qubit)
                qc.s(qubit)
        return qc

    def _label_to_xz(self, label: str) -> tuple[list[int], list[int]]:
        x, z = [], []
        for char in label:
            xi, zi = self._letter_to_xz[char]
            x.append(xi)
            z.append(zi)
        return x, z

    def _xz_to_label(self, x: list[int], z: list[int]) -> str:
        return ''.join(self._xz_to_letter[(xi, zi)] for xi, zi in zip(x, z))

    def _pauli_conj_by_cnot(self, label: str, c_loc: int, t_loc: int) -> str:
        x, z = self._label_to_xz(label)
        x[t_loc] ^= x[c_loc]
        z[c_loc] ^= z[t_loc]
        return self._xz_to_label(x, z)

    def _apply_pauli_layer(self, qc: QuantumCircuit, label: str) -> None:
        for qubit, char in enumerate(label):
            if char == 'I':
                continue
            if char == 'X':
                qc.x(qubit)
            elif char == 'Y':
                qc.y(qubit)
            elif char == 'Z':
                qc.z(qubit)

    def _rotate_to_z_for_label(self, qc: QuantumCircuit, label: str) -> None:
        for qubit, char in enumerate(label):
            if char == 'X':
                qc.h(qubit)
            elif char == 'Y':
                qc.sdg(qubit)
                qc.h(qubit)

    def _attach_measure_active_only(self, qc: QuantumCircuit, active: list[int]) -> None:
        creg = ClassicalRegister(len(active), "c")
        qc.add_register(creg)
        for classical_idx, qubit in enumerate(active):
            qc.measure(qubit, creg[classical_idx])

    def _extract_quasi_or_counts_list(self, samp_res):
        data = getattr(samp_res, "data", None)
        if data is not None:
            cobj = getattr(data, "c", None)
            if cobj is not None and hasattr(cobj, "get_counts"):
                return [cobj.get_counts()]

            for name in dir(data):
                if name.startswith("_") or name in ("meas", "quasi_dists"):
                    continue
                reg = getattr(data, name, None)
                if hasattr(reg, "get_counts"):
                    try:
                        return [reg.get_counts()]
                    except Exception:
                        pass

            if hasattr(data, "quasi_dists"):
                return list(data.quasi_dists)
            if hasattr(data, "meas") and hasattr(data.meas, "quasi_dists"):
                return list(data.meas.quasi_dists)
            if isinstance(data, dict):
                if "quasi_dists" in data:
                    return list(data["quasi_dists"])
                if "meas" in data and isinstance(data["meas"], dict) and "quasi_dists" in data["meas"]:
                    return list(data["meas"]["quasi_dists"])

        try:
            if hasattr(samp_res, "__iter__") and not isinstance(samp_res, (dict, str)):
                outs = []
                for rec in list(samp_res):
                    outs.extend(self._extract_quasi_or_counts_list(rec))
                if outs:
                    return outs
        except Exception:
            pass

        for attr in ("quasi_dists", "quasi_distributions"):
            qd = getattr(samp_res, attr, None)
            if qd is not None:
                return list(qd)

        rs = getattr(samp_res, "results", None)
        if rs and hasattr(rs[0], "data"):
            d = rs[0].data
            if hasattr(d, "quasi_dists"):
                return list(d.quasi_dists)
            if hasattr(d, "meas") and hasattr(d.meas, "quasi_dists"):
                return list(d.meas.quasi_dists)

        raise TypeError(f"Unsupported SamplerResult format: {type(samp_res)}")

    def _to_plain_dict(self, obj):
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return dict(obj)
        try:
            from dataclasses import asdict

            return asdict(obj)
        except Exception:
            pass
        if hasattr(obj, "__dict__"):
            return {
                key: value
                for key, value in vars(obj).items()
                if not key.startswith("_")
            }
        return {}

    def _build_parallel_sampler_config(self, sampler, shots=None):
        run_shots = self.shots if shots is None else shots
        options_obj = getattr(sampler, "options", None)
        options_dict = self._to_plain_dict(options_obj)
        backend_options = dict(options_dict.get("backend_options", {}) or {})
        run_options = dict(options_dict.get("run_options", {}) or {})
        run_options.setdefault("shots", run_shots)
        return {
            "backend_options": backend_options,
            "run_options": run_options,
        }

    def _bind_sampler_to_backend(self, sampler, backend):
        if backend is None:
            return sampler
        if hasattr(sampler, "mode"):
            try:
                sampler.mode = backend
            except Exception:
                pass
        return sampler

    def _resolve_submission_backend(self, sampler, backend=None):
        if backend is not None:
            return backend
        if self.backend is not None:
            return self.backend
        sampler_backend = getattr(sampler, "_backend", None)
        if sampler_backend is not None:
            return sampler_backend
        sampler_mode = getattr(sampler, "mode", None)
        if sampler_mode is not None:
            return sampler_mode
        return None

    def _prepare_circuits_for_sampler(self, circuits, backend=None):
        if backend is None:
            return circuits
        return transpile(circuits, backend=backend, optimization_level=0)

    def _run_sampler_job(self, sampler, circuits, shots):
        return sampler.run(circuits, shots=shots)

    def _exp_from_distribution(self, dist) -> float:
        total = sum(dist.values())
        if total <= 0:
            return 1.0
        exp = 0.0
        for key, value in dist.items():
            prob = value / total
            ones = key.count('1') if isinstance(key, str) else int(key).bit_count()
            exp += prob * (-1.0 if (ones & 1) else 1.0)
        return float(exp)

    def _expand_label_to_n(self, label_m: str, active: list[int], n: int, fill: str) -> str:
        out = [fill] * n
        for char, qubit in zip(label_m, active):
            out[qubit] = char
        return ''.join(out)

    def _local_indices(self, active: list[int], c: int, t: int) -> tuple[int, int]:
        to_loc = {global_idx: local_idx for local_idx, global_idx in enumerate(active)}
        return to_loc[c], to_loc[t]

    def _build_twirled_circuit_bypass_n(
        self,
        n: int,
        init_label_m: str,
        pre_label_m: str,
        active: list[int],
        c: int,
        t: int,
    ) -> QuantumCircuit:
        c_loc, t_loc = self._local_indices(active, c, t)
        init_label_n = self._expand_label_to_n(init_label_m, active, n, fill='0')
        pre_label_n = self._expand_label_to_n(pre_label_m, active, n, fill='I')
        post_label_m = self._pauli_conj_by_cnot(pre_label_m, c_loc, t_loc)
        post_label_n = self._expand_label_to_n(post_label_m, active, n, fill='I')

        qc = QuantumCircuit(n, name=f"twirl(c{c},t{t})")
        qc.compose(self._init_circuit_from_label(n, init_label_n), inplace=True)
        self._apply_pauli_layer(qc, pre_label_n)
        qc.cx(c, t)
        self._apply_pauli_layer(qc, post_label_n)
        return qc

    def _build_meas_circuit_for_observable_n(
        self,
        base_n: QuantumCircuit,
        obs_label_m: str,
        active: list[int],
    ) -> QuantumCircuit:
        n = base_n.num_qubits
        obs_label_n = self._expand_label_to_n(obs_label_m, active, n, fill='I')
        qc = QuantumCircuit(n, name=f"{base_n.name}__meas_{obs_label_m}")
        qc.compose(base_n, inplace=True)
        self._rotate_to_z_for_label(qc, obs_label_n)
        active_meas = [qubit for qubit in active if obs_label_n[qubit] != 'I']
        self._attach_measure_active_only(qc, active_meas)
        return qc

    def _flush_sampler_batch(
        self,
        sampler,
        circs: list[QuantumCircuit],
        meta: list[tuple[int, int]],
        acc: np.ndarray,
        shots: int,
        backend=None,
        jobs=None,
    ) -> None:
        if not circs:
            return

        submit_backend = self._resolve_submission_backend(sampler, backend=backend)
        prepared_circs = self._prepare_circuits_for_sampler(circs, backend=submit_backend)

        job = self._run_sampler_job(sampler, prepared_circs, shots)
        if jobs is not None:
            jobs.append(job)
        res = job.result()
        dists = self._extract_quasi_or_counts_list(res)

        if len(dists) != len(prepared_circs):
            try:
                new_dists = []
                for idx in range(len(prepared_circs)):
                    new_dists.extend(self._extract_quasi_or_counts_list(res[idx]))
                dists = new_dists
            except Exception as exc:
                raise ValueError(
                    f"Sampler returned {len(dists)} distributions but expected {len(prepared_circs)}"
                ) from exc

        for qd, (_, obs_idx) in zip(dists, meta):
            acc[obs_idx] += self._exp_from_distribution(qd)

        circs.clear()
        meta.clear()

    def compute_matrix_for_tqg(
        self,
        sampler,
        input_circuit: QuantumCircuit,
        key: tuple[int, int],
        shots=None,
        max_batch: int = 1024,
        parallel: bool = False,
        max_workers=None,
        backend=None,
        return_jobs: bool = False,
    ):
        circ = self._ensure_quantum_circuit(input_circuit)
        run_shots = self.shots if shots is None else shots
        run_backend = self.backend if backend is None else backend
        sampler = self._bind_sampler_to_backend(sampler, run_backend)
        n = circ.num_qubits
        active = self._get_active_qubits(circ)
        m = len(active)
        c, t = key

        init_labels_m = self._enum_initial_labels(m)
        obs_labels_m = self._enum_pauli_labels(m)
        pre_labels_m = self._enum_pauli_labels(m)

        matrix = np.zeros((len(obs_labels_m), len(init_labels_m)), dtype=float)
        jobs = []

        if not parallel:
            for col_idx, init_lab_m in enumerate(init_labels_m):
                acc = np.zeros(len(obs_labels_m), dtype=float)
                pending_circs = []
                pending_meta = []

                for pre_idx, pre_m in enumerate(pre_labels_m):
                    qc_base = self._build_twirled_circuit_bypass_n(n, init_lab_m, pre_m, active, c, t)
                    for obs_idx, obs_lab_m in enumerate(obs_labels_m):
                        if all(char == 'I' for char in obs_lab_m):
                            acc[obs_idx] += 1.0
                            continue
                        qc_meas = self._build_meas_circuit_for_observable_n(qc_base, obs_lab_m, active)
                        pending_circs.append(qc_meas)
                        pending_meta.append((pre_idx, obs_idx))
                        if len(pending_circs) >= max_batch:
                            self._flush_sampler_batch(
                                sampler,
                                pending_circs,
                                pending_meta,
                                acc,
                                shots=run_shots,
                                backend=run_backend,
                                jobs=jobs if return_jobs else None,
                            )

                if pending_circs:
                    self._flush_sampler_batch(
                        sampler,
                        pending_circs,
                        pending_meta,
                        acc,
                        shots=run_shots,
                        backend=run_backend,
                        jobs=jobs if return_jobs else None,
                    )

                matrix[:, col_idx] = acc / float(len(pre_labels_m))
                if self.verbose:
                    print(f"(c{c},t{t}) {col_idx + 1}/{len(init_labels_m)} (active={active})")
        else:
            if run_backend is not None and not isinstance(run_backend, AerSimulator):
                raise ValueError("parallel=True currently supports only local Aer-based sampling, not real-device runtime sampling.")
            parallel_sampler_config = self._build_parallel_sampler_config(sampler, shots=run_shots)
            index_chunks = np.array_split(np.arange(len(init_labels_m)), max_workers or 1)

            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            if self.verbose:
                print(f"Parallelizing inside CNOT(control={c}, target={t}) with multiprocessing.")

            mp_context = mp.get_context("spawn")

            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                future_map = {}
                submitted = 0
                for chunk in index_chunks:
                    if len(chunk) == 0:
                        continue
                    submitted += 1
                    if self.verbose:
                        print(
                            f"(c{c},t{t}) submitted chunk {submitted}: "
                            f"columns {chunk[0] + 1}-{chunk[-1] + 1}/{len(init_labels_m)}"
                        )
                    future = executor.submit(
                        _compute_tqg_matrix_chunk_process,
                        circ,
                        key,
                        [init_labels_m[idx] for idx in chunk],
                        list(chunk),
                        active,
                        obs_labels_m,
                        pre_labels_m,
                        run_shots,
                        max_batch,
                        parallel_sampler_config,
                    )
                    future_map[future] = list(chunk)

                completed = 0
                for future in as_completed(future_map):
                    chunk_indices, chunk_matrix = future.result()
                    matrix[:, chunk_indices] = chunk_matrix
                    completed += 1
                    if self.verbose:
                        print(
                            f"(c{c},t{t}) finished chunk {completed}/{submitted}: "
                            f"columns {chunk_indices[0] + 1}-{chunk_indices[-1] + 1}/{len(init_labels_m)}"
                        )

        payload = (matrix, obs_labels_m, init_labels_m)
        if return_jobs:
            return payload + (jobs,)
        return payload

    def compute_tqg_matrices(
        self,
        sampler,
        input_circuit: QuantumCircuit,
        shots=None,
        obs_batch: int = 256,
        parallel: bool = False,
        max_workers=None,
        sampler_factory=None,
        backend=None,
        return_jobs: bool = False,
    ):
        circ = self._ensure_quantum_circuit(input_circuit)
        active = self._get_active_qubits(circ)
        m = len(active)
        matrices = {}
        row_labels = None
        col_labels = None
        cnot_keys = self._find_unique_cnot_keys(circ)
        all_jobs = {}

        for key in cnot_keys:
            if self.verbose:
                print(f"Characterizing CNOT(control={key[0]}, target={key[1]})...")
            matrix_output = self.compute_matrix_for_tqg(
                sampler,
                circ,
                key,
                shots=shots,
                max_batch=obs_batch,
                parallel=parallel,
                max_workers=max_workers,
                backend=backend,
                return_jobs=return_jobs,
            )
            if return_jobs:
                matrix, row_labels, col_labels, jobs = matrix_output
                all_jobs[key] = jobs
            else:
                matrix, row_labels, col_labels = matrix_output
            matrices[key] = matrix
            if self.verbose:
                print(f"Finished CNOT(control={key[0]}, target={key[1]}).")

        output = {
            "num_qubits": m,
            "row_labels": row_labels,
            "col_labels": col_labels,
            "matrices": matrices,
            "active_qubits": active,
        }
        if return_jobs:
            output["jobs"] = all_jobs
            output["job_ids"] = {
                key: [job.job_id() for job in jobs if hasattr(job, "job_id")]
                for key, jobs in all_jobs.items()
            }
        return output

    def _cnot_pauli_twirling_matrix(self, n: int, c_loc: int, t_loc: int):
        dim = 4 ** n
        labels = self._enum_pauli_labels(n)
        idx = {label: i for i, label in enumerate(labels)}
        matrix = np.zeros((dim, dim), dtype=float)
        for col, label in enumerate(labels):
            row = idx[self._pauli_conj_by_cnot(label, c_loc, t_loc)]
            matrix[row, col] = 1.0
        return matrix, labels

    def _kron_power_dense(self, mat: np.ndarray, n: int) -> np.ndarray:
        out = mat
        for _ in range(n - 1):
            out = np.kron(out, mat)
        return out

    def averaged_pauli_twirling_matrix(self, pack: dict, B: np.ndarray, *, active_qubits=None):
        m = pack["num_qubits"]
        active = pack.get("active_qubits") if active_qubits is None else active_qubits
        if active is None:
            raise ValueError("pack must provide active_qubits")

        a_inv = self._kron_power_dense(inv(self._a_single), m)
        b_inv = inv(B)
        out = {}

        for (c, t), G in pack["matrices"].items():
            c_loc, t_loc = self._local_indices(active, c, t)
            cx_matrix, _ = self._cnot_pauli_twirling_matrix(m, c_loc, t_loc)
            out[(c, t)] = np.asarray(cx_matrix.T @ b_inv @ G @ a_inv, dtype=float)

        return {
            "num_qubits": m,
            "active_qubits": active,
            "row_labels": pack["row_labels"],
            "col_labels": pack["col_labels"],
            "matrices": out,
        }

    def _labels_to_xz(self, labels: list[str]):
        x = np.zeros((len(labels), len(labels[0])), np.uint8)
        z = np.zeros_like(x)
        for row, label in enumerate(labels):
            x_row, z_row = self._label_to_xz(label)
            x[row, :] = x_row
            z[row, :] = z_row
        return x, z

    def build_commutation_transform(self, labels: list[str]) -> np.ndarray:
        x, z = self._labels_to_xz(labels)
        parity = (x @ z.T - z @ x.T) & 1
        return (1 - 2 * parity).astype(np.int8)

    def compute_weights_for_package(self, avg_pack: dict):
        transform = self.build_commutation_transform(avg_pack["row_labels"])
        out = {}
        for key, matrix in avg_pack["matrices"].items():
            out[key] = transform @ np.diag(matrix)
        return out

    def compute_inv_weights_for_package(self, avg_pack: dict, eps: float = 0.0):
        n = avg_pack["num_qubits"]
        transform = self.build_commutation_transform(avg_pack["row_labels"])
        transform_inv = transform.astype(float) / (4.0 ** n)
        out = {}
        for key, matrix in avg_pack["matrices"].items():
            diag = np.diag(matrix)
            if eps > 0:
                signs = np.sign(diag)
                signs[signs == 0] = 1
                diag = signs * np.maximum(np.abs(diag), eps)
            out[key] = transform_inv @ (1.0 / diag)
        return out


def _compute_tqg_matrix_chunk_process(
    input_circuit,
    key,
    init_labels_chunk,
    chunk_indices,
    active,
    obs_labels_m,
    pre_labels_m,
    shots,
    max_batch,
    parallel_sampler_config,
):
    backend_options = dict(parallel_sampler_config.get("backend_options", {}) or {})
    run_options = dict(parallel_sampler_config.get("run_options", {}) or {})
    local_sampler = AerSampler(
        options={
            "backend_options": backend_options,
            "run_options": run_options,
        }
    )
    local_sampler.mode = AerSimulator(**backend_options)

    worker = PauliTwirlingPEC(shots=shots, verbose=False)
    circ = worker._ensure_quantum_circuit(input_circuit)
    n = circ.num_qubits
    c, t = key
    chunk_matrix = np.zeros((len(obs_labels_m), len(init_labels_chunk)), dtype=float)

    for local_col_idx, init_lab_m in enumerate(init_labels_chunk):
        acc = np.zeros(len(obs_labels_m), dtype=float)
        pending_circs = []
        pending_meta = []

        for pre_idx, pre_m in enumerate(pre_labels_m):
            qc_base = worker._build_twirled_circuit_bypass_n(n, init_lab_m, pre_m, active, c, t)
            for obs_idx, obs_lab_m in enumerate(obs_labels_m):
                if all(char == 'I' for char in obs_lab_m):
                    acc[obs_idx] += 1.0
                    continue
                qc_meas = worker._build_meas_circuit_for_observable_n(qc_base, obs_lab_m, active)
                pending_circs.append(qc_meas)
                pending_meta.append((pre_idx, obs_idx))
                if len(pending_circs) >= max_batch:
                    worker._flush_sampler_batch(local_sampler, pending_circs, pending_meta, acc, shots=shots)

        if pending_circs:
            worker._flush_sampler_batch(local_sampler, pending_circs, pending_meta, acc, shots=shots)

        chunk_matrix[:, local_col_idx] = acc / float(len(pre_labels_m))

    return chunk_indices, chunk_matrix
