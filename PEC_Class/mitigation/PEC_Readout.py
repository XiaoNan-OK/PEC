import numpy as np
from numpy.linalg import inv
from itertools import product
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler


class ReadoutQEM:
    def __init__(self, backend=None, shots=1024, backend_options=None, verbose=False):
        self.backend = backend
        self.shots = shots
        self.backend_options = backend_options
        self.verbose = verbose
        self.pauli_alphabet = ['I', 'X', 'Y', 'Z']
        self.prep_labels = ['zero', 'one', 'plus', 'right']

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

    def _generate_pauli_strings(self, k: int) -> list[str]:
        return [''.join(p) for p in product(self.pauli_alphabet, repeat=k)]

    def _build_initial_states_on(self, active: list[int]) -> dict[str, tuple[str, ...]]:
        init_specs = {}
        for choices in product(self.prep_labels, repeat=len(active)):
            key = "_".join([f"q{active[i]}{choices[i]}" for i in range(len(active))])
            init_specs[key] = tuple(choices)
        return init_specs

    def _build_measurement_pauli_on(self, active: list[int]) -> dict[str, str]:
        return {f"meas{pauli}": pauli for pauli in self._generate_pauli_strings(len(active))}

    def _precompute_A_matrix(self, k: int) -> np.ndarray:
        a1 = np.array([
            [1, 1, 1, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, -1, 0, 0],
        ], dtype=float)
        out = np.array([1.0])
        for _ in range(k):
            out = np.kron(out, a1)
        return out

    def _apply_prep(self, qc: QuantumCircuit, active: list[int], choices: tuple[str, ...]) -> None:
        for idx, label in enumerate(choices):
            qubit = active[idx]
            if label == 'one':
                qc.x(qubit)
            elif label == 'plus':
                qc.h(qubit)
            elif label == 'right':
                qc.h(qubit)
                qc.s(qubit)

    def _apply_meas_basis(self, qc: QuantumCircuit, active: list[int], pauli_str: str) -> None:
        measure_indices = [i for i, char in enumerate(pauli_str) if char != 'I']
        if not measure_indices:
            return

        creg = ClassicalRegister(len(measure_indices), 'c_meas')
        qc.add_register(creg)

        c_idx = 0
        for i, char in enumerate(pauli_str):
            qubit = active[i]
            if char == 'I':
                continue
            if char == 'X':
                qc.h(qubit)
            elif char == 'Y':
                qc.sdg(qubit)
                qc.h(qubit)
            qc.measure(qubit, c_idx)
            c_idx += 1

    def _expectation_from_counts_z_only(self, counts: dict) -> float:
        total = sum(counts.values())
        probs = {bitstring: count / total for bitstring, count in counts.items()}
        exp_val = 0.0
        for bitstring, prob in probs.items():
            parity = bitstring.count('1')
            exp_val += ((-1) ** parity) * prob
        return float(exp_val)

    def _identity_pauli_expectation(self, pauli_str: str):
        if all(char == 'I' for char in pauli_str):
            return 1.0
        return None

    def _resolve_backend(self, backend=None, backend_options=None):
        if backend is not None:
            return backend
        options = backend_options if backend_options is not None else self.backend_options
        if options is not None:
            return AerSimulator(**options)
        if self.backend is not None:
            return self.backend
        return AerSimulator()

    def _resolve_sampler(self, sampler=None, backend_options=None, shots=None):
        if sampler is not None:
            return sampler
        options = backend_options if backend_options is not None else self.backend_options
        run_shots = self.shots if shots is None else shots
        return AerSampler(options={"backend_options": options, "run_options": {"shots": run_shots}})

    def _bind_sampler_to_backend(self, sampler, backend):
        if backend is None:
            return sampler
        if hasattr(sampler, "mode"):
            try:
                sampler.mode = backend
            except Exception:
                pass
        return sampler

    def _run_sampler_job(self, sampler, circuits, shots):
        return sampler.run(circuits, shots=shots)

    def _extract_counts(self, pub_result):
        data = pub_result.data
        for attr in ("c", "meas", "cr", "klass"):
            if hasattr(data, attr):
                bit_array = getattr(data, attr)
                if hasattr(bit_array, "get_counts"):
                    return bit_array.get_counts()
        raise AttributeError("Unable to extract counts from sampler result data.")

    def run_sampler_on_subsystem(
        self,
        base_circuit: QuantumCircuit,
        sampler=None,
        backend=None,
        shots=None,
        batch_mode: bool = True,
        backend_options=None,
        return_jobs: bool = False,
    ):
        active = self._get_active_qubits(base_circuit)
        if not active:
            raise ValueError("No active qubits detected.")

        run_shots = self.shots if shots is None else shots
        backend = self._resolve_backend(backend=backend, backend_options=backend_options)
        sampler = self._resolve_sampler(sampler=sampler, backend_options=backend_options, shots=run_shots)
        sampler = self._bind_sampler_to_backend(sampler, backend)

        qreg = active[-1] + 1
        creg = len(active)
        init_specs = self._build_initial_states_on(active)
        meas_specs = self._build_measurement_pauli_on(active)
        init_keys = list(init_specs.keys())
        meas_keys = list(meas_specs.keys())

        if not batch_mode:
            if self.verbose:
                print("No acceleration: single process, single circuit per run.")
            evs = []
            jobs = []
            for meas_key in meas_keys:
                pauli = meas_specs[meas_key]
                identity_evs = self._identity_pauli_expectation(pauli)
                if identity_evs is not None:
                    evs.extend([identity_evs] * len(init_keys))
                    if self.verbose:
                        print(f"{pauli}: analytically assigned {len(init_keys)} initial states")
                    continue
                for init_key in init_keys:
                    qc = QuantumCircuit(qreg, creg)
                    self._apply_prep(qc, active, init_specs[init_key])
                    self._apply_meas_basis(qc, active, pauli)
                    qt = transpile(qc, backend=backend, optimization_level=0)
                    job = self._run_sampler_job(sampler, [qt], run_shots)
                    jobs.append(job)
                    primitive_result = job.result()
                    result = primitive_result[0]
                    counts = self._extract_counts(result)
                    evs.append(self._expectation_from_counts_z_only(counts))
                if self.verbose:
                    print(f"{pauli}: completed {len(init_keys)} initial states")
            payload = (np.asarray(evs, dtype=float), active, init_keys, meas_keys)
            if return_jobs:
                return payload + (jobs,)
            return payload

        if self.verbose:
            print("Batch mode: single process, batch circuits per measurement.")
        evs = []
        jobs = []
        for meas_key in meas_keys:
            pauli = meas_specs[meas_key]
            identity_evs = self._identity_pauli_expectation(pauli)
            if identity_evs is not None:
                evs.extend([identity_evs] * len(init_keys))
                if self.verbose:
                    print(f"{pauli}: analytically assigned {len(init_keys)} initial states")
                continue
            circuits_batch = []
            for init_key in init_keys:
                qc = QuantumCircuit(qreg, creg)
                self._apply_prep(qc, active, init_specs[init_key])
                self._apply_meas_basis(qc, active, pauli)
                qt = transpile(qc, backend=backend, optimization_level=0)
                circuits_batch.append(qt)

            job = self._run_sampler_job(sampler, circuits_batch, run_shots)
            jobs.append(job)
            result = job.result()
            evs_for_this_meas = []
            for out in result:
                counts = self._extract_counts(out)
                evs_for_this_meas.append(self._expectation_from_counts_z_only(counts))
            evs.extend(evs_for_this_meas)
            if self.verbose:
                print(f"{pauli}: completed {len(evs_for_this_meas)} initial states")

        payload = (np.asarray(evs, dtype=float), active, init_keys, meas_keys)
        if return_jobs:
            return payload + (jobs,)
        return payload

    def build_corrected_observables_from_G(self, G: np.ndarray, k: int):
        A = self._precompute_A_matrix(k)
        B = G @ inv(A)
        B_inv = inv(B)

        pauli_basis_vectors = {
            'I': np.array([[1, 0, 0, 0]], dtype=float),
            'X': np.array([[0, 1, 0, 0]], dtype=float),
            'Y': np.array([[0, 0, 1, 0]], dtype=float),
            'Z': np.array([[0, 0, 0, 1]], dtype=float),
        }

        readout_weights = {}
        for pauli_str in self._generate_pauli_strings(k):
            a_vec = pauli_basis_vectors[pauli_str[0]]
            for char in pauli_str[1:]:
                a_vec = np.kron(a_vec, pauli_basis_vectors[char])
            readout_weights[pauli_str] = a_vec @ B_inv
        return A, B, readout_weights

    def calibrate(
        self,
        base_circuit: QuantumCircuit,
        sampler=None,
        backend=None,
        shots=None,
        batch_mode: bool = True,
        backend_options=None,
        return_jobs: bool = False,
    ):
        run_output = self.run_sampler_on_subsystem(
            base_circuit,
            sampler=sampler,
            backend=backend,
            shots=shots,
            batch_mode=batch_mode,
            backend_options=backend_options,
            return_jobs=return_jobs,
        )
        if return_jobs:
            evs, active, init_keys, meas_keys, jobs = run_output
        else:
            evs, active, init_keys, meas_keys = run_output
        k = len(active)
        dim = 4 ** k
        G = evs.reshape(dim, dim)
        A, B, readout_weights = self.build_corrected_observables_from_G(G, k)
        output = {
            "active_qubits": active,
            "G": G,
            "A": A,
            "B": B,
            "ReadoutWeight": readout_weights,
            "init_keys": init_keys,
            "meas_keys": meas_keys,
        }
        if return_jobs:
            output["jobs"] = jobs
            output["job_ids"] = [job.job_id() for job in jobs if hasattr(job, "job_id")]
        return output
