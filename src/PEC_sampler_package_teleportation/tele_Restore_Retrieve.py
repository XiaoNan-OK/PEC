import ast
import json
import pickle
import numpy as np
from typing import Dict, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
from scipy.io import savemat

Pair = Tuple[int, int]
#----- Restore Readout Dictionary -----
def save_readout_result(result, 
                        fname:str = "OWeight",
                        parent_dir: Union[str, Path] = "ROWeight",
                        timefmt: str = "%Y%m%d-%H%M%S",):
    """
    result : dict（就是你那包 active_qubits, G, A, B, ReadoutWeight, init_keys, meas_keys）
    fname : 檔案名稱，預設 OWeight
    parent_dir : 上層資料夾名稱
    """

    save_dir = Path(parent_dir).expanduser() / datetime.now().strftime(timefmt)
    save_dir.mkdir(parents=True, exist_ok=True)
    base = save_dir / Path(fname).name
    save_path = base.with_suffix(".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved to: {save_path}")
    return save_path

#----- Restore Two-Qubit Gate Vector -----
def save_vectors(
    data: Dict[Pair, np.ndarray],
    fname: str = "GWeight",
    parent_dir: Union[str, Path] = "TQGWeight",
    timefmt: str = "%Y%m%d-%H%M%S",
) -> Dict[str, Any]:
    """
    存 (c,t) → 1D 向量 的資料結構到時間戳資料夾：
      <parent_dir>/<timestamp>/{fname}.npy
                                       .npz
                                       .mat
                                       _meta.json + 每條向量一個 CSV（{fname}_c{c}_t{t}.csv）

    參數
    ----
    data: Dict[(c,t), 1D np.ndarray]
        例如：{(1,0): vec1, (2,1): vec2, ...}；所有向量長度需一致。
    fname: 檔名前綴（不含副檔名）
    parent_dir: 放置時間資料夾的父層
    timefmt: 時間資料夾命名格式

    回傳
    ----
    manifest: Dict，包含各輸出檔案路徑
    """
    assert isinstance(data, dict) and len(data) > 0, "data 不可為空"
    # 驗證所有 value 為 1D 向量且長度一致
    keys_sorted = sorted(data.keys())
    vecs = []
    L = None
    for k in keys_sorted:
        v = np.asarray(data[k]).astype(float)
        assert v.ndim == 1, f"所有 value 必須為 1D 向量，鍵 {k} 的 ndim={v.ndim}"
        L = L or v.shape[0]
        assert v.shape[0] == L, f"向量長度不一致：預期 {L}，鍵 {k} 的長度 {v.shape[0]}"
        vecs.append(v)
    stack = np.stack(vecs, axis=0)  # (N, L)
    keys_arr = np.array(keys_sorted, dtype=int)  # (N, 2)

    # 建立輸出資料夾
    run_dir = Path(parent_dir).expanduser() / datetime.now().strftime(timefmt)
    run_dir.mkdir(parents=True, exist_ok=True)
    base = run_dir / Path(fname).name

    manifest: Dict[str, Any] = {"dir": str(run_dir)}

    # 1) .npy（Python 物件） ------------------------------------
    npy_path = base.with_suffix(".npy")
    np.save(npy_path, {"vector_dim": L, "keys": keys_arr, "vectors": stack})
    manifest["npy"] = str(npy_path)

    # 2) .npz（壓縮、跨語言友善） -------------------------------
    npz_path = base.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        vector_dim=np.array(L, dtype=np.int32),
        keys=keys_arr,
        vectors=stack,
    )
    manifest["npz"] = str(npz_path)

    # 3) CSV + meta.json ---------------------------------------
    csv_paths = []
    meta = {"vector_dim": int(L), "vectors": []}  # list of {"c": int, "t": int, "file": str}
    for (c, t), vec in zip(keys_sorted, stack):
        csv_path = run_dir / f"{base.name}_c{c}_t{t}.csv"
        # 確保以單列輸出
        np.savetxt(csv_path, vec.reshape(1, -1), delimiter=",")
        csv_paths.append(str(csv_path))
        meta["vectors"].append({"c": int(c), "t": int(t), "file": csv_path.name})
    meta_path = run_dir / f"{base.name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    manifest["csv_list"] = csv_paths
    manifest["meta_json"] = str(meta_path)

    # 4) MATLAB .mat -------------------------------------------
    mat_path = base.with_suffix(".mat")
    savemat(
        mat_path,
        {"vector_dim": np.array([[L]], dtype=np.int32), "keys": keys_arr, "vectors": stack},
        do_compression=True,
    )
    manifest["mat"] = str(mat_path)

    return manifest

#----- Restore Two-Qubit Gate Matrix -----
def save_matrix(
    matrix: Dict[str, Any],
    fname: str = "grammatrix",  
    parent_dir: Union[str, Path] = ".",          # 放置時間資料夾的父層
    timefmt: str = "%Y%m%d-%H%M%S",              # 資料夾名稱的時間格式
) -> Dict[str, Any]:
    """
    將所有檔案存入「以時間為名稱」的資料夾中，內部檔名統一使用 'grammatrix' 前綴：
      <parent_dir>/<timestamp>/grammatrix.npy
                                  /grammatrix.npz
                                  /grammatrix.mat
                                  /grammatrix_meta.json
                                  /grammatrix_c{c}_t{t}.csv  (每個矩陣一檔)

    回傳 manifest（包含資料夾路徑與各檔案路徑）。
    """
    # --- build：<parent_dir>/<timestamp> ---
    run_dir = Path(parent_dir).expanduser() / datetime.now().strftime(timefmt)
    run_dir.mkdir(parents=True, exist_ok=True)
    fname = Path(fname).name
    base_path = run_dir / fname

    # --- 驗證與整理資料 ---
    rows = list(matrix["row_labels"])
    cols = list(matrix["col_labels"])
    assert len(rows) == len(cols), f"row/col length mismatch: {len(rows)}/{len(cols)}"
    D = len(rows)
    num_qubits = int(matrix.get("num_qubits", round(np.log2(D))))
    matrices_dict: Dict[Tuple[int, int], np.ndarray] = matrix["matrices"]
    assert len(matrices_dict) > 0, "matrix['matrices'] no value"

    keys_sorted = sorted(matrices_dict.keys())
    mats = [np.asarray(matrices_dict[k]) for k in keys_sorted]
    stack = np.stack(mats, axis=0)  # (N, D, D)
    assert stack.shape[1:] == (D, D), f"matrix shape mismatch: {stack.shape}"
    keys_arr = np.array(keys_sorted, dtype=int)  # (N, 2)

    manifest = {"dir": str(run_dir)}

    # --- 1) .npy（Python 物件，載入最快） ---
    npy_path = base_path.with_suffix(".npy")
    np.save(npy_path, {
        "num_qubits": num_qubits,
        "row_labels": rows,
        "col_labels": cols,
        "matrices_keys": keys_arr,
        "matrices": stack
    })
    manifest["npy"] = str(npy_path)

    # --- 2) .npz（壓縮、跨語言友善） ---
    npz_path = base_path.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        num_qubits=np.array(num_qubits, dtype=np.int32),
        row_labels=np.array(rows, dtype=object),
        col_labels=np.array(cols, dtype=object),
        matrices_keys=keys_arr,
        matrices=stack
    )
    manifest["npz"] = str(npz_path)

    # --- 3) CSV + meta.json ---
    meta = {
        "num_qubits": num_qubits,
        "row_labels": rows,
        "col_labels": cols,
        "matrices": []  # list of {"c": int, "t": int, "file": str}
    }
    csv_paths = []
    for (c, t), mat in zip(keys_sorted, stack):
        csv_path = run_dir / f"{base_path.name}_c{c}_t{t}.csv"
        np.savetxt(csv_path, mat, delimiter=",")
        csv_paths.append(str(csv_path))
        meta["matrices"].append({"c": int(c), "t": int(t), "file": csv_path.name})
    meta_path = run_dir / f"{base_path.name}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    manifest["csv_list"] = csv_paths
    manifest["meta_json"] = str(meta_path)

    # --- 4) MATLAB .mat ---
    mat_path = base_path.with_suffix(".mat")
    savemat(
        mat_path,
        {
            "num_qubits": np.array([[num_qubits]], dtype=np.int32),
            "row_labels": np.array(rows, dtype=object),
            "col_labels": np.array(cols, dtype=object),
            "matrices_keys": keys_arr,
            "matrices": stack
        },
        do_compression=True
    )
    manifest["mat"] = str(mat_path)

    return manifest

#----- Restore Experiment result -----
def save_experiment_results(result, 
                           fname:str = "results",
                           parent_dir: Union[str, Path] = "PEC_results",
                           timefmt: str = "%Y%m%d-%H%M%S",):
    """
    result : dictionary
    fname : 檔案名稱
    parent_dir : 上層資料夾名稱
    """

    save_dir = Path(parent_dir).expanduser() / datetime.now().strftime(timefmt)
    save_dir.mkdir(parents=True, exist_ok=True)
    base = save_dir / Path(fname).name
    save_path = base.with_suffix(".pkl")
    with open(save_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved to: {save_path}")
    return save_path

#----- Restore Experiment graph -----
def save_chsh_figures(fig_trial, fig_dist, n_trials: int,
                      parent_dir="output_graph",
                      timefmt="%Y%m%d-%H%M%S"):
    """
    會在 parent_dir 底下建立：
        <timestamp>_<n_trials>/
            CHSH_trial.png
            CHSH_distribution.png
    例如：
        output_graph/20251117-163210_50/CHSH_trial.png
    """
    # 1. 先確保 output_graph 存在
    base = Path(parent_dir)
    base.mkdir(parents=True, exist_ok=True)

    # 2. 建 timestamp_nTrials 資料夾
    timestamp = datetime.now().strftime(timefmt)
    save_dir = base / f"{timestamp}_{n_trials}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 3. 路徑
    path_trial = save_dir / "CHSH_trial.png"
    path_dist  = save_dir / "CHSH_distribution.png"

    # 4. 存檔
    fig_trial.savefig(path_trial, dpi=300, bbox_inches="tight")
    fig_dist.savefig(path_dist, dpi=300, bbox_inches="tight")

    print(f"[SAVE] CHSH_trial → {path_trial}")
    print(f"[SAVE] CHSH_distribution → {path_dist}")

    return save_dir

#----- Retrieve Readout PEC Weight -----
def load_readout_result(path):
    with open(path, "rb") as f:
        return pickle.load(f)

#----- Retrieve Vector -----
try:
    from scipy.io import loadmat as _loadmat
except ImportError:
    loadmat = None  # 若沒裝 scipy，也能載回 npy/npz/CSV
def _coerce_cnot_key(k) -> Tuple[int, int]:
    """把各種格式的 key 轉成 (int,int)。"""
    if isinstance(k, (tuple, list)) and len(k) == 2:
        return (int(k[0]), int(k[1]))
    if isinstance(k, str):
        try:
            t = ast.literal_eval(k)
            if isinstance(t, (tuple, list)) and len(t) == 2:
                return (int(t[0]), int(t[1]))
        except Exception:
            pass
        parts = [p.strip() for p in k.strip("()[] ").split(",")]
        if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
            return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Unrecognized CNOT key format: {k!r}")

def _pack_vectors(keys_arr: np.ndarray, stack: np.ndarray) -> Dict[str, Any]:
    keys = [tuple(map(int, k)) for k in np.asarray(keys_arr).reshape(-1, 2).tolist()]
    assert stack.ndim == 2, f"期望 (N, L)，拿到 {stack.shape}"
    L = int(stack.shape[1])
    assert len(keys) == stack.shape[0], "keys 與向量數量不符"
    vectors: Dict[Pair, np.ndarray] = {k: stack[i].astype(float) for i, k in enumerate(keys)}
    return {
        "vector_dim": L,
        "keys": keys,          # 供需要穩定排序時使用
        "vectors": vectors,    # Dict[(c,t)] → 1D np.ndarray
        "stack": stack,        # (N, L) 快速批次運算
    }

def load_vectors_from_npy(path: Union[str, Path]) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True).item()
    if isinstance(obj, dict) and "keys" not in obj:
        d = { _coerce_cnot_key(k): np.asarray(v, float).ravel() for k, v in obj.items() }
        keys = np.array(list(d.keys()), dtype=int)
        vecs = np.stack([d[k] for k in d.keys()], axis=0)
        return _pack_vectors(keys, vecs)
    # 原本的 keys + vectors 格式
    return _pack_vectors(obj["keys"], np.asarray(obj["vectors"]))

def load_vectors_from_npz(path: Union[str, Path]) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        if "arr_0" in z and isinstance(z["arr_0"].item(), dict):
            d = z["arr_0"].item()
            d = { _coerce_cnot_key(k): np.asarray(v, float).ravel() for k, v in d.items() }
            keys = np.array(list(d.keys()), dtype=int)
            vecs = np.stack([d[k] for k in d.keys()], axis=0)
            return _pack_vectors(keys, vecs)
        return _pack_vectors(z["keys"], z["vectors"])

def load_vectors_from_mat(path: Union[str, Path]) -> Dict[str, Any]:
    if _loadmat is None:
        raise RuntimeError("需要 scipy 才能讀 .mat（pip install scipy）")
    md = _loadmat(path)
    for k in list(md.keys()):
        if k.startswith("__") and k.endswith("__"):
            md.pop(k, None)
    # MATLAB 讀回來的整數可能是 2D，需要擠壓與轉型
    keys_arr = np.asarray(md["keys"]).astype(int)
    stack = np.asarray(md["vectors"]).astype(float)
    return _pack_vectors(keys_arr, stack)

def load_vectors_from_csv_meta(prefix: Union[str, Path]) -> Dict[str, Any]:
    """
    讀取 {prefix}_meta.json + 各 CSV（每檔一條向量，單列 CSV）。
    prefix 可為前綴（不含副檔名）或直接給 meta.json 路徑。
    """
    prefix = Path(prefix)
    meta_path = prefix if prefix.suffix == ".json" else prefix.with_name(prefix.name + "_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_qubits = int(meta["num_qubits"])
    rows = meta["row_labels"]; cols = meta["col_labels"]

    base_dir = meta_path.parent
    items = meta["matrices"]  # list of {"c": int, "t": int, "file": str}
    items = sorted(items, key=lambda d: (int(d["c"]), int(d["t"])))

    mats, keys = [], []
    for rec in items:
        csv_file = Path(rec["file"])
        if not csv_file.is_absolute():
            csv_file = base_dir / csv_file
        mats.append(np.loadtxt(csv_file, delimiter=","))
        keys.append((int(rec["c"]), int(rec["t"])))

    stack = np.stack(mats, axis=0)
    keys_arr = np.array(keys, dtype=int)
    return _pack_matrix(num_qubits, rows, cols, keys_arr, stack)

def load_vectors(path_or_prefix: Union[str, Path]) -> Dict[str, Any]:
    """
    智慧載入：
      - 若給完整檔名（含副檔名），依副檔名分派。
      - 若給前綴，會依優先序嘗試：.npy -> .npz -> .mat -> _meta.json(+CSV)。
    """
    p = Path(path_or_prefix).expanduser()
    if p.suffix:
        ext = p.suffix.lower()
        if ext == ".npy":
            return load_vectors_from_npy(p)
        if ext == ".npz":
            return load_vectors_from_npz(p)
        if ext == ".mat":
            return load_vectors_from_mat(p)
        if ext == ".json":
            return load_vectors_from_csv_meta(p)
        raise ValueError(f"不支援的副檔名：{ext}")

    candidates = [
        p.with_suffix(".npy"),
        p.with_suffix(".npz"),
        p.with_suffix(".mat"),
        p.with_name(p.name + "_meta.json"),
    ]
    for cand in candidates:
        if cand.exists():
            if cand.suffix == ".npy":
                return load_vectors_from_npy(cand)
            if cand.suffix == ".npz":
                return load_vectors_from_npz(cand)
            if cand.suffix == ".mat":
                return load_vectors_from_mat(cand)
            if cand.suffix == ".json":
                return load_vectors_from_csv_meta(cand)
    raise FileNotFoundError(f"No usable file：{[str(c) for c in candidates]}")

#----- Retrieve Matrix -----
try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None  # 若沒裝 scipy，也能載回 npy/npz/CSV

def _labels_to_list(arr) -> list:
    """把各種 ndarray/object 形式的標籤轉乾淨的 Python list[str]."""
    if isinstance(arr, np.ndarray):
        arr = arr.squeeze()
        # 若是 MATLAB char array（2D字元陣列）或 object 陣列
        if arr.dtype.kind in ("U", "S", "O"):
            lst = arr.tolist()
            if isinstance(lst, (list, tuple)):
                return [str(x) for x in lst]
            return [str(lst)]
        # 其他型別直接轉字串
        return [str(x) for x in arr.tolist()]
    if isinstance(arr, (list, tuple)):
        return [str(x) for x in arr]
    return [str(arr)]

def _pack_matrix(num_qubits: int, row_labels, col_labels, keys_arr, stack) -> Dict[str, Any]:
    rows = _labels_to_list(row_labels)
    cols = _labels_to_list(col_labels)
    keys = [tuple(map(int, k)) for k in np.asarray(keys_arr).reshape(-1, 2).tolist()]
    mats = np.asarray(stack)
    assert mats.ndim == 3, f"期望 (N,D,D)，拿到 {mats.shape}"
    D = len(rows)
    assert D == len(cols) == mats.shape[1] == mats.shape[2], "row/col 長度或矩陣尺寸不符"
    assert len(keys) == mats.shape[0], "keys 與矩陣數量不符"
    matrices: Dict[Tuple[int,int], np.ndarray] = {k: mats[i] for i, k in enumerate(keys)}
    return {
        "num_qubits": int(num_qubits),
        "row_labels": rows,
        "col_labels": cols,
        "matrices": matrices,
    }

# ---------- load individually ----------
def load_matrix_from_npy(path: Union[str, Path]) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True).item()
    return _pack_matrix(
        obj["num_qubits"],
        obj["row_labels"],
        obj["col_labels"],
        obj["matrices_keys"],
        obj["matrices"],
    )

def load_matrix_from_npz(path: Union[str, Path]) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return _pack_matrix(
            int(z["num_qubits"]),
            z["row_labels"],
            z["col_labels"],
            z["matrices_keys"],
            z["matrices"],
        )

def load_matrix_from_mat(path: Union[str, Path]) -> Dict[str, Any]:
    if loadmat is None:
        raise RuntimeError("需要 scipy 才能讀 .mat（pip install scipy）")
    md = loadmat(path)
    # 去掉 MATLAB 預設的欄位
    for k in list(md.keys()):
        if k.startswith("__") and k.endswith("__"):
            md.pop(k, None)
    return _pack_matrix(
        int(np.array(md["num_qubits"]).squeeze()),
        md["row_labels"],
        md["col_labels"],
        md["matrices_keys"],
        md["matrices"],
    )

def load_matrix_from_csv_meta(prefix: Union[str, Path]) -> Dict[str, Any]:
    """讀取 {prefix}_meta.json + 各個 CSV。"""
    prefix = Path(prefix)
    meta_path = prefix if prefix.suffix == ".json" else prefix.with_name(prefix.name + "_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_qubits = int(meta["num_qubits"])
    rows = meta["row_labels"]
    cols = meta["col_labels"]

    # 收集 keys 與矩陣；CSV 路徑用 meta.json 所在資料夾為基準
    base_dir = meta_path.parent
    items = meta["matrices"]  # list of {"i": int, "j": int, "file": str}
    # 保持穩定排序
    items = sorted(items, key=lambda d: (int(d["i"]), int(d["j"])))
    mats = []
    keys = []
    for rec in items:
        csv_file = Path(rec["file"])
        if not csv_file.is_absolute():
            csv_file = base_dir / csv_file
        mats.append(np.loadtxt(csv_file, delimiter=","))
        keys.append((int(rec["i"]), int(rec["j"])))

    stack = np.stack(mats, axis=0)
    keys_arr = np.array(keys, dtype=int)
    return _pack_matrix(num_qubits, rows, cols, keys_arr, stack)

# ---------- Auto determine ----------
def load_matrix(path_or_prefix: Union[str, Path]) -> Dict[str, Any]:
    """
    智慧載入：
      - 若給完整檔名（含副檔名），依副檔名分派。
      - 若給前綴，會依優先序嘗試：.npy -> .npz -> .mat -> _meta.json(+CSV)。
    """
    p = Path(path_or_prefix).expanduser()
    if p.suffix:
        ext = p.suffix.lower()
        if ext == ".npy":
            return load_matrix_from_npy(p)
        if ext == ".npz":
            return load_matrix_from_npz(p)
        if ext == ".mat":
            return load_matrix_from_mat(p)
        if ext == ".json":
            return load_matrix_from_csv_meta(p)
        raise ValueError(f"不支援的副檔名：{ext}")

    # 沒副檔名：依序嘗試
    candidates = [
        p.with_suffix(".npy"),
        p.with_suffix(".npz"),
        p.with_suffix(".mat"),
        p.with_name(p.name + "_meta.json"),
    ]
    for cand in candidates:
        if cand.exists():
            if cand.suffix == ".npy":
                return load_matrix_from_npy(cand)
            if cand.suffix == ".npz":
                return load_matrix_from_npz(cand)
            if cand.suffix == ".mat":
                return load_matrix_from_mat(cand)
            if cand.suffix == ".json":
                return load_matrix_from_csv_meta(cand)
    raise FileNotFoundError(f"No usable file：{[str(c) for c in candidates]}")