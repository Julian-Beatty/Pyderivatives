import glob
import pickle
from pathlib import Path


def merge_information_dict_parts(
    pattern: str,
    *,
    output_file: str,
    sort_parts: bool = True,
    verbose: bool = True,
):
    """
    Merge split information-dict pickle parts into a single pickle.

    Parameters
    ----------
    pattern : str
        Glob pattern for part files, e.g.
        "XLE_information_dict_part*_of_050.pkl"
    output_file : str
        Filename for merged pickle, e.g.
        "XLE_information_dict_FULL.pkl"
    sort_parts : bool, default True
        Whether to sort matched part files before merging.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    dict
        The merged dictionary with keys:
        - "stock_df"
        - "result_dict"
    """

    parts = glob.glob(pattern)
    if not parts:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    if sort_parts:
        parts = sorted(parts)

    if verbose:
        print(f"[merge] Found {len(parts)} parts")

    merged = {"stock_df": None, "result_dict": {}}

    for i, p in enumerate(parts, 1):
        if verbose:
            print(f"[merge] ({i}/{len(parts)}) Loading {Path(p).name}")

        with open(p, "rb") as f:
            pkg = pickle.load(f)

        # sanity checks
        if "stock_df" not in pkg or "result_dict" not in pkg:
            raise KeyError(f"{p} missing required keys")

        if merged["stock_df"] is None:
            merged["stock_df"] = pkg["stock_df"]

        merged["result_dict"].update(pkg["result_dict"])

    with open(output_file, "wb") as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"[merge] Saved merged file -> {output_file}")
        print(f"[merge] Total dates = {len(merged['result_dict'])}")

    return merged


import pickle
from pathlib import Path

def merge_kernel_parts(pattern="CVX_kernel_dict_part*_of_*.pkl", out_name="CVX_kernel_dict_FULL.pkl"):
    parts = sorted(Path(".").glob(pattern))
    if not parts:
        raise FileNotFoundError(f"No files match {pattern}")

    merged = {}
    meta = None
    for p in parts:
        with open(p, "rb") as f:
            pkg = pickle.load(f)
        if meta is None:
            meta = {k: pkg.get(k) for k in ["asset", "min_date", "max_date", "N"]}
        kd = pkg.get("kernel_dict", {})
        overlap = set(merged).intersection(kd)
        if overlap:
            raise ValueError(f"Overlapping dates found in {p}: {sorted(list(overlap))[:5]}")
        merged.update(kd)

    out_pkg = {"kernel_dict": merged, "meta": meta, "n_dates": len(merged)}
    with open(out_name, "wb") as f:
        pickle.dump(out_pkg, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[DONE] merged {len(parts)} parts -> {out_name} with {len(merged)} dates")
    return out_pkg

#merge_kernel_parts()



