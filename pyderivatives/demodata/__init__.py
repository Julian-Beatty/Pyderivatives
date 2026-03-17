from importlib.resources import files
from pathlib import Path


def get_demo_file(filename: str) -> str:
    """
    Return the absolute path to a bundled demo data file.

    Parameters
    ----------
    filename : str
        Name of the file inside pyderivatives.demodata.

    Returns
    -------
    str
        Absolute path to the bundled file.

    Raises
    ------
    FileNotFoundError
        If the requested file is not bundled with the package.
    """
    root = files("pyderivatives.demodata")
    path = root.joinpath(filename)

    if not path.is_file():
        available = sorted(
            p.name for p in root.iterdir()
            if p.is_file() and p.name != "__init__.py"
        )
        raise FileNotFoundError(
            f"Demo file '{filename}' not found in pyderivatives.demodata. "
            f"Available files: {available}"
        )

    return str(path)