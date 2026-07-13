# pyderivatives/density_evaluation/windows.py

def get_fit_dates(dates, i, window_type="expanding", window_size=None, reserve_obs=30):
    """
    Returns dates strictly before evaluation date i.

    Parameters
    ----------
    dates : list-like
        Ordered evaluation dates.
    i : int
        Index of current forecast date.
    window_type : {"expanding", "rolling"}
    window_size : int or None
        Number of past observations for rolling window.
    reserve_obs : int
        Minimum number of observations required.
    """

    if window_type not in {"expanding", "rolling"}:
        raise ValueError("window_type must be 'expanding' or 'rolling'.")

    if window_type == "expanding":
        fit_dates = dates[:i]

    else:
        if window_size is None:
            raise ValueError("window_size must be provided for rolling windows.")

        fit_dates = dates[max(0, i - window_size):i]

    if len(fit_dates) < reserve_obs:
        return None

    return fit_dates

