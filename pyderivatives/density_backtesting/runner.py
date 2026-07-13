# runner.py

from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from .base import EvaluationError
from .config import EvaluationConfig
from .forecast import ForecastDataset
from .preprocessing import MarketData


def _clean_date(x):
    return pd.Timestamp(x).tz_localize(None).normalize()

def _reference_path_map(
    dataset: ForecastDataset,
    *,
    reference_model: str,
    horizon: int,
    required_models: Sequence[str],
):
    common_dates = set(dataset.common_dates(required_models, horizon=horizon))
    common_dates = {_clean_date(d) for d in common_dates}

    df_ref = dataset.get_model(
        reference_model,
        horizon=horizon,
        require_nonempty=True,
    ).sort_values("date")

    df_ref = df_ref[df_ref["date"].map(_clean_date).isin(common_dates)]

    selected = {}
    last_end = None

    for _, row in df_ref.iterrows():
        start = _clean_date(row["date"])
        end = _clean_date(row["end_date"])

        if last_end is not None and start <= last_end:
            continue

        selected[start] = {
            "end_date": end,
            "realized_horizon_days": int(row["realized_horizon_days"]),
            "T_actual": float(row["T_actual"]),
        }

        last_end = end

    return selected
def _filter_to_shared_path(
    dataset: ForecastDataset,
    *,
    reference_model: str,
    horizon: int,
    required_models: Sequence[str],
) -> ForecastDataset:
    path = _reference_path_map(
        dataset,
        reference_model=reference_model,
        horizon=horizon,
        required_models=required_models,
    )

    selected_dates = set(path.keys())

    forecasts = [
        f for f in dataset.forecasts
        if int(f.horizon) == int(horizon)
        and _clean_date(f.date) in selected_dates
    ]

    return ForecastDataset(
        forecasts=forecasts,
        errors=[],
        config=dataset.config,
        metadata={
            **dataset.metadata,
            "evaluation_strategy_applied": "shared_path",
            "path_reference_model": reference_model,
            "path_required_models": list(required_models),
            "path_horizon": int(horizon),
            "n_shared_path_dates": int(len(selected_dates)),
            "shared_path_dates": sorted(str(d.date()) for d in selected_dates),
            "shared_path_reference": {
                str(k.date()): {
                    "end_date": str(v["end_date"].date()),
                    "realized_horizon_days": int(v["realized_horizon_days"]),
                    "T_actual": float(v["T_actual"]),
                }
                for k, v in path.items()
            },
        },
    ).with_metadata()


def run_backtest(
    *,
    market_data: MarketData,
    models: Sequence,
    config: EvaluationConfig,
    start_i: Optional[int] = None,
    end_i: Optional[int] = None,
    model_names: Optional[Sequence[str]] = None,
    progress_every: int = 25,
    verbose: bool = True,
) -> ForecastDataset:
    import copy

    config.validate()
    horizons = config.resolved_horizons()

    if config.evaluation_strategy in {"single_path", "shared_path"} and len(horizons) != 1:
        raise ValueError(
            f"{config.evaluation_strategy} evaluation requires exactly one horizon."
        )

    allowed = None if model_names is None else set(model_names)

    selected = [
        model
        for model in models
        if allowed is None or model.name in allowed
    ]

    selected_names = [m.name for m in selected]

    if config.evaluation_strategy == "shared_path":
        ref = config.path_reference_model

        if ref not in selected_names:
            raise ValueError(
                f"path_reference_model='{ref}' is not in selected models."
            )

        # Force reference model to run first so it defines the shared path.
        selected = (
            [m for m in selected if m.name == ref]
            + [m for m in selected if m.name != ref]
        )
        selected_names = [m.name for m in selected]

    # Date alignment controls the candidate forecast dates independently of
    # the overlap/evaluation strategy. Keep each model's original date index
    # so fitted models still see their full historical information set.
    model_date_sets = {
        model.name: {
            _clean_date(d)
            for d in model.evaluation_dates(market_data, config)
        }
        for model in selected
    }

    eligible_dates = None
    alignment_reference = None

    if config.date_alignment == "intersection":
        if model_date_sets:
            eligible_dates = set.intersection(*model_date_sets.values())
        else:
            eligible_dates = set()

    elif config.date_alignment == "reference":
        alignment_reference = (
            config.date_alignment_reference_model
            or config.path_reference_model
        )

        if alignment_reference not in model_date_sets:
            raise ValueError(
                f"date alignment reference model '{alignment_reference}' "
                "is not in selected models."
            )

        eligible_dates = set(model_date_sets[alignment_reference])

    forecasts = []
    errors = []

    effective_strategy = (
        "all" if config.evaluation_strategy == "shared_path"
        else config.evaluation_strategy
    )

    shared_path_reference = {}
    horizon0 = int(horizons[0])

    for model in selected:
        dates = model.evaluation_dates(market_data, config)
        dates = sorted(pd.Timestamp(d).tz_localize(None) for d in dates)

        # Enumerate before alignment so `index` remains the model's original
        # history index. This is important for reserve periods and transform
        # fitting windows.
        indexed_dates = list(enumerate(dates))

        if eligible_dates is not None:
            indexed_dates = [
                (i, d)
                for i, d in indexed_dates
                if _clean_date(d) in eligible_dates
            ]

        s = 0 if start_i is None else int(start_i)
        e = len(indexed_dates) if end_i is None else int(end_i)

        date_items = indexed_dates[s:e]
        date_items = date_items[:: int(config.eval_step)]

        if verbose:
            print(
                f"[{model.name}] starting "
                f"| dates={len(date_items)} "
                f"| horizons={horizons} "
                f"| start_i={s} | end_i={e} "
                f"| window={config.window_type} "
                f"| reserve_obs={config.reserve_obs} "
                f"| min_fit_obs={config.min_fit_obs} "
                f"| evaluation_strategy={config.evaluation_strategy} "
                f"| date_alignment={config.date_alignment}",
                flush=True,
            )

        model_forecasts_before = len(forecasts)
        model_errors_before = len(errors)

        last_end_by_horizon = {}

        for count, (i, date) in enumerate(date_items, start=1):
            if verbose and (
                count == 1
                or count % progress_every == 0
                or count == len(date_items)
            ):
                print(
                    f"[{model.name}] {count}/{len(date_items)} "
                    f"| date={date.date()} "
                    f"| stored={len(forecasts)} "
                    f"| errors={len(errors)}",
                    flush=True,
                )

            for horizon_days in horizons:
                horizon_days = int(horizon_days)
                clean_date = _clean_date(date)

                # For shared_path, non-reference models only forecast on
                # reference-model selected dates.
                override_info = None
                if (
                    config.evaluation_strategy == "shared_path"
                    and model.name != config.path_reference_model
                ):
                    override_info = shared_path_reference.get(clean_date)

                    if override_info is None:
                        continue

                if effective_strategy == "single_path":
                    last_end = last_end_by_horizon.get(horizon_days)

                    if last_end is not None and clean_date <= _clean_date(last_end):
                        continue

                try:
                    call_config = config

                    # For GARCH/KDE/etc., force realized horizon to match
                    # reference RND actual observed maturity.
                    if (
                        override_info is not None
                        and getattr(config, "shared_path_align_realized_horizon", True)
                    ):
                        call_config = copy.copy(config)
                        object.__setattr__(
                            call_config,
                            "_override_realized_horizon_days",
                            int(override_info["realized_horizon_days"]),
                        )
                        object.__setattr__(
                            call_config,
                            "_override_end_date",
                            override_info["end_date"],
                        )

                    f = model.forecast_one(
                        date=date,
                        index=i,
                        horizon_days=horizon_days,
                        market_data=market_data,
                        config=call_config,
                    )

                    forecasts.append(f)

                    if effective_strategy == "single_path":
                        end_date = f.metadata.get("end_date", None)

                        if end_date is None:
                            raise ValueError(
                                "single_path requires each forecast to store "
                                "metadata['end_date']."
                            )

                        last_end_by_horizon[horizon_days] = _clean_date(end_date)

                except Exception as exc:
                    errors.append(
                        EvaluationError(
                            date=date,
                            model_name=model.name,
                            stage="forecast",
                            message=str(exc),
                            metadata={
                                "index": i,
                                "horizon_days": horizon_days,
                                "exception_type": exc.__class__.__name__,
                            },
                        )
                    )

        if verbose:
            model_forecasts = len(forecasts) - model_forecasts_before
            model_errors = len(errors) - model_errors_before

            print(
                f"[{model.name}] finished "
                f"| forecasts={model_forecasts} "
                f"| errors={model_errors}",
                flush=True,
            )

        # Once reference model is done, build the reference non-overlapping path.
        if (
            config.evaluation_strategy == "shared_path"
            and model.name == config.path_reference_model
        ):
            ref_dataset = ForecastDataset(
                forecasts=[f for f in forecasts if f.model_name == model.name],
                errors=[],
                config=config,
            ).with_metadata()

            shared_path_reference = _reference_path_map(
                ref_dataset,
                reference_model=config.path_reference_model,
                horizon=horizon0,
                required_models=[config.path_reference_model],
            )

            if verbose:
                print(
                    f"[shared_path] reference path built "
                    f"| reference_model={config.path_reference_model} "
                    f"| n_dates={len(shared_path_reference)}",
                    flush=True,
                )

    dataset = ForecastDataset(
        forecasts=forecasts,
        errors=errors,
        config=config,
        metadata={
            "date_alignment": config.date_alignment,
            "date_alignment_reference_model": alignment_reference,
            "n_eligible_dates": (
                None if eligible_dates is None else int(len(eligible_dates))
            ),
        },
    ).with_metadata()

    if config.evaluation_strategy == "shared_path":
        dataset = _filter_to_shared_path(
            dataset,
            reference_model=config.path_reference_model,
            horizon=horizon0,
            required_models=[m.name for m in selected],
        )

    return dataset