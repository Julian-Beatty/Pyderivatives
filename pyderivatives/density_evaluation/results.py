import pandas as pd

from .tests import evaluate_pit_tests
from .hit_tests import standard_hit_tests, standard_patton_hit_tests
class DensityEvaluationResults:
    def __init__(self, forecasts):
        self.forecasts = forecasts

    def to_frame(self):
        rows = []

        for f in self.forecasts:
            rows.append({
                "date": f.date,
                "model": f.model_name,
                "horizon": f.horizon,
                "realized": f.realized,
                "pit": f.pit,
                "log_score": f.log_score,
            })

        return pd.DataFrame(rows)

    def log_score_table(self):
        df = self.to_frame()
        return (
            df.groupby("model")["log_score"]
            .agg(["sum", "mean", "count"])
            .sort_values("mean", ascending=False)
        )

    def pit_series(self, model=None):
        df = self.to_frame()
        if model is not None:
            df = df[df["model"] == model]
        return df[["date", "model", "pit"]]

    def test_by_model(self):
        df = self.to_frame()
    
        out = {}
    
        for model, sub in df.groupby("model"):
            pit = sub["pit"]
    
            out[model] = {
                "density_tests": evaluate_pit_tests(pit),
                "simple_hit_tests": standard_hit_tests(pit),
                "patton_hit_tests": standard_patton_hit_tests(pit),
            }
    
        return out

    def summary_table(self):
        df = self.to_frame()

        rows = []

        for model, sub in df.groupby("model"):
            tests = evaluate_pit_tests(sub["pit"])
            pit_sum = tests["pit_summary"]
            ks = tests["ks_uniform"]
            zdiag = tests["z_diagnostics"]
            berk = tests["berkowitz_lr3"]

            rows.append({
                "model": model,
                "n": pit_sum["n"],
                "mean_pit": pit_sum["mean"],
                "std_pit": pit_sum["std"],
                "mean_log_score": sub["log_score"].mean(),
                "total_log_score": sub["log_score"].sum(),
                "ks_pvalue": ks["pvalue"],
                "jb_pvalue": zdiag["jb_pvalue"],
                "berkowitz_pvalue": berk["pvalue"],
                "z_mean": zdiag["z_mean"],
                "z_std": zdiag["z_std"],
                "z_skew": zdiag["z_skew"],
                "z_kurtosis": zdiag["z_kurtosis"],
            })

        return (
            pd.DataFrame(rows)
            .sort_values("mean_log_score", ascending=False)
            .reset_index(drop=True)
        )

    @classmethod
    def combine(cls, *results_or_forecasts):
        forecasts = []

        for obj in results_or_forecasts:
            if isinstance(obj, DensityEvaluationResults):
                forecasts.extend(obj.forecasts)
            else:
                forecasts.extend(obj)

        return cls(forecasts)