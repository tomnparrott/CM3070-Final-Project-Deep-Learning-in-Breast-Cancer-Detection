import sys
from pathlib import Path
import pandas as pd


def rate(flags):
    """Percentage of rows where the audit flag is set."""
    return (flags.mean() * 100) if len(flags) else 0.0


def build_audit_summary(predictions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_csv)

    df["outcome"] = "TN"
    df.loc[(df.y_true == 0) & (df.y_pred == 1), "outcome"] = "FP"
    df.loc[(df.y_true == 1) & (df.y_pred == 0), "outcome"] = "FN"
    df.loc[(df.y_true == 1) & (df.y_pred == 1), "outcome"] = "TP"

    rows = []
    rows.append(("overall", "", len(df), rate(df["audit_flag_any"])))

    if "density_group" in df.columns:
        for g, d in df.groupby("density_group"):
            rows.append(("density_group", g, len(d), rate(d["audit_flag_any"])))

    for o, d in df.groupby("outcome"):
        rows.append(("outcome", o, len(d), rate(d["audit_flag_any"])))

    return pd.DataFrame(rows, columns=["group_by", "group", "n", "pct_flag_any"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audit_summary.py <path_to_test_predictions_with_meta.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    summary = build_audit_summary(csv_path)

    out_path = csv_path.parent / "audit_summary.csv"
    summary.to_csv(out_path, index=False)

    print(summary.sort_values(["group_by", "pct_flag_any"], ascending=[True, False]))
    print(f"\nSaved: {out_path}")
