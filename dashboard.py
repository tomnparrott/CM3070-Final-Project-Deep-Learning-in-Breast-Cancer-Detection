from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from SRC.inference import (
    load_config,
    processed_dir_from_cfg,
    resolve_latest_run_dir,
    list_run_dirs,
    resolve_checkpoint_path,
    get_device,
    load_model_from_checkpoint,
    preprocess_series_dir,
    predict_proba,
    gradcam_resnet18,
    extract_zip_to_temp,
    ExtractedZip,
)

# Set the page to wide layout so there's room for side by side charts and columns
st.set_page_config(
    page_title="Final Project Dashboard",
    layout="wide",
)

# Colour palette used across all the charts - defined here so everything looks consistent and I only need to change colours in one place
COL_PRIMARY = "#2b6cb0"
COL_SECONDARY = "#e53e3e"
COL_NEUTRAL = "#718096"
COL_GREEN = "#38a169"
COL_AMBER = "#dd6b20"

# Read a JSON file safely - if something goes wrong (file missing, corrupt) just return an empty dict rather than crashing the whole dashboard
def safe_read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

# The evaluation pipeline saves results under eval_outputs/<date>/<run_name>/metrics. This follows that structure and returns the most recent metrics directory so the dashboard always shows the latest evaluation
def find_latest_eval_dir(run_dir: Path) -> Path | None:
    root = run_dir / "eval_outputs"
    if not root.exists():
        return None
    date_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    if not date_dirs:
        return None
    latest = date_dirs[-1]
    candidate = latest / run_dir.name / "metrics"
    if candidate.exists():
        return candidate
    # fallback: first run subfolder
    subs = sorted([p for p in latest.iterdir() if p.is_dir()])
    if subs:
        c = subs[-1] / "metrics"
        if c.exists():
            return c
    return None

# The evaluation script saves plots (ROC, PR curve, confusion matrix) this finds them so the dashboard can display them
def find_latest_figures_dir(run_dir: Path) -> Path | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return None
    figs = mdir.parent / "figures"
    return figs if figs.exists() else None

# Failure cases directory holds the Grad-CAM overlays for misclassified cases
def find_latest_failure_dir(run_dir: Path) -> Path | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return None
    fc = mdir.parent / "failure_cases"
    return fc if fc.exists() else None

# Each of these loaders grabs a specific CSV or JSON from the latest evaluation run. If the file doesn't exist it just returns 'None' so the dashboard can skip that section rather than crashing
@st.cache_data
def load_manifest(processed_dir: Path) -> pd.DataFrame | None:
    p = processed_dir / "manifest.csv"
    return pd.read_csv(p) if p.exists() else None

def load_case_index(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is not None:
        p = mdir / "case_index.csv"
        if p.exists():
            return pd.read_csv(p)
    p = run_dir / "case_index.csv"
    return pd.read_csv(p) if p.exists() else None

def load_test_predictions(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return None
    p = mdir / "test_predictions_with_meta.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    _add_density_group(df)
    _add_outcome(df)
    return df

def load_subgroup_metrics(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return None
    p = mdir / "subgroup_metrics.csv"
    return pd.read_csv(p) if p.exists() else None

def load_bootstrap_cis(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return None
    p = mdir / "bootstrap_cis.csv"
    return pd.read_csv(p) if p.exists() else None

def load_threshold_sweep(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return None
    p = mdir / "threshold_sweep.csv"
    return pd.read_csv(p) if p.exists() else None

def load_calibration_bins(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return None
    p = mdir / "calibration_bins.csv"
    return pd.read_csv(p) if p.exists() else None

def load_audit_summary(run_dir: Path) -> pd.DataFrame | None:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return None
    p = mdir / "audit_summary.csv"
    return pd.read_csv(p) if p.exists() else None

def load_test_metrics(run_dir: Path) -> dict:
    mdir = find_latest_eval_dir(run_dir)
    if mdir is None:
        return {}
    return safe_read_json(mdir / "test_metrics.json")

def load_run_info(run_dir: Path) -> dict:
    return safe_read_json(run_dir / "run_info.json")

# The case index from evaluation only has raw breast density values (1-4) and y_true/y_pred. These helpers add a density group (1-2 vs 3-4, since dense breasts are harder to read) and an outcome label (TP/FP/TN/FN) so the dashboard can filter and display cases better
def _add_density_group(df: pd.DataFrame) -> None:
    if "density_group" in df.columns:
        return
    if "breast_density" not in df.columns:
        df["density_group"] = "Unknown"
        return
    mapping = {"1": "1-2", "2": "1-2", "3": "3-4", "4": "3-4"}
    df["density_group"] = (
        df["breast_density"].astype(str).str.strip().map(mapping).fillna("Unknown")
    )

def _add_outcome(df: pd.DataFrame) -> None:
    if "outcome" in df.columns:
        return
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return
    df["outcome"] = "TN"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "outcome"] = "FP"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "outcome"] = "FN"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 1), "outcome"] = "TP"

def enrich_case_index(ci: pd.DataFrame, manifest: pd.DataFrame | None) -> pd.DataFrame:
    if ci is None or ci.empty:
        return ci
    if manifest is None or manifest.empty:
        _add_density_group(ci)
        _add_outcome(ci)
        return ci
    if "image_dir" not in ci.columns or "image_dir" not in manifest.columns:
        _add_density_group(ci)
        _add_outcome(ci)
        return ci

    meta_cols = [
        c for c in [
            "breast_density", "abnormality_type", "laterality",
            "view", "abnormality_id", "source_csv", "patient_id", "label",
        ] if c in manifest.columns
    ]
    m = manifest[["image_dir"] + meta_cols].copy()
    df = ci.copy().merge(m, on="image_dir", how="left", suffixes=("", "_m"))
    # For loop for each metadata column, if it's missing in the case index but present in the manifest, fill it in. If it's present in both but has missing/unknown values in the case index, fill those in from the manifest
    for c in ["breast_density", "abnormality_type", "laterality", "view",
              "abnormality_id", "source_csv"]:
        mc = f"{c}_m"
        if mc in df.columns:
            if c not in df.columns:
                df[c] = df[mc]
            else:
                bad = df[c].isna() | df[c].astype(str).str.lower().isin({"unknown", "nan", ""})
                df.loc[bad, c] = df.loc[bad, mc]

    df.drop(columns=[c for c in df.columns if c.endswith("_m")], inplace=True)
    _add_density_group(df)
    _add_outcome(df)
    return df

def _norm_paths(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\\\\", "/", regex=True)
        .str.replace("\\", "/", regex=False)
        .str.strip().str.rstrip("/")
    )

def merge_audit_fields(ci: pd.DataFrame, preds: pd.DataFrame | None) -> pd.DataFrame:
    if ci is None or ci.empty or preds is None or preds.empty:
        return ci
    if "image_dir" not in ci.columns or "image_dir" not in preds.columns:
        return ci

    wanted = [
        "audit_flag_any", "audit_cam_outside_ratio", "audit_cam_edge_ratio",
        "audit_cam_ring_ratio", "audit_p_masked", "audit_delta_masked",
        "threshold_density_policy", "y_pred_density_policy",
    ]
    cols = [c for c in wanted if c in preds.columns]
    if not cols:
        return ci

    left = ci.copy()
    right = preds.copy()
    # if the case index already has some audit columns from a previous merge, drop them to avoid pandas adding _x/_y suffixes
    left.drop(columns=[c for c in left.columns if c in cols], errors="ignore", inplace=True)

    left["_k"] = _norm_paths(left["image_dir"])
    right["_k"] = _norm_paths(right["image_dir"])
    right = right.drop_duplicates(subset=["_k"], keep="first")

    out = left.merge(right[["_k"] + cols], on="_k", how="left").drop(columns=["_k"])
    for c in wanted:
        if c not in out.columns:
            out[c] = np.nan
    return out

# Matplotlib helpers for displaying images and Grad-CAM overlays. The overlay blends the raw mammogram with the Grad-CAM heatmap so you can see where the model is focusing
def overlay_fig(img: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    ax.imshow(cam, alpha=alpha, cmap="jet")
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    return fig

def heatmap_fig(cam: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cam, cmap="jet")
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    return fig

# Side by side bar chart comparing two metrics across groups, with an optional reference line showing the overall value. Used in the subgroup tab to compare sensitivity and specificity across density groups and abnormality types
def grouped_bar_chart(
    labels: list[str],
    vals_a: list[float],
    vals_b: list[float],
    name_a: str,
    name_b: str,
    title: str,
    ylabel: str,
    ref_line: float | None = None,
) -> plt.Figure:
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w / 2, vals_a, w, label=name_a, color=COL_PRIMARY)
    ax.bar(x + w / 2, vals_b, w, label=name_b, color=COL_SECONDARY)
    if ref_line is not None:
        ax.axhline(ref_line, ls="--", color=COL_NEUTRAL, lw=1, label=f"Overall ({ref_line:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig

# The decision panel shows the model's recommendation for a single case - flag for review or no flag. Also shows the risk score, threshold used, and ground truth if available. The expander underneath explains what the numbers mean for anyone who isn't familiar with the setup
def triage_label(pred: int) -> str:
    return "Flag for radiologist review" if pred == 1 else "No flag"

def resolve_eval_threshold(ci: pd.DataFrame | None, tm: dict) -> float | None:
    if ci is not None and not ci.empty and "threshold_used" in ci.columns:
        try:
            return float(ci["threshold_used"].iloc[0])
        except Exception:
            pass
    for k in ["threshold", "threshold_at_spec_0.90"]:
        if k in tm:
            try:
                return float(tm[k])
            except Exception:
                continue
    return None

def decision_panel(prob: float, pred: int, thr: float, mode: str, y_true: int | None) -> None:
    rec = triage_label(pred)
    if pred == 1:
        st.warning(f"### Recommendation: {rec}")
    else:
        st.success(f"### Recommendation: {rec}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Risk score (0-1)", f"{prob:.3f}")
    c2.metric("Threshold", f"{thr:.3f}")
    c3.metric("Mode", mode)

    if y_true is None:
        st.info("No ground truth for this case")
    else:
        label_text = "malignant" if y_true == 1 else "benign"
        st.write(f"**Ground truth:** {y_true} ({label_text})")

    with st.expander("Info", expanded=False):
        st.markdown("The risk score is the model's estimate of how likely the case is malignant. If the risk score is at or above the threshold, the case gets flagged for review.")

def cleanup_zip() -> None:
    old: ExtractedZip | None = st.session_state.get("_zip_obj")
    if old is not None:
        try:
            old.cleanup()
        except Exception:
            pass
    st.session_state["_zip_obj"] = None
    st.session_state["_zip_name"] = None

# Compute sensitivity, specificity, and counts per subgroup directly from the case index. This is used as a fallback if the precomputed subgroup_metrics.csv isn't available, though the precomputed version os prefered because it includes AUC which needs the raw probabilities not just binary predictions
def compute_subgroup_live(ci: pd.DataFrame, col: str) -> pd.DataFrame:
    if ci is None or ci.empty or col not in ci.columns:
        return pd.DataFrame()
    thr = float(ci["threshold_used"].iloc[0]) if "threshold_used" in ci.columns else 0.5
    rows = []
    for val, g in ci.groupby(col):
        yt = g["y_true"].astype(int).values
        yp = g["y_pred"].astype(int).values
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        rows.append({
            "group": str(val),
            "n": len(g),
            "pos": int((yt == 1).sum()),
            "neg": int((yt == 0).sum()),
            "sensitivity": tp / (tp + fn + 1e-8),
            "specificity": tn / (tn + fp + 1e-8),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)

# Filtering and sorting for the case review and batch triage tabs. The audit filters let you narrow down to cases where the Grad-CAM highlighted suspicious regions
def apply_audit_filters(
    df: pd.DataFrame,
    flagged_only: bool,
    min_outside: float | None,
    use_delta: bool,
    max_delta: float | None,
) -> pd.DataFrame:
    out = df
    if flagged_only and "audit_flag_any" in out.columns:
        out = out[out["audit_flag_any"].fillna(0).astype(int) == 1]
    if min_outside and min_outside > 0 and "audit_cam_outside_ratio" in out.columns:
        out = out[out["audit_cam_outside_ratio"].fillna(0).astype(float) >= min_outside]
    if use_delta and max_delta is not None and "audit_delta_masked" in out.columns:
        out = out[out["audit_delta_masked"].astype(float) <= max_delta]
    return out

def sort_cases(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "Largest drop when masked" and "audit_delta_masked" in df.columns:
        return df.sort_values("audit_delta_masked", ascending=True)
    if mode == "Highest CAM outside breast" and "audit_cam_outside_ratio" in df.columns:
        return df.sort_values("audit_cam_outside_ratio", ascending=False)
    return df.sort_values("y_prob", ascending=False)

# Sidebar controls - lets you pick which training run and checkpoint to use, toggle GPU, adjust the manual threshold, and control Grad-CAM display settings. All the inference happens with whatever is selected here
cfg = load_config()
proc_dir = processed_dir_from_cfg(cfg)
latest_run = resolve_latest_run_dir(proc_dir)
all_runs = list_run_dirs(proc_dir)

with st.sidebar:
    st.header("Run selection")
    if all_runs:
        names = [p.name for p in all_runs]
        default = names.index(latest_run.name) if latest_run.name in names else 0
        chosen = st.selectbox("Training run", names, index=default, key="sb_run")
        run_dir = proc_dir / "runs" / chosen
    else:
        run_dir = latest_run
        st.warning("No runs found")

    ckpt_choice = st.selectbox("Checkpoint", ["best", "final"], index=0, key="sb_ckpt")
    ckpt_path = resolve_checkpoint_path(run_dir, ckpt_choice)

    st.divider()
    st.header("Inference settings")
    prefer_cuda = st.toggle("Use GPU if available", value=True, key="sb_cuda")
    device = get_device(prefer_cuda=prefer_cuda)
    st.caption(f"Device: {device}")

    threshold_slider = st.slider("Manual threshold", 0.0, 1.0, 0.5, 0.01, key="sb_thr")
    tta = st.toggle("Test-time augmentation (flip)", value=False, key="sb_tta")
    show_gradcam = st.toggle("Show Grad-CAM", value=True, key="sb_cam")
    cam_alpha = st.slider("Heatmap opacity", 0.05, 0.80, 0.35, 0.05, key="sb_alpha")
    viz_mode = st.selectbox(
        "Visualisation",
        ["Input only", "Heatmap only", "Overlay"],
        index=2,
        key="sb_viz",
    )

# Load the model from the selected checkpoint. st.cache_resource means it stays in memory across reruns so the dashboard doesn't have to reload the weights every time theres a change in filter or slider
@st.cache_resource
def get_model(ckpt_str: str, dev: str):
    return load_model_from_checkpoint(Path(ckpt_str), device=dev)

model = get_model(str(ckpt_path), device)
img_size = int(cfg["data"]["img_size"])

# Load all the evaluation artefacts - metrics, predictions, subgroup breakdowns, bootstrap CIs, threshold sweep, calibration bins, and audit summary. Each one is optional so if any file is missing that section of the dashboard just won't render
manifest_df = load_manifest(proc_dir)
case_index_raw = load_case_index(run_dir)
case_index = enrich_case_index(case_index_raw, manifest_df) if case_index_raw is not None else None

preds_meta = load_test_predictions(run_dir)
if preds_meta is not None and not preds_meta.empty:
    case_index = merge_audit_fields(case_index, preds_meta)

subgroup_file = load_subgroup_metrics(run_dir)
bootstrap_df = load_bootstrap_cis(run_dir)
sweep_df = load_threshold_sweep(run_dir)
cal_bins_df = load_calibration_bins(run_dir)
audit_summary_df = load_audit_summary(run_dir)
test_metrics = load_test_metrics(run_dir)
run_info = load_run_info(run_dir)

eval_threshold = resolve_eval_threshold(case_index, test_metrics)

with st.sidebar:
    st.divider()
    st.header("Operating point")
    use_eval_thr = st.toggle(
        "Use evaluation threshold",
        value=True,
        disabled=(eval_threshold is None),
        key="sb_eval_thr",
    )
    if eval_threshold is not None:
        st.caption(f"Eval threshold: {eval_threshold:.3f}")
    else:
        st.caption("Eval threshold not available - using manual slider.")

    use_density_policy = st.toggle(
        "Density-aware threshold policy",
        value=False,
        key="sb_density_pol",
    )

# Dashboard header and tab layout
st.title("Final Project - Breast Cancer Screening Results")
tab_summary, tab_review, tab_triage, tab_subgroup, tab_threshold, tab_info = st.tabs([
    "Summary",
    "Case Review",
    "Batch Triage",
    "Subgroup Analysis",
    "Threshold Analysis",
    "Model Info",
])

# Summary tab - shows the headline metrics from the test set evaluation (AUC, sensitivity, specificity, calibration scores), the confusion matrix, bootstrap CIs if available, and the saved evaluation figures
with tab_summary:
    st.subheader("Performance summary")

    if not test_metrics:
        st.warning("No evaluation metrics found for this run.")
    else:
        auc = test_metrics.get("auc", float("nan"))
        ap = test_metrics.get("avg_precision", float("nan"))
        sens = test_metrics.get("recall_sensitivity", float("nan"))
        spec = test_metrics.get("specificity", float("nan"))
        thr = test_metrics.get("threshold", float("nan"))
        brier = test_metrics.get("brier_score", float("nan"))
        ece = test_metrics.get("ece", float("nan"))
        tp = test_metrics.get("tp", 0)
        fp = test_metrics.get("fp", 0)
        tn = test_metrics.get("tn", 0)
        fn = test_metrics.get("fn", 0)
        total = tp + fp + tn + fn

        st.markdown(f"At the chosen operating point (threshold {thr:.3f}, targeting specificity >= 0.90), the model correctly identifies about **{sens:.0%}** of malignant cases while keeping the false alarm rate at **{1 - spec:.0%}**. Out of **{tp + fn}** malignant cases in the test set, **{fn}** were missed.")

        # show the main metrics in two rows of four columns
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC", f"{auc:.4f}")
        c2.metric("Average precision", f"{ap:.4f}")
        c3.metric("Sensitivity", f"{sens:.3f}")
        c4.metric("Specificity", f"{spec:.3f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Brier score", f"{brier:.3f}")
        c6.metric("ECE", f"{ece:.3f}")
        c7.metric("Test cases", f"{total}")
        c8.metric("Threshold", f"{thr:.3f}")

        # confusion matrix counts as a single line rather than a full table
        st.markdown(f"**Confusion matrix:** {tp} TP, {fp} FP, {tn} TN, {fn} FN")

        # bootstrap CIs show how much the metrics vary with different test samples. 1000 resamples with replacement
        if bootstrap_df is not None and not bootstrap_df.empty:
            overall = bootstrap_df[bootstrap_df["scope"] == "overall"].copy()
            if not overall.empty:
                st.markdown("**95% bootstrap confidence intervals** (1000 resamples):")
                ci_rows = []
                for _, r in overall.iterrows():
                    ci_rows.append({
                        "Metric": r["metric"],
                        "Estimate": f"{r['estimate']:.4f}",
                        "95% CI": f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]",
                    })
                st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

        # display the pre-saved evaluation figures in a 2 column
        figs_dir = find_latest_figures_dir(run_dir)
        if figs_dir is not None:
            st.markdown("---")
            fig_pairs = [
                ("ROC curve", "roc_curve.png"),
                ("PR curve", "pr_curve.png"),
                ("Confusion matrix", "confusion_matrix.png"),
                ("Calibration", "calibration_curve.png"),
            ]
            left, right = st.columns(2)
            for i, (caption, fname) in enumerate(fig_pairs):
                p = figs_dir / fname
                if p.exists():
                    col = left if i % 2 == 0 else right
                    with col:
                        st.image(str(p), caption=caption, use_container_width=True)

# Case review tab - pick a case from the test set or upload one and see the model's prediction, the Grad-CAM heatmap, and all the available metrics. Filter by outcome, density, abnormality type, and sort by risk score or audit flags
with tab_review:
    st.subheader("Individual case review")
    st.caption("Pick a test set case or point to an unseen mammogram to see the model's prediction and Grad-CAM explanation.")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        source = st.radio(
            "Input source",
            ["Test set (case index)", "Manual folder path", "Upload ZIP"],
            horizontal=True,
            key="cr_source",
        )

        selected_dir: Path | None = None
        meta: dict = {}
        y_true: int | None = None
        per_case_thr: float | None = None

        if source == "Test set (case index)":
            if case_index is None or case_index.empty:
                st.warning("case_index.csv not found for this run.")
            else:
                f1, f2, f3 = st.columns(3)
                with f1:
                    out_filter = st.multiselect(
                        "Outcome",
                        sorted(case_index["outcome"].unique().tolist()),
                        default=sorted(case_index["outcome"].unique().tolist()),
                        key="cr_out",
                    )
                with f2:
                    dens_filter = st.multiselect(
                        "Density group",
                        sorted(case_index["density_group"].unique().tolist()),
                        default=sorted(case_index["density_group"].unique().tolist()),
                        key="cr_dens",
                    )
                with f3:
                    abn_filter = st.multiselect(
                        "Abnormality",
                        sorted(case_index["abnormality_type"].unique().tolist()),
                        default=sorted(case_index["abnormality_type"].unique().tolist()),
                        key="cr_abn",
                    )

                filtered = case_index[
                    case_index["outcome"].isin(out_filter)
                    & case_index["density_group"].isin(dens_filter)
                    & case_index["abnormality_type"].isin(abn_filter)
                ].copy()

                # only show the audit filters if the evaluation actually generated audit data
                has_audit = "audit_flag_any" in filtered.columns and filtered["audit_flag_any"].notna().any()
                if has_audit:
                    with st.expander("Audit filters", expanded=False):
                        a1, a2 = st.columns(2)
                        with a1:
                            flagged_only = st.checkbox("Flagged only", False, key="cr_flag")
                        with a2:
                            min_out = st.slider("Min CAM outside", 0.0, 1.0, 0.0, 0.01, key="cr_min_out")
                        filtered = apply_audit_filters(filtered, flagged_only, min_out, False, None)

                sort_opts = ["Risk score (p)"]
                if "audit_delta_masked" in filtered.columns and filtered["audit_delta_masked"].notna().any():
                    sort_opts.append("Largest drop when masked")
                if "audit_cam_outside_ratio" in filtered.columns and filtered["audit_cam_outside_ratio"].notna().any():
                    sort_opts.append("Highest CAM outside breast")
                sort_mode = st.selectbox("Sort by", sort_opts, index=0, key="cr_sort")
                filtered = sort_cases(filtered, sort_mode)

                st.write(f"**{len(filtered)}** cases match filters")
                if not filtered.empty:
                    view = filtered.head(500).copy()
                    view["label"] = (
                        view["outcome"].astype(str)
                        + " | p=" + view["y_prob"].round(3).astype(str)
                        + " | " + view["abnormality_type"].astype(str)
                        + " | dens=" + view["density_group"].astype(str)
                    )
                    idx = st.selectbox(
                        "Select case",
                        view.index.tolist(),
                        format_func=lambda i: view.loc[i, "label"],
                        key="cr_case",
                    )
                    row = view.loc[idx]
                    selected_dir = Path(row["image_dir"])
                    meta = row.to_dict()
                    try:
                        y_true = int(meta["y_true"]) if pd.notna(meta.get("y_true")) else None
                    except Exception:
                        y_true = None
                    try:
                        per_case_thr = float(meta["threshold_used"]) if pd.notna(meta.get("threshold_used")) else None
                    except Exception:
                        per_case_thr = None

        elif source == "Manual folder path":
            p = st.text_input("Folder containing DICOMs for one case", key="cr_path")
            if p.strip():
                selected_dir = Path(p.strip())
                if not selected_dir.exists():
                    st.error("Path does not exist.")

        else:  # ZIP upload
            st.write("Upload a ZIP containing the DICOM files for a single case.")
            up = st.file_uploader("ZIP file", type=["zip"], key="cr_zip")
            if st.button("Clear upload", key="cr_zip_clear"):
                cleanup_zip()
                st.rerun()

            zip_obj: ExtractedZip | None = st.session_state.get("_zip_obj")
            if up is not None:
                cleanup_zip()
                zip_obj = extract_zip_to_temp(up.getvalue())
                st.session_state["_zip_obj"] = zip_obj
                st.session_state["_zip_name"] = up.name

            if zip_obj is not None:
                selected_dir = zip_obj.series_dir
                st.success(f"Using: {st.session_state.get('_zip_name', 'upload')}")
                y_true = None

    with col_right:
        if selected_dir is None:
            st.info("Select or provide a case to view")
        else:
            try:
                prep = preprocess_series_dir(selected_dir, img_size, num_channels=3, crop_foreground=True)

                if use_eval_thr:
                    thr = per_case_thr if per_case_thr is not None else eval_threshold
                    if thr is None:
                        thr = threshold_slider
                    mode_text = "Evaluation threshold"
                else:
                    thr = threshold_slider
                    mode_text = "Manual"

                prob = predict_proba(model, prep.x, device=device, tta=tta)
                pred = int(prob >= float(thr))

                # if the density aware policy is on, swap the threshold for the per-density-group one calibrated on validation. This means denser breasts can get a different threshold based on their density grouping
                if use_density_policy and meta and source == "Test set (case index)":
                    thr_pol = meta.get("threshold_density_policy")
                    if pd.notna(thr_pol):
                        thr = float(thr_pol)
                        mode_text = "Density policy"
                        pred = int(prob >= thr)
                    pred_pol = meta.get("y_pred_density_policy")
                    if pd.notna(pred_pol):
                        pred = int(pred_pol)

                decision_panel(prob, pred, float(thr), mode_text, y_true)

                # shortcut-risk audit - shows whether the Grad-CAM activation looks like it might be relying on shortcuts - like focusing outside the breast region or near image edges rather than on actual tissue)
                if meta and pd.notna(meta.get("audit_flag_any")):
                    st.markdown("##### Shortcut-risk audit")
                    m1, m2, m3 = st.columns(3)
                    try:
                        m1.metric("Flagged", int(float(meta["audit_flag_any"])))
                    except Exception:
                        m1.metric("Flagged", "n/a")
                    try:
                        m2.metric("CAM outside", f"{float(meta['audit_cam_outside_ratio']):.3f}")
                    except Exception:
                        m2.metric("CAM outside", "n/a")
                    try:
                        m3.metric("Delta (masked)", f"{float(meta['audit_delta_masked']):.3f}")
                    except Exception:
                        m3.metric("Delta (masked)", "n/a")

                # show the preprocessed mammogram and/or Grad-CAM depending on sidebar selection
                st.markdown("##### Visualisation")
                if viz_mode == "Input only" or not show_gradcam:
                    st.image(prep.img_display, caption="Preprocessed mammogram", use_container_width=True)
                else:
                    cam = gradcam_resnet18(model, prep.x, device=device)
                    if viz_mode == "Heatmap only":
                        st.pyplot(heatmap_fig(cam))
                    else:
                        st.pyplot(overlay_fig(prep.img_display, cam, alpha=cam_alpha))

            except Exception as e:
                st.error(str(e))

# Batch triage tab - shows all test cases in a filterable table sorted by risk score. Filter by outcome, density, abnormality type, view, and audit flags. Includes a CSV download button so the data can be analysed elsewhere
with tab_triage:
    st.subheader("Batch triage view")

    if case_index is None or case_index.empty:
        st.warning("case_index.csv not found.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            bt_out = st.multiselect("Outcome", sorted(case_index["outcome"].unique().tolist()),
                                    default=sorted(case_index["outcome"].unique().tolist()), key="bt_out")
        with c2:
            bt_dens = st.multiselect("Density", sorted(case_index["density_group"].unique().tolist()),
                                     default=sorted(case_index["density_group"].unique().tolist()), key="bt_dens")
        with c3:
            bt_abn = st.multiselect("Abnormality", sorted(case_index["abnormality_type"].unique().tolist()),
                                    default=sorted(case_index["abnormality_type"].unique().tolist()), key="bt_abn")
        with c4:
            bt_view = st.multiselect("View", sorted(case_index["view"].unique().tolist()),
                                     default=sorted(case_index["view"].unique().tolist()), key="bt_view")

        df = case_index[
            case_index["outcome"].isin(bt_out)
            & case_index["density_group"].isin(bt_dens)
            & case_index["abnormality_type"].isin(bt_abn)
            & case_index["view"].isin(bt_view)
        ].copy()
        
        has_audit = "audit_flag_any" in df.columns and df["audit_flag_any"].notna().any()
        if has_audit:
            with st.expander("Audit filters", expanded=False):
                a1, a2, a3 = st.columns(3)
                with a1:
                    bt_flag = st.checkbox("Flagged only", False, key="bt_flag")
                with a2:
                    bt_min = st.slider("Min CAM outside", 0.0, 1.0, 0.0, 0.01, key="bt_min")
                with a3:
                    bt_delta = st.checkbox("Delta filter", False, key="bt_delta_chk")
                bt_max_d = None
                if bt_delta:
                    bt_max_d = st.slider("Max delta", -1.0, 1.0, -0.05, 0.01, key="bt_max_d")
                df = apply_audit_filters(df, bt_flag, bt_min, bt_delta, bt_max_d)

        sort_opts = ["Risk score (p)"]
        if "audit_delta_masked" in df.columns and df["audit_delta_masked"].notna().any():
            sort_opts.append("Largest drop when masked")
        if "audit_cam_outside_ratio" in df.columns and df["audit_cam_outside_ratio"].notna().any():
            sort_opts.append("Highest CAM outside breast")
        bt_sort = st.selectbox("Sort", sort_opts, key="bt_sort")
        df = sort_cases(df, bt_sort)

        st.write(f"**{len(df)}** cases")

        display_cols = [
            "risk_rank", "patient_id", "view", "laterality",
            "density_group", "abnormality_type",
            "y_true", "y_prob", "y_pred", "outcome",
        ]
        # add the audit columns to the display, important for triage and case review
        for extra in ["audit_flag_any", "audit_cam_outside_ratio", "audit_delta_masked"]:
            if extra in df.columns:
                display_cols.append(extra)
        if use_density_policy:
            for extra in ["threshold_density_policy", "y_pred_density_policy"]:
                if extra in df.columns:
                    display_cols.append(extra)
        display_cols = [c for c in display_cols if c in df.columns]

        st.dataframe(df[display_cols], use_container_width=True, height=500)
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "triage_filtered.csv",
            "text/csv",
        )

# Subgroup analysis tab - breaks down performance by breast density and abnormality type to display model performs differently across patient groups. Uses the precomputed subgroup_metrics.csv for AUC per group, plus bootstrap CIs and the Grad-CAM audit summary
with tab_subgroup:
    st.subheader("Subgroup performance")
    st.caption("Performance broken down by breast density and abnormality type.")

    if case_index is None or case_index.empty:
        st.warning("case_index.csv not found.")
    else:
        if subgroup_file is not None and not subgroup_file.empty:
            for group_col in ["breast_density", "abnormality_type"]:
                sub = subgroup_file[subgroup_file["group_by"] == group_col].copy()
                if sub.empty:
                    continue

                st.markdown(f"#### By {group_col.replace('_', ' ')}")

                # full metrics table
                show_cols = ["group", "n", "pos", "neg", "auc", "avg_precision",
                             "sensitivity", "specificity", "ppv", "npv"]
                show_cols = [c for c in show_cols if c in sub.columns]
                display = sub[show_cols].copy()
                            # round to 4dp so the table doesn't look messy
                for c in display.select_dtypes(include="float").columns:
                    display[c] = display[c].round(4)
                st.dataframe(display, use_container_width=True, hide_index=True)

                # bar chart comparing sensitivity and specificity across the groups, with a dashed line for the overall value to display above/below average
                labels = sub["group"].astype(str).tolist()
                sens_vals = sub["sensitivity"].astype(float).tolist()
                spec_vals = sub["specificity"].astype(float).tolist()

                overall_sens = test_metrics.get("recall_sensitivity", None)

                fig = grouped_bar_chart(
                    labels, sens_vals, spec_vals,
                    "Sensitivity", "Specificity",
                    f"Sensitivity and specificity by {group_col.replace('_', ' ')}",
                    "Rate",
                    ref_line=overall_sens,
                )
                st.pyplot(fig)
                plt.close(fig)

                # separate AUC bar chart per group - captures discrimination ability independent of the threshold
                if "auc" in sub.columns:
                    overall_auc = test_metrics.get("auc")
                    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
                    auc_vals = sub["auc"].astype(float).tolist()
                    x = np.arange(len(labels))
                    bars = ax2.bar(x, auc_vals, color=COL_PRIMARY, width=0.5)
                    if overall_auc:
                        ax2.axhline(overall_auc, ls="--", color=COL_NEUTRAL, lw=1,
                                    label=f"Overall AUC ({overall_auc:.3f})")
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(labels, rotation=25, ha="right")
                    ax2.set_ylabel("AUC")
                    ax2.set_title(f"AUC by {group_col.replace('_', ' ')}")
                    ax2.set_ylim(0.4, 1.0)
                    ax2.legend(fontsize=8)
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)

                st.markdown("---")

        # per-subgroup bootstrap CIs - shows how uncertain the subgroup metrics are given the small sample sizes in each group
        if bootstrap_df is not None and not bootstrap_df.empty:
            sub_boot = bootstrap_df[bootstrap_df["scope"] == "subgroup"].copy()
            if not sub_boot.empty:
                st.markdown("#### Bootstrap confidence intervals by subgroup")
                for gb in sub_boot["group_by"].unique():
                    chunk = sub_boot[sub_boot["group_by"] == gb].copy()
                    st.markdown(f"**{gb.replace('_', ' ')}**")
                    pivot = chunk.pivot_table(
                        index="group", columns="metric",
                        values=["estimate", "ci_low", "ci_high"],
                    )
                    # pivot gives a multiindex which can be hard to read, so it is flattened into a simple table with group, metric, estimate, and CI columns
                    flat = pd.DataFrame()
                    for metric in chunk["metric"].unique():
                        for g in chunk["group"].unique():
                            r = chunk[(chunk["group"] == g) & (chunk["metric"] == metric)]
                            if r.empty:
                                continue
                            r = r.iloc[0]
                            flat = pd.concat([flat, pd.DataFrame([{
                                "Group": g,
                                "Metric": metric,
                                "Estimate": f"{r['estimate']:.4f}",
                                "95% CI": f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]",
                            }])], ignore_index=True)
                    if not flat.empty:
                        st.dataframe(flat, use_container_width=True, hide_index=True)

        # Grad-CAM audit summary - shows what percentage of cases in each group had suspicious patterns
        if audit_summary_df is not None and not audit_summary_df.empty:
            st.markdown("#### Grad-CAM audit summary")
            st.caption("Percentage of cases where the Grad-CAM activation fell outside the breast region, near image edges, or in a boundary ring.")
            st.dataframe(audit_summary_df, use_container_width=True, hide_index=True)

# Threshold explorer tab - drag slider across full range of decision thresholds to see how it changes sensitivity, specificity, PPV, NPV, and the confusion matrix. Also shows the calibration reliability diagram and the named operating points from the evaluation
with tab_threshold:
    st.subheader("Threshold analysis")
    st.caption("The decision threshold controls the trade-off between sensitivity and specificity. This tab shows how changing the threshold affects the confusion matrix and performance metrics.")

    if sweep_df is None or sweep_df.empty:
        st.warning("threshold_sweep.csv not found for this run.")
    else:
        # slider defaults to threshold the evaluation used
        t_select = st.slider(
            "Select threshold",
            float(sweep_df["threshold"].min()),
            float(sweep_df["threshold"].max()),
            float(test_metrics.get("threshold", 0.5)),
            0.01,
            key="te_thr",
        )
        # the sweep CSV has threshold steps to snap to the nearest one
        nearest_idx = (sweep_df["threshold"] - t_select).abs().idxmin()
        row = sweep_df.loc[nearest_idx]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sensitivity", f"{row['sensitivity']:.3f}")
        c2.metric("Specificity", f"{row['specificity']:.3f}")
        c3.metric("PPV", f"{row['ppv']:.3f}")
        c4.metric("NPV", f"{row['npv']:.3f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("TP", int(row["tp"]))
        c6.metric("FP", int(row["fp"]))
        c7.metric("TN", int(row["tn"]))
        c8.metric("FN", int(row["fn"]))

        # plot sensitivity and specificity as two lines across the threshold range
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(sweep_df["threshold"], sweep_df["sensitivity"],
                label="Sensitivity", color=COL_SECONDARY, lw=2)
        ax.plot(sweep_df["threshold"], sweep_df["specificity"],
                label="Specificity", color=COL_PRIMARY, lw=2)
        ax.axvline(t_select, ls="--", color=COL_AMBER, lw=1.5, label=f"Selected ({t_select:.2f})")

        # mark the evaluation operating point as a dotted line to see where it sits
        eval_t = test_metrics.get("threshold")
        if eval_t is not None:
            ax.axvline(float(eval_t), ls=":", color=COL_NEUTRAL, lw=1,
                       label=f"Eval OP ({float(eval_t):.3f})")

        ax.set_xlabel("Threshold")
        ax.set_ylabel("Rate")
        ax.set_title("Sensitivity and specificity vs. threshold")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Table of the named operating points that the evaluation script pre computed
        ops = test_metrics.get("operating_points", {})
        if ops:
            st.markdown("##### Named operating points")
            op_rows = []
            for name, vals in ops.items():
                op_rows.append({
                    "Operating point": name,
                    "Threshold": f"{vals.get('threshold', 0):.4f}",
                    "Sensitivity": f"{vals.get('sensitivity', 0):.4f}",
                    "Specificity": f"{vals.get('specificity', 0):.4f}",
                })
            st.dataframe(pd.DataFrame(op_rows), use_container_width=True, hide_index=True)

    # calibration reliability diagram - bins the predicted probabilities and compares them to the actual fraction of positives in each bin
    if cal_bins_df is not None and not cal_bins_df.empty:
        st.markdown("---")
        st.markdown("#### Calibration")
        st.caption(
            "How well the predicted probabilities match the actual outcomes."
        )

        fig3, ax3 = plt.subplots(figsize=(6, 5))
        ax3.plot([0, 1], [0, 1], ls="--", color=COL_NEUTRAL, label="Perfect calibration")
        ax3.bar(
            cal_bins_df["mean_prob"],
            cal_bins_df["frac_pos"],
            width=0.08,
            alpha=0.7,
            color=COL_PRIMARY,
            label="Observed fraction positive",
        )
        ax3.set_xlabel("Mean predicted probability")
        ax3.set_ylabel("Fraction positive")
        ax3.set_title("Calibration reliability diagram")
        ax3.legend(fontsize=8)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        ece_val = test_metrics.get("ece")
        mce_val = test_metrics.get("mce")
        brier_val = test_metrics.get("brier_score")
        if ece_val is not None:
            st.write(
                f"ECE: **{ece_val:.3f}**, MCE: **{mce_val:.3f}**, "
                f"Brier: **{brier_val:.3f}**"
            )

        st.dataframe(cal_bins_df, use_container_width=True, hide_index=True)

# Model info tab - pulls out the key training hyperparameters from run_info.json and displays them as readable metrics
with tab_info:
    st.subheader("Training configuration and run metadata")

    if run_info:
        ri = run_info
        config_block = ri.get("config", {})
        train_cfg = config_block.get("train", {})
        data_cfg = config_block.get("data", {})

        st.markdown("#### Training summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Architecture", "ResNet-18")
        c2.metric("Image size", f"{data_cfg.get('img_size', '?')}px")
        c3.metric("Epochs", str(train_cfg.get("epochs", "?")))
        c4.metric("Learning rate", str(train_cfg.get("lr", "?")))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Batch size", str(train_cfg.get("batch_size", "?")))
        c6.metric("pos_weight", str(train_cfg.get("pos_weight", "?")))
        c7.metric("Augmentation", str(train_cfg.get("augment", "?")))
        c8.metric("Freeze backbone", str(train_cfg.get("freeze_backbone", "?")))

        seed = config_block.get("seed", ri.get("seed", "?"))
        st.write(f"**Seed:** {seed}")

        started = ri.get("started")
        finished = ri.get("finished")
        if started:
            st.write(f"**Started:** {started}")
        if finished:
            st.write(f"**Finished:** {finished}")

        with st.expander("Full run_info.json", expanded=False):
            st.json(run_info)
    else:
        st.warning("run_info.json not found.")

    if test_metrics:
        with st.expander("Full test_metrics.json", expanded=False):
            st.json(test_metrics)

    # show which checkpoint file is loaded and how big it is
    st.markdown("---")
    st.markdown("#### Checkpoint")
    st.write(f"Using: **{ckpt_path.name}**")
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        st.write(f"Size: {size_mb:.1f} MB")
