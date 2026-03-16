from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

# =========================
# Configurazione globale
# =========================
sns.set_theme(style="darkgrid")

# =========================
# Font più grandi (globali)
# =========================
plt.rcParams.update(
    {
        "font.size": 20,  # base
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "legend.title_fontsize": 20,
    }
)

LOG_DIR = Path("/leonardo_scratch/fast/uToID_bench/xffl/examples/EuroPar/plots")
ENGINE = "pyarrow"
SEP = ";"

METHODS: List[str] = [
    "FSDP",
    "HSDP",
    "FL+FSDP",
    "FL+HSDP",
]

TIME_COL = "Relative Time (Process)"
STEP_COL = "Step"

X_LABEL = "Time (minutes)"
Y_LABEL = "Perplexity"

X_LIM = (0, 730)
Y_LIM = (1e2, 1e5)

XTICKS = [i * 60 for i in range(0, (X_LIM[1] // 60) + 1)]
XTICKLABELS = list(range(0, (X_LIM[1] // 60) + 1))


# =========================
# Funzioni
# =========================
def load_time_data(method: str) -> pd.DataFrame:
    return pd.read_csv(
        LOG_DIR / f"time_to_perp_{method}.csv",
        sep=SEP,
        engine=ENGINE,
    )


def load_step_data() -> pd.DataFrame:
    return pd.read_csv(
        LOG_DIR / "step_to_perp.csv",
        sep=SEP,
        engine=ENGINE,
    )


def preprocess(
    method: str, time_df: pd.DataFrame, step_df: pd.DataFrame
) -> pd.DataFrame:
    perp = f"Group: {method}_new - train/Step_perplexity"
    perp_step = f"Group: {method}_new - _step"
    perp_min = f"Group: {method}_new - train/Step_perplexity__MIN"
    perp_max = f"Group: {method}_new - train/Step_perplexity__MAX"

    # Tempo medio per step
    time_processed = (
        time_df[[TIME_COL, perp_step]]
        .dropna()
        .rename(columns={perp_step: STEP_COL})
        .astype({STEP_COL: int})
        .groupby(STEP_COL, as_index=False)
        .mean()
    )

    # Merge con metriche
    step_metrics = step_df[[STEP_COL, perp, perp_min, perp_max]]
    merged = pd.merge(time_processed, step_metrics, on=STEP_COL)

    # Normalizza tempo (parte da zero)
    merged[TIME_COL] -= merged[TIME_COL].iloc[0]

    return merged


def plot_method(ax, data: pd.DataFrame, method: str):
    perp = f"Group: {method}_new - train/Step_perplexity"
    perp_min = f"Group: {method}_new - train/Step_perplexity__MIN"
    perp_max = f"Group: {method}_new - train/Step_perplexity__MAX"

    sns.lineplot(
        ax=ax,
        data=data,
        x=TIME_COL,
        y=perp,
        label=method,
    )

    ax.fill_between(
        data[TIME_COL],
        data[perp_min],
        data[perp_max],
        alpha=0.2,
    )


# =========================
# Main
# =========================
def main():
    fig, ax = plt.subplots(figsize=(16, 9))

    step_df = load_step_data()

    for method in METHODS:
        time_df = load_time_data(method)
        processed = preprocess(method, time_df, step_df)
        plot_method(ax, processed, method)

    ax.set(
        xlabel=X_LABEL,
        ylabel=Y_LABEL,
        xlim=X_LIM,
        ylim=Y_LIM,
        yscale="log",
    )

    # =========================
    # Log-scale grid più densa
    # =========================
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))
    )
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    ax.grid(which="major", linestyle="-", linewidth=1, alpha=1)
    ax.grid(which="minor", linestyle="--", linewidth=0.9, alpha=1)

    ax.set_xticks(XTICKS)
    ax.set_xticklabels(XTICKLABELS)

    ax.legend()
    fig.tight_layout()

    fig.savefig("out.png", dpi=300, bbox_inches="tight")
    fig.savefig("time_to_perp.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
