import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)

logs = (
    "/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/learning_logs"
)

# Read time-to-loss data

dtype = {
    "Relative Time (Process)": float,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - _step": int,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - _step__MIN": int,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - _step__MAX": int,
    "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss": float,
    "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MIN": float,
    "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MAX": float,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step": int,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step__MIN": int,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step__MAX": int,
    "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss": float,
    "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN": float,
    "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX": float,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step": int,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step__MIN": int,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step__MAX": int,
    "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss": float,
    "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN": float,
    "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX": float,
}

data = pd.read_csv(
    filepath_or_buffer=f"{logs}/time_to_loss.csv",
    sep=",",
    dtype=dtype,
    engine="pyarrow",
)

# Read time-to-loss data

dtype = {
    "Step": int,
    "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss": float,
    "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MIN": float,
    "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MAX": float,
    "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss": float,
    "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN": float,
    "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX": float,
    "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss": float,
    "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN": float,
    "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX": float,
}

data_step = pd.read_csv(
    filepath_or_buffer=f"{logs}/step_to_loss.csv",
    sep=",",
    dtype=dtype,
    engine="pyarrow",
)


# Pre-process FSDP
data_fsdp = data[
    [
        "Relative Time (Process)",
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step",
    ]
]
data_fsdp = (
    data_fsdp.dropna()
    .rename(
        columns={
            "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step": "Step",
        }
    )
    .astype(
        {
            "Step": int,
        }
    )
    .groupby("Step")
    .mean()
    .astype(
        {
            "Relative Time (Process)": int,
        }
    )
    .reset_index()
)

data_step_fsdp = data_step[
    [
        "Step",
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss",
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN",
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX",
    ]
]

data_fsdp = pd.merge(data_fsdp, data_step_fsdp, on="Step")

# Pre-process HSDP
data_hsdp = data[
    [
        "Relative Time (Process)",
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step",
    ]
]
data_hsdp = (
    data_hsdp.dropna()
    .rename(
        columns={
            "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - _step": "Step",
        }
    )
    .astype(
        {
            "Step": int,
        }
    )
    .groupby("Step")
    .mean()
    .astype(
        {
            "Relative Time (Process)": int,
        }
    )
    .reset_index()
)

data_step_hsdp = data_step[
    [
        "Step",
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss",
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN",
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX",
    ]
]

data_hsdp = pd.merge(data_hsdp, data_step_hsdp, on="Step")

# Pre-process FSDP
data_fl = data[
    [
        "Relative Time (Process)",
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - _step",
    ]
]
data_fl = (
    data_fl.dropna()
    .rename(
        columns={
            "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - _step": "Step",
        }
    )
    .astype(
        {
            "Step": int,
        }
    )
    .groupby("Step")
    .mean()
    .astype(
        {
            "Relative Time (Process)": int,
        }
    )
    .reset_index()
)

data_step_fl = data_step[
    [
        "Step",
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss",
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MIN",
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MAX",
    ]
]

data_fl = pd.merge(data_fl, data_step_fl, on="Step")

# Time adjustment
data_fsdp["Relative Time (Process)"] = (
    data_fsdp["Relative Time (Process)"] - data_fsdp["Relative Time (Process)"][0]
)
data_hsdp["Relative Time (Process)"] = (
    data_hsdp["Relative Time (Process)"] - data_hsdp["Relative Time (Process)"][0]
)
data_fl["Relative Time (Process)"] = (
    data_fl["Relative Time (Process)"] - data_fl["Relative Time (Process)"][0]
)


# Plot

xlabel = "Time (s)"
xlim = (0, 1050)
xticks = [0, 200, 400, 600, 800, 1000]

ylabel = "Loss"
ylim = (3.5, 14.5)
yticks = [0, 2, 4, 6, 8, 10, 12, 14, 16]

fig = sns.lineplot(
    data=data_fsdp,
    x="Relative Time (Process)",
    y="Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss",
    label="FSDP",
)
fig.fill_between(
    data_fsdp["Relative Time (Process)"],
    data_fsdp[
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN"
    ],
    data_fsdp[
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX"
    ],
    alpha=0.2,
)

fig = sns.lineplot(
    data=data_hsdp,
    x="Relative Time (Process)",
    y="Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss",
    label="HSDP",
)
fig.fill_between(
    data_hsdp["Relative Time (Process)"],
    data_hsdp[
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN"
    ],
    data_hsdp[
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX"
    ],
    alpha=0.2,
)

fig = sns.lineplot(
    data=data_fl,
    x="Relative Time (Process)",
    y="Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss",
    label="Intra-Silo FedAvg",
)
fig.fill_between(
    data_fl["Relative Time (Process)"],
    data_fl[
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MIN"
    ],
    data_fl[
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MAX"
    ],
    alpha=0.2,
)

fig.set(
    xlabel=xlabel,
    xticks=xticks,
    xlim=xlim,
    ylabel=ylabel,
    yticks=yticks,
    ylim=ylim,
)

fig.legend()

fig.figure.savefig("out.png", bbox_inches="tight", dpi=300, format="png")
fig.figure.savefig("time_to_loss.pdf", bbox_inches="tight", dpi=300, format="pdf")
