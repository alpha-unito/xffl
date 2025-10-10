import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)

logs = (
    "/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/learning_logs"
)

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

data = pd.read_csv(
    filepath_or_buffer=f"{logs}/step_to_loss.csv",
    sep=",",
    dtype=dtype,
    engine="pyarrow",
)

xlabel = "Training step"
xlim = (0, 256)
xticks = [0, 50, 100, 150, 200, 250]

ylabel = "Loss"
ylim = (3.5, 14.5)
yticks = [0, 2, 4, 6, 8, 10, 12, 14, 16]


fig = sns.lineplot(
    data=data,
    x="Step",
    y="Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss",
    label="FSDP",
)
fig.fill_between(
    data["Step"],
    data["Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN"],
    data["Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX"],
    alpha=0.2,
)

fig = sns.lineplot(
    data=data,
    x="Step",
    y="Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss",
    label="HSDP",
)
fig.fill_between(
    data["Step"],
    data["Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MIN"],
    data["Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - train/loss__MAX"],
    alpha=0.2,
)

fig = sns.lineplot(
    data=data,
    x="Step",
    y="Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss",
    label="Intra-Silo FedAvg",
)
fig.fill_between(
    data["Step"],
    data[
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - train/loss__MIN"
    ],
    data[
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
fig.figure.savefig("step_to_loss.pdf", bbox_inches="tight", dpi=300, format="pdf")
