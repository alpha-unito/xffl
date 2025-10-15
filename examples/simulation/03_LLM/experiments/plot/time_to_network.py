import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)

logs = (
    "/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/learning_logs"
)

# Read time-to-network data
dtype = {
    "Relative Time (Process)": float,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent": int,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent__MIN": int,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent__MAX": int,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv": int,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv__MIN": int,
    # "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv__MAX": int,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent": int,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent__MIN": int,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent__MAX": int,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv": int,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv__MIN": int,
    # "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv__MAX": int,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent": int,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent__MIN": int,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent__MAX": int,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv": int,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv__MIN": int,
    # "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv__MAX": int,
}

data = pd.read_csv(
    filepath_or_buffer=f"{logs}/time_to_network.csv",
    sep=",",
    dtype=dtype,
    engine="pyarrow",
)


data_fsdp = data[
    [
        "Relative Time (Process)",
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent",
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv",
    ]
]
data_fsdp = (
    data_fsdp.dropna()
    .astype(
        {
            "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent": float,
            "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv": float,
            "Relative Time (Process)": int,
        }
    )
    .groupby("Relative Time (Process)")
    .mean()
    .reset_index()
)

data_fsdp["Relative Time (Process)"] = (
    data_fsdp["Relative Time (Process)"] - data_fsdp["Relative Time (Process)"][0]
)


data_hsdp = data[
    [
        "Relative Time (Process)",
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent",
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv",
    ]
]
data_hsdp = (
    data_hsdp.dropna()
    .astype(
        {
            "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent": float,
            "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv": float,
            "Relative Time (Process)": int,
        }
    )
    .groupby("Relative Time (Process)")
    .mean()
    .reset_index()
)

data_hsdp["Relative Time (Process)"] = (
    data_hsdp["Relative Time (Process)"] - data_hsdp["Relative Time (Process)"][0]
)


data_fl = data[
    [
        "Relative Time (Process)",
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent",
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv",
    ]
]
data_fl = (
    data_fl.dropna()
    .astype(
        {
            "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent": float,
            "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv": float,
            "Relative Time (Process)": int,
        }
    )
    .groupby("Relative Time (Process)")
    .mean()
    .reset_index()
)

data_fl["Relative Time (Process)"] = (
    data_fl["Relative Time (Process)"] - data_fl["Relative Time (Process)"][0]
)

# Plot
xlabel = "Time"
xlim = (0, 1150)
xticks = [0, 200, 400, 600, 800, 1000]

ylabel = "Bytes"
ylim = (-40000000, 6e9)
yticks = [
    0,
    1e9,
    2e9,
    3e9,
    4e9,
    5e9,
    6e9,
]


fig = sns.lineplot(
    data=data_fsdp,
    x="Relative Time (Process)",
    y="Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent",
    label="FSDP - data sent",
    color=sns.color_palette()[0],
)
fig = sns.lineplot(
    data=data_fsdp,
    x="Relative Time (Process)",
    y="Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv",
    label="HSDP - data received",
    linestyle=":",
    color=sns.color_palette()[0],
)

fig = sns.lineplot(
    data=data_hsdp,
    x="Relative Time (Process)",
    y="Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent",
    label="FSDP - data sent",
    color=sns.color_palette()[1],
)
fig = sns.lineplot(
    data=data_hsdp,
    x="Relative Time (Process)",
    y="Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv",
    label="HSDP - data received",
    linestyle=":",
    color=sns.color_palette()[1],
)

fig = sns.lineplot(
    data=data_fl,
    x="Relative Time (Process)",
    y="Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent",
    label="Intra-Silo FedAvg - data sent",
    color=sns.color_palette()[2],
)
fig = sns.lineplot(
    data=data_fl,
    x="Relative Time (Process)",
    y="Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv",
    label="Intra-Silo FedAvg - data received",
    linestyle=":",
    color=sns.color_palette()[2],
)


fig.set(
    xlabel=xlabel,
    xticks=xticks,
    xlim=xlim,
    ylabel=ylabel,
    yticks=yticks,
    ylim=ylim,
)
# plt.yscale("log")

fig.legend()

fig.figure.savefig("out.png", bbox_inches="tight", dpi=300, format="png")
fig.figure.savefig("time_to_network.pdf", bbox_inches="tight", dpi=300, format="pdf")
