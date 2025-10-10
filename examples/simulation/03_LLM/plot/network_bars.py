import matplotlib.pyplot as plt
import numpy as np
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

bar_data = pd.DataFrame(columns=["algo", "type", "bytes"])

# data_fsdp[
#     "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
# ] = data_fsdp[
#     "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
# ].diff()

# data_fsdp[
#     "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
# ] = data_fsdp[
#     "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
# ].diff()

data_fsdp = (
    data_fsdp.dropna()
    .astype(
        {
            "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent": int,
            "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv": int,
            "Relative Time (Process)": int,
        }
    )
    .groupby("Relative Time (Process)")
    .mean()
    .reset_index()
)
# data_fsdp = data_fsdp[data_fsdp["Relative Time (Process)"] >= 136]
# data_fsdp = data_fsdp.reset_index(drop=True)

bar_data.loc[len(bar_data)] = [
    "FSDP",
    "Data sent",
    data_fsdp[
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
    ]
    .iloc[-1]
    .item(),
]

bar_data.loc[len(bar_data)] = [
    "FSDP",
    "Data received",
    data_fsdp[
        "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
    ]
    .iloc[-1]
    .item(),
]

# Time adjustment
# data_fsdp[
#     "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
# ] = (
#     data_fsdp[
#         "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
#     ]
#     - data_fsdp[
#         "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
#     ][0]
# )
# data_fsdp[
#     "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
# ] = (
#     data_fsdp[
#         "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
#     ]
#     - data_fsdp[
#         "Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
#     ][0]
# )

# data_fsdp["Relative Time (Process)"] = (
#     data_fsdp["Relative Time (Process)"] - data_fsdp["Relative Time (Process)"][0]
# )


data_hsdp = data[
    [
        "Relative Time (Process)",
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent",
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv",
    ]
]

# data_hsdp[
#     "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
# ] = data_hsdp[
#     "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
# ].diff()

# data_hsdp[
#     "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
# ] = data_hsdp[
#     "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
# ].diff()

data_hsdp = (
    data_hsdp.dropna()
    .astype(
        {
            "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent": int,
            "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv": int,
            "Relative Time (Process)": int,
        }
    )
    .groupby("Relative Time (Process)")
    .mean()
    .reset_index()
)
# data_hsdp = data_hsdp[data_hsdp["Relative Time (Process)"] >= 150]
# data_hsdp = data_hsdp.reset_index(drop=True)

bar_data.loc[len(bar_data)] = [
    "HSDP",
    "Data sent",
    data_hsdp[
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
    ]
    .iloc[-1]
    .item(),
]

bar_data.loc[len(bar_data)] = [
    "HSDP",
    "Data received",
    data_hsdp[
        "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
    ]
    .iloc[-1]
    .item(),
]

# Time adjustment
# data_hsdp[
#     "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
# ] = (
#     data_hsdp[
#         "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
#     ]
#     - data_hsdp[
#         "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent"
#     ][0]
# )
# data_hsdp[
#     "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
# ] = (
#     data_hsdp[
#         "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
#     ]
#     - data_hsdp[
#         "Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv"
#     ][0]
# )

# data_hsdp["Relative Time (Process)"] = (
#     data_hsdp["Relative Time (Process)"] - data_hsdp["Relative Time (Process)"][0]
# )


data_fl = data[
    [
        "Relative Time (Process)",
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent",
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv",
    ]
]

# data_fl[
#     "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent"
# ] = data_fl[
#     "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent"
# ].diff()

# data_fl[
#     "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv"
# ] = data_fl[
#     "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv"
# ].diff()

data_fl = (
    data_fl.dropna()
    .astype(
        {
            "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent": int,
            "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv": int,
            "Relative Time (Process)": int,
        }
    )
    .groupby("Relative Time (Process)")
    .mean()
    .reset_index()
)
# data_fl = data_fl[data_fl["Relative Time (Process)"] >= 137]
# data_fl = data_fl.reset_index(drop=True)

bar_data.loc[len(bar_data)] = [
    "Intra-silo FedAvg",
    "Data sent",
    data_fl[
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent"
    ]
    .iloc[-1]
    .item(),
]

bar_data.loc[len(bar_data)] = [
    "Intra-silo FedAvg",
    "Data received",
    data_fl[
        "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv"
    ]
    .iloc[-1]
    .item(),
]

# Time adjustment
# data_fl[
#     "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent"
# ] = (
#     data_fl[
#         "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent"
#     ]
#     - data_fl[
#         "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent"
#     ][0]
# )
# data_fl[
#     "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv"
# ] = (
#     data_fl[
#         "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv"
#     ]
#     - data_fl[
#         "Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv"
#     ][0]
# )

# data_fl["Relative Time (Process)"] = (
#     data_fl["Relative Time (Process)"] - data_fl["Relative Time (Process)"][0]
# )

# Plot
xlabel = "Training algorithm"
# xlim = (0, 256)
# xticks = [0, 50, 100, 150, 200, 250]

ylabel = "Cumulative network traffic (bytes)"
ylim = (1, 1e10)
yticks = [1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10]  # 1e0, 1e1, 1e2, 1e3, 1e4, 1e5,
yticklabels = ["1e6", "5e6", "1e7", "5e7", "1e8", "5e8", "1e9", "5e9", "1e10"]

# fig = sns.lineplot(
#     data=data_fsdp,
#     x=data_fsdp.index * 255 / data_fsdp.index[-1],  # x="Relative Time (Process)",
#     y="Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent",
#     label="FSDP - data sent",
#     color=sns.color_palette()[0],
# )
# fig = sns.lineplot(
#     data=data_fsdp,
#     x=data_fsdp.index * 255 / data_fsdp.index[-1],  # x="Relative Time (Process)",
#     y="Group: PAP_FSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv",
#     label="FSDP - data received",
#     linestyle=":",
#     color=sns.color_palette()[0],
# )

# fig = sns.lineplot(
#     data=data_hsdp,
#     x=data_hsdp.index * 255 / data_hsdp.index[-1],  # "Relative Time (Process)",
#     y="Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.sent",
#     label="HSDP - data sent",
#     color=sns.color_palette()[1],
# )
# fig = sns.lineplot(
#     data=data_hsdp,
#     x=data_hsdp.index * 255 / data_hsdp.index[-1],  # "Relative Time (Process)",
#     y="Group: PAP_HSDP_llama3.1-8b_clean_mc4_it_ns_128_ppn_1 - system/network.recv",
#     label="HSDP - data received",
#     linestyle=":",
#     color=sns.color_palette()[1],
# )

# fig = sns.lineplot(
#     data=data_fl,
#     x=data_fl.index * 255 / data_fl.index[-1],  # "Relative Time (Process)",
#     y="Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.sent",
#     label="Intra-Silo FedAvg - data sent",
#     color=sns.color_palette()[2],
# )
# fig = sns.lineplot(
#     data=data_fl,
#     x=data_fl.index * 255 / data_fl.index[-1],  # "Relative Time (Process)",
#     y="Group: PAP_FL_llama3.1-8b_clean_mc4_it_ns_128_ppn_1_fs_1_fb_8 - system/network.recv",
#     label="Intra-Silo FedAvg - data received",
#     linestyle=":",
#     color=sns.color_palette()[2],
# )

fig = sns.barplot(
    data=bar_data,
    x="type",
    y="bytes",
    hue="algo",
    # label="FSDP - data sent",
    # color=sns.color_palette()[0],
)

plt.yscale("log")

fig.set(
    xlabel=xlabel,
    # xticks=xticks,
    # xticklabels=xticks,
    # xlim=xlim,
    ylabel=ylabel,
    yticks=yticks,
    yticklabels=yticklabels,
    ylim=ylim,
)

fig.legend()

print(bar_data.iloc[0]["bytes"])
quit()

fig.figure.savefig("out.png", bbox_inches="tight", dpi=300, format="png")
fig.figure.savefig("network_bars.pdf", bbox_inches="tight", dpi=300, format="pdf")
