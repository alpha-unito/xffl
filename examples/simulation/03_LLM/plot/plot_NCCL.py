import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# -- General settings --
FONT_SIZE = 10

sns.set_theme(style="darkgrid")

plt.rc("xtick", labelsize=FONT_SIZE)
plt.rc("ytick", labelsize=FONT_SIZE)


# -- Data extraction --
logs = "/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/logs"

model = "llama3.1-8b"
fs = 1
processes_per_node = 1

data = {}
for nodes in [2, 4, 8, 16, 32, 64, 128]:
    data[nodes] = pd.read_csv(
        filepath_or_buffer=f"{logs}/{model}_ns_{nodes}_fs_{fs}_ppn_{processes_per_node}.csv",
        sep=",",
        dtype={
            "Aggregation Strategy": str,
            "NCCL Algo": str,
            "NCLL Proto": str,
            "Multiple CUDA Streams": bool,
            "Contiguous Memory": bool,
            "Average Time (s) - {nodes}": float,
            "Theoretical Throughput (Gb/s) - {nodes}": float,
            "Real Throughput (Gb/s) - {nodes}": float,
            "Max GPU RAM Allocated (GB) - {nodes}": float,
        },
        engine="pyarrow",
    )
    data[nodes]["nodes"] = nodes

data = pd.concat([dataframe for _, dataframe in data.items()])

data_layer = data[
    (data["Aggregation Strategy"] == "layer_by_layer")
    & (data["Multiple CUDA Streams"] == True)
    & (data["Contiguous Memory"] == True)
]

data_bucket = data[
    (data["Aggregation Strategy"] == "bucket_flatten")
    & (data["Multiple CUDA Streams"] == True)
    & (data["Contiguous Memory"] == True)
]

data_coalesced = data[
    (data["Aggregation Strategy"] == "bucket_coalesced")
    & (data["Multiple CUDA Streams"] == True)
    & (data["Contiguous Memory"] == True)
]

data_layer_optimized = data[
    (data["Aggregation Strategy"] == "layer_by_layer_optimized")
    & (data["Multiple CUDA Streams"] == True)
    & (data["Contiguous Memory"] == True)
]

data_bucket_optimized = data[
    (data["Aggregation Strategy"] == "bucket_optimized_flatten")
    & (data["Multiple CUDA Streams"] == True)
    & (data["Contiguous Memory"] == True)
]

data_coalesced_optimized = data[
    (data["Aggregation Strategy"] == "bucket_optimized_coalesced")
    & (data["Multiple CUDA Streams"] == True)
    & (data["Contiguous Memory"] == True)
]

data_layer_ = data[
    (data["Aggregation Strategy"] == "layer_by_layer")
    & (data["Multiple CUDA Streams"] == False)
    & (data["Contiguous Memory"] == False)
]

data_bucket_ = data[
    (data["Aggregation Strategy"] == "bucket_flatten")
    & (data["Multiple CUDA Streams"] == False)
    & (data["Contiguous Memory"] == False)
]

data_coalesced_ = data[
    (data["Aggregation Strategy"] == "bucket_coalesced")
    & (data["Multiple CUDA Streams"] == False)
    & (data["Contiguous Memory"] == False)
]

data_layer_optimized_ = data[
    (data["Aggregation Strategy"] == "layer_by_layer_optimized")
    & (data["Multiple CUDA Streams"] == False)
    & (data["Contiguous Memory"] == False)
]

data_bucket_optimized_ = data[
    (data["Aggregation Strategy"] == "bucket_optimized_flatten")
    & (data["Multiple CUDA Streams"] == False)
    & (data["Contiguous Memory"] == False)
]

data_coalesced_optimized_ = data[
    (data["Aggregation Strategy"] == "bucket_optimized_coalesced")
    & (data["Multiple CUDA Streams"] == False)
    & (data["Contiguous Memory"] == False)
]

# -- Subplot ---

xticks = [2, 4, 8, 16, 32, 64, 128]
yticks = [80, 90, 100, 110, 120]
ytickslabels = [80, 90, 100, 110, 120]
x_lim = (2, 128)
y_lim = (80, 120)

fig, ax = plt.subplots(
    nrows=6,
    ncols=2,
    sharex=True,
    sharey=True,
    figsize=(10, 9),
    dpi=300,
    layout="tight",
    subplot_kw={
        "xlabel": None,
        "ylabel": None,
        "xscale": "log",
        "ylim": y_lim,
        "yticks": yticks,
        "yticklabels": ytickslabels,
        "xlim": x_lim,
        "xticks": xticks,
        "xticklabels": xticks,
    },
)
fig.supylabel("Throughput (Gb/s)", fontsize=14, y=0.45)
fig.supxlabel("Number of nodes", fontsize=14, x=0.75, y=0.03)

ax[0][0].set_title("With optimizations (multiple streams + contiguous memory)")
ax[0][1].set_title("Without optimizations")

ax[0][0].set_ylabel("Layer")
ax[1][0].set_ylabel("Layer opt.")
ax[2][0].set_ylabel("Bucket")
ax[3][0].set_ylabel("Bucket opt.")
ax[4][0].set_ylabel("Coalesced")
ax[5][0].set_ylabel("Coalesced opt.")

hue_order = [
    "ring",
    "tree",
    "collnet",
    "collnetchain",
    "collnetdirect",
    "nvls",
    "nvlstree",
    "pat",
]
style_order = [
    "LL",
    "LL128",
    "SIMPLE",
]
palette = dict(
    zip(
        hue_order,
        sns.color_palette(palette="husl", n_colors=8),
    )
)

# -- Plotting --

fig1 = sns.lineplot(
    data=data_layer,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    legend="full",
    sort=True,
    ax=ax[0][0],
)

fig2 = sns.lineplot(
    data=data_layer_optimized,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[1][0],
)

fig3 = sns.lineplot(
    data=data_bucket,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[2][0],
)

fig4 = sns.lineplot(
    data=data_bucket_optimized,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[3][0],
)

fig5 = sns.lineplot(
    data=data_coalesced,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[4][0],
)

fig6 = sns.lineplot(
    data=data_coalesced_optimized,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[5][0],
)

# ----- Secondo column -----

fig7 = sns.lineplot(
    data=data_layer_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[0][1],
)

fig8 = sns.lineplot(
    data=data_layer_optimized_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[1][1],
)

fig9 = sns.lineplot(
    data=data_bucket_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[2][1],
)

fig10 = sns.lineplot(
    data=data_bucket_optimized_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[3][1],
)

fig11 = sns.lineplot(
    data=data_coalesced_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[4][1],
)

fig12 = sns.lineplot(
    data=data_coalesced_optimized_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue="NCCL Algo",
    hue_order=hue_order,
    palette=palette,
    style="NCLL Proto",
    style_order=style_order,
    legend=None,
    sort=True,
    ax=ax[5][1],
)

for axis in ax.flat:
    axis.set_xlabel(None)


fig1.legend(
    title=None,
    loc="center right",
    handles=fig1.get_legend_handles_labels()[0],
    labels=[
        "NCCL Algorithm",
        "Ring",
        "Tree",
        "Collnet",
        "CollnetChain",
        "CollnetDirect",
        "NVLS",
        "NVLSTree",
        "PAT",
        "NCLL Protocol",
        "SIMPLE",
        "LL",
        "LL128",
    ],
    ncol=1,
    bbox_to_anchor=(2.4, -2.3),
    fontsize=12,
)

plt.subplots_adjust(hspace=0.3, wspace=0.05, bottom=0.1, top=0.8, right=1.4)
fig.figure.savefig("out.png", bbox_inches="tight", dpi=300, format="png")
fig.figure.savefig("multiple-node.pdf", bbox_inches="tight", dpi=300, format="pdf")
quit()
