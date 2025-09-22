import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)

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

fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex="all",
    figsize=(10, 10),
)

hue = data[["Multiple CUDA Streams", "Contiguous Memory"]].apply(tuple, axis=1)

xticks = [2, 4, 8, 16, 32, 64, 128]
yticks = [80, 90, 100, 110, 120]
x_lim = (1.98, 129)
y_lim = (84, 120)

style_order = [
    "layer_by_layer",
    "layer_by_layer_optimized",
    "bucket_flatten",
    "bucket_optimized_flatten",
    "bucket_coalesced",
    "bucket_optimized_coalesced",
]
order = [
    "layer_by_layer",
    "layer_by_layer_optimized",
    "bucket_flatten",
    "bucket_optimized_flatten",
    "bucket_coalesced",
    "bucket_optimized_coalesced",
]
labels = [
    "Layer",
    "Layer\nbalanced",
    "Bucket",
    "Bucket\nbalanced",
    "Coalesced",
    "Coalesced\nbalanced",
]

palette = {
    (False, False): "limegreen",
    (False, True): "green",
    (True, False): "lightskyblue",
    (True, True): "dodgerblue",
}

fig1 = sns.lineplot(
    data=data,
    x="nodes",
    y="Theoretical Throughput (Gb/s)",
    hue=hue,
    hue_order=palette.keys(),
    style="Aggregation Strategy",
    style_order=style_order,
    legend="full",
    palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    sort=True,
    ax=ax[0],
)
fig1.set(
    ylabel="Without memory overheads",
    ylim=y_lim,
)
# fig1.set_xticks([], minor=True)
# fig1.set_xticks(ticks=xticks)
# fig1.grid(True, axis="both")

fig2 = sns.lineplot(
    data=data,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=hue,
    hue_order=palette.keys(),
    style="Aggregation Strategy",
    style_order=style_order,
    palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[1],
)
fig2.set(
    ylabel="With memory overheads",
    ylim=y_lim,
    xlim=x_lim,
    xlabel="Number of nodes",
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
)
fig2.grid(True, axis="both")

fig1.legend(
    title=None,
    loc="upper right",
    handles=fig1.get_legend_handles_labels()[0],
    labels=["Optimization"]
    + [
        "No optimizations",
        "Contiguous\nMemory",
        "Multiple\nCUDA Streams",
        "Both",
    ]
    + ["\nAggregation algorithm"]
    + labels,
    ncol=1,
    bbox_to_anchor=(1.21, 0.92),
    fontsize=12,
)

fig.supylabel("Throughput (Gb/s)", fontsize=14, y=0.37)
plt.subplots_adjust(hspace=0.07, wspace=0.06, bottom=0.15, top=0.6, right=1.4)
fig.figure.savefig("out.png", bbox_inches="tight", dpi=300, format="png")
fig.figure.savefig("multiple-node.pdf", bbox_inches="tight", dpi=300, format="pdf")
