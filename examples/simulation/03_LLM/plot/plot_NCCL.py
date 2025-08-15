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

fig, ax = plt.subplots(
    nrows=6,
    ncols=2,
    sharex="all",
    figsize=(10, 10),
)


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
    "Layer\noptimized",
    "Bucket",
    "Bucket\noptimized",
    "Coalesced",
    "Coalesced\noptimized",
]

palette = {
    (False, False): "limegreen",
    (False, True): "green",
    (True, False): "lightskyblue",
    (True, True): "dodgerblue",
}

fig1 = sns.lineplot(
    data=data_layer,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_layer[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    legend="full",
    # palette=palette,
    # err_style="bars",
    # errorbar=("pi", 100),
    sort=True,
    ax=ax[0][0],
)
fig1.set(
    ylabel="Layer",
    ylim=y_lim,
)
# fig1.set_xticks([], minor=True)
# fig1.set_xticks(ticks=xticks)
ax[0][0].set_title("With optimizations (streams/contiguous)")
fig1.grid(True, axis="both")

fig2 = sns.lineplot(
    data=data_layer_optimized,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_layer_optimized[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[1][0],
)
fig2.set(
    ylabel="Layer optimized",
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
)
fig2.grid(True, axis="both")

fig3 = sns.lineplot(
    data=data_bucket,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_bucket[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[2][0],
)
fig3.set(
    ylabel="Bucket",
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
)
fig3.grid(True, axis="both")

fig4 = sns.lineplot(
    data=data_bucket_optimized,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_bucket_optimized[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[3][0],
)
fig4.set(
    ylabel="Bucket optimized",
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
)
fig4.grid(True, axis="both")

fig5 = sns.lineplot(
    data=data_coalesced,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_coalesced[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[4][0],
)
fig5.set(
    ylabel="Coalesced",
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
)
fig5.grid(True, axis="both")

fig6 = sns.lineplot(
    data=data_coalesced_optimized,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_coalesced_optimized[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[5][0],
)
fig6.set(
    ylabel="Coalesced optimized",
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
)
fig6.grid(True, axis="both")


fig7 = sns.lineplot(
    data=data_layer_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_layer_[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    legend=None,
    # palette=palette,
    # err_style="bars",
    # errorbar=("pi", 100),
    sort=True,
    ax=ax[0][1],
)
fig7.set(ylabel=None, ylim=y_lim, yticklabels=[])
# fig1.set_xticks([], minor=True)
# fig1.set_xticks(ticks=xticks)
fig1.grid(True, axis="both")
ax[0][1].set_title("Without optimizations (streams/contiguous)")

fig8 = sns.lineplot(
    data=data_layer_optimized_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_layer_optimized_[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[1][1],
)
fig8.set(
    ylabel=None,
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
    yticklabels=[],
)
fig8.grid(True, axis="both")

fig9 = sns.lineplot(
    data=data_bucket_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_bucket_[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[2][1],
)
fig9.set(
    ylabel=None,
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
    yticklabels=[],
)
fig9.grid(True, axis="both")

fig10 = sns.lineplot(
    data=data_bucket_optimized_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_bucket_optimized_[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[3][1],
)
fig10.set(
    ylabel=None,
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
    yticklabels=[],
)
fig10.grid(True, axis="both")

fig11 = sns.lineplot(
    data=data_coalesced_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_coalesced_[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[4][1],
)
fig11.set(
    ylabel=None,
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
    yticklabels=[],
)
fig11.grid(True, axis="both")

fig12 = sns.lineplot(
    data=data_coalesced_optimized_,
    x="nodes",
    y="Real Throughput (Gb/s)",
    hue=data_coalesced_optimized_[["NCCL Algo", "NCLL Proto"]].apply(tuple, axis=1),
    # hue_order=palette.keys(),
    # style="Aggregation Strategy",
    # style_order=style_order,
    # palette=palette,
    err_style="bars",
    errorbar=("pi", 100),
    legend=None,
    sort=True,
    ax=ax[5][1],
)
fig12.set(
    ylabel=None,
    ylim=y_lim,
    xlim=x_lim,
    xlabel=None,
    xscale="log",
    xticks=xticks,
    xticklabels=xticks,
    yticklabels=[],
)
fig12.grid(True, axis="both")


fig1.legend(
    title=None,
    loc="upper right",
    # handles=fig1.get_legend_handles_labels()[0],
    # labels=["Optimization"]
    # + [
    #    "No optimizations",
    #    "Contiguous\nMemory",
    #    "Multiple\nCUDA Streams",
    #    "Both",
    # ]
    # + ["\nAggregation algorithm"]
    # + labels,
    ncol=1,
    bbox_to_anchor=(2.6, 1),
    fontsize=12,
)


fig.supylabel("Throughput (Gb/s)", fontsize=14, y=0.5)
fig.supxlabel("Number of nodes", fontsize=14, x=0.75, y=0)
plt.subplots_adjust(hspace=0.1, wspace=0.1, bottom=0.1, top=0.8, right=1.4)
fig.figure.savefig("out.png", bbox_inches="tight", dpi=300, format="png")
# fig.figure.savefig("multiple-node.pdf", bbox_inches="tight", dpi=300, format="pdf")
quit()
