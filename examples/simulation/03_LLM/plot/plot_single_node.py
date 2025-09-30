import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")

plt.rc("xtick", labelsize=12)
plt.rc("ytick", labelsize=12)

logs = "/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/logs"

model = "llama3.1-8b"
nodes = 1
fs = 1

dtype = {
    "Aggregation Strategy": str,
    "NCCL Algo": str,
    "NCLL Proto": str,
    "Multiple CUDA Streams": bool,
    "Contiguous Memory": bool,
    "Average Time (s)": float,
    "Theoretical Throughput (Gb/s)": float,
    "Real Throughput (Gb/s)": float,
    "Max GPU RAM Allocated (GB)": float,
}

data_local_2 = pd.read_csv(
    filepath_or_buffer=f"{logs}/{model}_ns_{nodes}_fs_{fs}_ppn_2.csv",
    sep=",",
    dtype=dtype,
    engine="pyarrow",
)

data_local_4 = pd.read_csv(
    filepath_or_buffer=f"{logs}/{model}_ns_{nodes}_fs_{fs}_ppn_4.csv",
    sep=",",
    dtype=dtype,
    engine="pyarrow",
)


fig, ax = plt.subplots(
    nrows=2,
    ncols=2,
    sharex="all",
    figsize=(10, 10),
)

hue = data_local_2[["Multiple CUDA Streams", "Contiguous Memory"]].apply(tuple, axis=1)
yticks_2 = [450, 500, 550, 600, 650]
yticks_4 = [1200, 1400, 1600, 1800]
y_lim_2 = (425, 675)
y_lim_4 = (1150, 1850)
size = 5

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

fig1 = sns.swarmplot(
    data=data_local_2,
    x="Aggregation Strategy",
    y="Theoretical Throughput (Gb/s)",
    hue=hue,
    size=size,
    palette=palette,
    legend="full",
    order=order,
    ax=ax[0][0],
)
fig1.set(
    title="Without memory overheads",
    ylabel="2 GPUs",
    yticks=yticks_2,
    ylim=y_lim_2,
    xticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    xticklabels=[],
)
fig1.set_xticks(ticks=[0, 1, 2, 3, 4, 5], minor=True, labels=labels)
fig1.grid(True, axis="both")

fig2 = sns.swarmplot(
    data=data_local_2,
    x="Aggregation Strategy",
    y="Real Throughput (Gb/s)",
    hue=hue,
    size=size,
    palette=palette,
    legend=None,
    order=order,
    ax=ax[0][1],
)
fig2.set(
    ylabel=None,
    yticks=yticks_2,
    title="With memory overheads",
    yticklabels=[],
    ylim=y_lim_2,
)
fig2.grid(True, axis="both")

fig3 = sns.swarmplot(
    data=data_local_4,
    x="Aggregation Strategy",
    y="Theoretical Throughput (Gb/s)",
    hue=hue,
    size=size,
    palette=palette,
    legend=None,
    order=order,
    ax=ax[1][0],
)
fig3.set(
    xlabel=None,
    ylabel="4 GPUs",
    yticks=yticks_4,
    ylim=y_lim_4,
)
fig3.tick_params(axis="x", labelrotation=45, which="minor")
fig3.grid(True, axis="both")

fig4 = sns.swarmplot(
    data=data_local_4,
    x="Aggregation Strategy",
    y="Real Throughput (Gb/s)",
    hue=hue,
    size=size,
    palette=palette,
    legend=None,
    order=order,
    ax=ax[1][1],
)
fig4.set(
    ylabel=None,
    xlabel=None,
    yticks=yticks_4,
    yticklabels=[],
    ylim=y_lim_4,
)
fig4.tick_params(axis="x", labelrotation=45, which="minor")
fig4.grid(True, axis="both")

# fig.suptitle("Single-node FedAVG performance", fontsize=16, x=0.5, y=1.01)
# fig.supxlabel("FedAvg implementation", fontsize=14, x=0.5)
fig.supylabel("Throughput (Gb/s)", fontsize=14, y=0.5)

fig1.legend(
    title=None,
    loc="upper center",
    handles=fig1.get_legend_handles_labels()[0],
    labels=["No optimizations", "Contiguous Memory", "Multiple CUDA Streams", "Both"],
    ncol=4,
    bbox_to_anchor=(0.92, 1.21),
    fontsize=12,
)

plt.subplots_adjust(hspace=0.07, wspace=0.02, bottom=0.15, top=0.9)
fig.figure.savefig("out.png", bbox_inches="tight", dpi=300, format="png")
fig.figure.savefig("single-node.pdf", bbox_inches="tight", dpi=300, format="pdf")

coalesced_t = data_local_4[(data_local_4["Aggregation Strategy"] == "bucket_coalesced")]
coalesced_optimized_t = data_local_4[
    (data_local_4["Aggregation Strategy"] == "bucket_optimized_coalesced")
]
print(
    coalesced_t[
        coalesced_t["Theoretical Throughput (Gb/s)"]
        == coalesced_t["Theoretical Throughput (Gb/s)"].min()
    ].to_string()
)
print(
    coalesced_t[
        coalesced_t["Real Throughput (Gb/s)"]
        == coalesced_t["Real Throughput (Gb/s)"].min()
    ].to_string()
)
print(
    coalesced_optimized_t[
        coalesced_optimized_t["Theoretical Throughput (Gb/s)"]
        == coalesced_optimized_t["Theoretical Throughput (Gb/s)"].min()
    ].to_string()
)
print(
    coalesced_optimized_t[
        coalesced_optimized_t["Theoretical Throughput (Gb/s)"]
        == coalesced_optimized_t["Theoretical Throughput (Gb/s)"].min()
    ].to_string()
)
