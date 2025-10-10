import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

sns.set_theme(style="darkgrid")

plt.rc("xtick", labelsize=15)
plt.rc("ytick", labelsize=16)

logs = "/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/aggregation_logs"

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
yticks_2 = [450, 500, 550, 600, 650, 700, 750, 800]
yticks_4 = [1200, 1400, 1600, 1800, 2000, 2200, 2400]
y_lim_2 = (445, 665)  # (425, 675)
y_lim_4 = (1180, 1820)  # (1150, 1850)
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

# divider1 = make_axes_locatable(ax[0][0])
# ax1 = divider1.new_vertical(size="20%", pad=0.1)
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
# fig.add_axes(ax1)
ax[0][0].set_ylim((425, 675))
ax[0][0].spines["top"].set_visible(False)
# ax1.set_ylim((790, 810))
# ax1.set_yticks((800,))
# ax1.set_xticks(ticks=[0, 1, 2, 3, 4, 5], minor=True, labels=[])
# ax1.set_xticklabels([])
# ax1.set_xlim((0, 6))
# ax1.axhline(y=800, color="r", linestyle="--", label="Maximum theoretical bandwidth")
# ax1.spines["bottom"].set_visible(False)
fig1.set_title("Without memory overheads", fontsize=16)

fig1.set(
    # title="Without memory overheads",
    ylabel="2 GPUs",
    # yticks=yticks_2,
    # ylim=y_lim_2,
    xticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    xticklabels=[],
)
fig1.set_ylabel("2 GPUs", fontsize=16)
fig1.set_xticks(ticks=[0, 1, 2, 3, 4, 5], minor=True, labels=labels)
fig1.grid(True, axis="both")

# # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
# d = 0.015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax1.transAxes, color="k", clip_on=False)
# ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
# ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax[0][0].transAxes)  # switch to the bottom axes
# ax[0][0].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax[0][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


# ax[0][1].axhline(
#     y=800, color="r", linestyle="--", label="Maximum theoretical bandwidth"
# )
# ax[0][1].set_ylim(780, 810)
# ax[0][1].set_yticks([800])
# ax[0][1].set_aspect(0.2)

# divider2 = make_axes_locatable(ax[0][1])
# ax2 = divider2.new_vertical(size="20%", pad=0.1)
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
    # title="With memory overheads",
    yticklabels=[],
    ylim=y_lim_2,
)
# fig2.axhline(y=800, color="r", linestyle="--", label="Maximum theoretical bandwidth")
# fig2.set_title("With memory overheads", fontsize=14)
fig2.grid(True, axis="both")

# fig.add_axes(ax2)
ax[0][1].set_ylim((425, 675))
ax[0][1].spines["top"].set_visible(False)
# ax2.set_ylim((790, 810))
# ax2.set_yticks([])
# ax2.set_yticklabels([])
# ax2.set_yticks((800,))
# ax2.set_xticks(ticks=[0, 1, 2, 3, 4, 5], minor=True, labels=[])
# ax2.set_xticklabels([])
# ax2.set_xlim((0, 6))
# ax2.axhline(y=800, color="r", linestyle="--", label="Maximum theoretical bandwidth")
# ax2.spines["bottom"].set_visible(False)
fig2.set_title("With memory overheads", fontsize=16)


# # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
# d = 0.015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax2.transAxes, color="k", clip_on=False)
# ax2.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
# ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax[0][1].transAxes)  # switch to the bottom axes
# ax[0][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax[0][1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


# divider3 = make_axes_locatable(ax[1][0])
# ax3 = divider3.new_vertical(size="20%", pad=0.1)
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
    # ylabel="4 GPUs",
    yticks=yticks_4,
    ylim=y_lim_4,
)
# fig3.axhline(y=2400, color="r", linestyle="--", label="Maximum theoretical bandwidth")
fig3.set_ylabel("4 GPUs", fontsize=16)
fig3.tick_params(axis="x", labelrotation=45, which="minor")
fig3.grid(True, axis="both")

# fig.add_axes(ax3)
ax[1][0].set_ylim((1180, 1820))
ax[1][0].spines["top"].set_visible(False)
# ax3.set_ylim((2380, 2420))
# ax3.set_yticks((2400,))
# ax3.set_xticks(ticks=[0, 1, 2, 3, 4, 5], minor=True, labels=[])
# ax3.set_xticklabels([])
# ax3.set_xlim((0, 6))
# ax3.axhline(y=2400, color="r", linestyle="--", label="Maximum theoretical bandwidth")
# ax3.spines["bottom"].set_visible(False)

# # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
# d = 0.015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax3.transAxes, color="k", clip_on=False)
# ax3.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
# ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax[1][0].transAxes)  # switch to the bottom axes
# ax[1][0].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax[1][0].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


# divider4 = make_axes_locatable(ax[1][1])
# ax4 = divider4.new_vertical(size="20%", pad=0.1)
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
# fig4.axhline(y=2400, color="r", linestyle="--", label="Maximum theoretical bandwidth")
fig4.tick_params(axis="x", labelrotation=45, which="minor")
fig4.grid(True, axis="both")

# fig.add_axes(ax4)
ax[1][1].set_ylim((1180, 1820))
ax[1][1].spines["top"].set_visible(False)
# ax4.set_ylim((2380, 2420))
# ax4.set_yticks([])
# ax4.set_yticklabels([])
# ax4.set_xticks(ticks=[0, 1, 2, 3, 4, 5], minor=True, labels=[])
# ax4.set_xticklabels([])
# ax4.set_xlim((0, 6))
# ax4.axhline(y=2400, color="r", linestyle="--", label="Maximum theoretical bandwidth")
# ax4.spines["bottom"].set_visible(False)

# # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
# d = 0.015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax4.transAxes, color="k", clip_on=False)
# ax4.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
# ax4.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax[1][1].transAxes)  # switch to the bottom axes
# ax[1][1].plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax[1][1].plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


# fig.suptitle("Single-node FedAVG performance", fontsize=16, x=0.5, y=1.01)
# fig.supxlabel("FedAvg implementation", fontsize=14, x=0.5)
fig.supylabel("Throughput (Gb/s)", fontsize=18, y=0.54, x=0)

fig1.legend(
    title=None,
    loc="upper center",
    handles=fig1.get_legend_handles_labels()[0],
    labels=["No opt.", "Contig. Mem.", "Mult. Streams", "Both"],
    ncol=4,
    bbox_to_anchor=(0.92, 1.28),
    fontsize=18,
)

plt.subplots_adjust(hspace=0.07, wspace=0.02, bottom=0.15, top=0.9)
fig.figure.savefig("out.png", bbox_inches="tight", dpi=300, format="png")
fig.figure.savefig("single-node.pdf", bbox_inches="tight", dpi=300, format="pdf")

quit()

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
