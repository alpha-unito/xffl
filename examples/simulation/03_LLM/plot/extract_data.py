import pandas as pd

logs = "/leonardo_scratch/fast/uToID_bench/xffl/examples/simulation/03_LLM/logs"

model = "llama3.1-8b"
fs = 1

data_list = []
for nodes in [1, 2, 4, 8, 16, 32, 64, 128]:
    if nodes == 1:
        for processes_per_node in [2, 4]:
            data = pd.read_csv(
                filepath_or_buffer=f"{logs}/{model}_ns_{nodes}_fs_{fs}_ppn_{processes_per_node}.csv",
                sep=",",
                dtype={
                    "Aggregation Strategy": str,
                    "NCCL Algo": str,
                    "NCLL Proto": str,
                    "Multiple CUDA Streams": bool,
                    "Contiguous Memory": bool,
                    "Average Time (s)": float,
                    "Theoretical Throughput (Gb/s)": float,
                    "Real Throughput (Gb/s)": float,
                    "Max GPU RAM Allocated (GB)": float,
                },
                engine="pyarrow",
            )

            for strategy in [
                "layer_by_layer",
                "layer_by_layer_optimized",
                "bucket_flatten",
                "bucket_optimized_flatten",
                "bucket_coalesced",
                "bucket_optimized_coalesced",
            ]:
                print(
                    f"---------------- {strategy} - {nodes} - {processes_per_node} --------------"
                )
                print(
                    f"{data[data['Aggregation Strategy'] == strategy][f'Average Time (s)'].mean():.2f}"
                    + "{\\tiny$\pm$"
                    + f"{data[data['Aggregation Strategy'] == strategy][f'Average Time (s)'].std():.2f}"
                    + "}"
                )
                print("\n")

            data_list.append(data)
    else:
        processes_per_node = 1
        data = pd.read_csv(
            filepath_or_buffer=f"{logs}/{model}_ns_{nodes}_fs_{fs}_ppn_{processes_per_node}.csv",
            sep=",",
            dtype={
                "Aggregation Strategy": str,
                "NCCL Algo": str,
                "NCLL Proto": str,
                "Multiple CUDA Streams": bool,
                "Contiguous Memory": bool,
                "Average Time (s)": float,
                "Theoretical Throughput (Gb/s)": float,
                "Real Throughput (Gb/s)": float,
                "Max GPU RAM Allocated (GB)": float,
            },
            engine="pyarrow",
        )

        for strategy in [
            "layer_by_layer",
            "layer_by_layer_optimized",
            "bucket_flatten",
            "bucket_optimized_flatten",
            "bucket_coalesced",
            "bucket_optimized_coalesced",
        ]:
            print(
                f"---------------- {strategy} - {nodes} - {processes_per_node} --------------"
            )
            print(
                f"{data[data['Aggregation Strategy'] == strategy][f'Average Time (s)'].mean():.2f}"
                + "{\\tiny$\pm$"
                + f"{data[data['Aggregation Strategy'] == strategy][f'Average Time (s)'].std():.2f}"
                + "}"
            )
            print("\n")

            data_list.append(data)

data = pd.concat([dataframe for dataframe in data_list])

for strategy in [
    "layer_by_layer",
    "layer_by_layer_optimized",
    "bucket_flatten",
    "bucket_optimized_flatten",
    "bucket_coalesced",
    "bucket_optimized_coalesced",
]:
    print(f"---------------- {strategy} - Maximum memory occupancy --------------")
    print(
        f"{data[data['Aggregation Strategy'] == strategy]['Max GPU RAM Allocated (GB)'].max():.2f}"
    )
    print("\n")
