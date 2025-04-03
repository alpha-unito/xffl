"""Utilities for the Federated Scaling feature"""

from typing import List, Optional, Tuple


def get_cells_ids(
    nodes: List[str], cell_size: int
) -> Tuple[List[int], List[int]]:  # TODO: this method suppose sorted node list
    """Calculates an incremental cell ID for each node participating in the training

    :param nodes: List of nodes assigned to the training
    :type nodes: List[str]
    :param cell_size: Size (number of nodes) of the cell
    :type cell_size: int
    :return: List of cell IDs for each node
    :rtype: List[int]
    """
    nodes: List[int] = [
        int("".join(filter(str.isdigit, node))) // cell_size for node in nodes
    ]

    cells_ids: List[Optional[int]] = []
    local_world_sizes: List[Optional[int]] = []

    if len(set(nodes)) <= 1:
        cells_ids = [None for _ in nodes]
    else:
        cell_id: int = 0
        current_cell_id: int = nodes[0]
        world_size: int = 0
        for node in nodes:
            if node != current_cell_id:
                local_world_sizes.append(world_size)
                cell_id += 1
                current_cell_id = node
                world_size = 1
            else:
                world_size += 1
            cells_ids.append(cell_id)
        local_world_sizes.append(world_size)

    return cells_ids, local_world_sizes


print(
    str(
        get_cells_ids(
            ["lrdn0001", "lrdn001", "lrdn0585", "lrdn0939", "lrdn0946"], cell_size=180
        )[1]
    )
    .replace("]", "")
    .replace("[", "")
)
