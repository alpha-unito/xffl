import math
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from xffl.utils.utils import get_default_nccl_process_group_options, get_timeout

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""


@dataclass
class DistributedState:
    """This dataclass traces all the distributed environment parameters"""

    # GLOBAL
    # These are global information about the distributed processes (invariant to the other groups)
    rank: Optional[int] = None
    """Global rank"""
    world_size: Optional[int] = None
    """Global world size"""

    # LOCAL GROUP
    # These are local information about the distributed processes on the same node (invariant to the other groups)
    # The current xFFL software assumes homogeneous nodes - no node differences are modelled of specifiable
    group_local_rank: Optional[int] = None
    """Rank of the process inside the local computing node"""
    group_local_size: Optional[int] = None
    """Group size of a computing node"""
    group_rank: Optional[int] = None
    """Rank of the computing node with respect to all the other ones"""
    group_world_size: Optional[int] = None
    """Global number of computing nodes involved in the training process"""

    # REPLICA GROUP
    # These are local information about the distributed processes sharding the same model replica (depend on the federated group)
    # If the FederatedScaling feature is active, then the replica_rank and replica_world_size should be intended referred to the local federated group
    # The current xFFL software assumes homogeneous replicas - each model replica is thus sharded across the same number of nodes
    replica_local_rank: Optional[int] = None
    """Rank of the process inside the local replica sharding group"""
    replica_local_size: Optional[int] = None
    """Group size of a replica sharding group"""
    replica_rank: Optional[int] = None
    """Rank of the replica group with respect to all the other ones (eventually, inside the federated group)"""
    replica_world_size: Optional[int | Tuple[int, ...]] = None
    """Global number of replica sharding groups involved in the training process (eventually, inside the federated groups)"""

    # FEDERATED GROUP
    # These are local information about the distributed processes part of the same federated group (if FederatedScaling is active)
    federated_local_rank: Optional[int] = None
    """Rank of the process inside the local federated group"""
    federated_local_size: Optional[int | Tuple[int, ...]] = None
    """Group size of a federated group (eventually, list of group sizes if the federation is asymmetric)"""
    federated_rank: Optional[int] = None
    """Federated group rank with respect to all the other ones"""
    federated_world_size: Optional[int] = None
    """Global number of federated groups involved in the training process"""

    # MESH
    fsdp_mesh: Optional[DeviceMesh] = None
    """FSDP device mesh"""
    hsdp_mesh: Optional[DeviceMesh] = None
    """HSDP device mesh"""
    federated_group: Optional[ProcessGroup] = None
    """Process group collecting ranks holding the same model's shard across federated groups"""
    replica_group: Optional[ProcessGroup] = None
    """Process group collecting ranks holding the same model's shard inside federated groups"""

    # TECHNICAL
    backend: Optional[Literal["nccl", "gloo", "mpi"]] = None
    """Communication backend"""
    master_addr: Optional[str] = None
    """Rendez-vous address"""
    master_port: Optional[int] = None
    """Rendez-vous port"""
    device: Optional[Literal["cpu", "cuda"]] = None
    """Chosen deployment device"""

    def set_global(self, rank: int, world_size: int) -> None:
        if 0 <= rank < world_size and world_size > 0:
            self.rank = rank
            self.world_size = world_size
        else:
            logger.error(
                f"Impossible setting up a distributed computation with rank {rank} and world size {world_size}"
            )

    def set_technical(
        self,
        backend: Literal["nccl", "gloo", "mpi"],
        master_addr: str,
        master_port: int,
        device: Literal["cpu", "cuda"],
    ) -> None:
        if backend not in ["nccl", "gloo", "mpi"]:
            logger.error(
                f"Impossible setting up distributed environment with backend {backend}"
            )
        elif master_addr is None or master_addr == "":
            logger.error(
                f"Impossible setting up distributed environment with master address {master_addr}"
            )
        elif master_port < 0 or master_port > 65535:
            logger.error(
                f"Impossible setting up distributed environment with master port {master_port}"
            )
        elif device not in ["cpu", "cuda"]:
            logger.error(
                f"Impossible setting up distributed environment with device {device}"
            )
        else:
            self.backend = backend
            self.master_addr = master_addr
            self.master_port = master_port
            self.device = device

    def set_group(
        self,
        group_local_rank: int,
        group_local_size: int,
        group_rank: int,
        group_world_size: int,
    ) -> None:
        if torch.distributed.is_initialized():
            if group_local_rank < 0 or group_local_rank >= group_local_size:
                logger.error(
                    f"Impossible setting up distributed environment on local node with local rank {group_local_rank} and local world size {group_local_size}"
                )
            elif (
                self.world_size % group_local_size != 0
                or self.world_size // group_local_size != group_world_size
            ):
                logger.error(
                    f"Impossible setting up distributed environment on local node with global world size {self.world_size} and local world size {group_local_size}: global world size is not divisible by the local world size into {group_world_size} groups"
                )
            elif group_rank < 0 or group_rank >= group_world_size:
                logger.error(
                    f"Impossible setting up distributed environment on local node with group world size {group_world_size} and group rank {group_rank}"
                )
            else:
                self.group_local_rank = group_local_rank
                self.group_local_size = group_local_size
                self.group_rank = group_rank
                self.group_world_size = group_world_size
        else:
            logger.error(
                f"Impossible setting up local distributed environment configuration: the distributed environment is not initialized"
            )

    def is_group_setup(self) -> bool:
        return (
            self.group_local_rank is not None
            and self.group_local_size is not None
            and self.group_rank is not None
            and self.group_world_size is not None
        )

    def set_fsdp_mesh(self) -> None:
        if torch.distributed.is_initialized():
            self.fsdp_mesh = init_device_mesh(
                device_type=str(self.device),
                mesh_shape=(self.world_size,),
                mesh_dim_names=("shard",),
            )
        else:
            logger.error(
                f"Impossible setting up FSDP: the distributed environment is not initialized"
            )

    def is_fsdp_setup(self) -> bool:
        return self.rank is not None and self.world_size is not None

    def set_hsdp(
        self,
        replica_local_rank: int,
        replica_local_size: int,
        replica_rank: int,
        replica_world_size: int,
    ) -> None:
        """Creates a 2D device mesh allowing Hybrid Sharding Data Parallel (HSDP) training"""
        if torch.distributed.is_initialized():
            world_size: int = (
                self.federated_local_size
                if self.is_federated_scaling_setup()
                else self.world_size
            )
            if replica_local_rank < 0 or replica_local_rank >= replica_local_size:
                logger.error(
                    f"Impossible setting up distributed HSDP environment with local rank {replica_local_rank} and local world size {replica_local_size}"
                )
            elif (
                world_size % replica_local_size != 0
                or world_size // replica_local_size != replica_world_size
            ):
                logger.error(
                    f"Impossible setting up distributed HSDP environment with global world size {world_size} and local world size {replica_local_size}: global world size is not divisible by the local world size into {replica_world_size} replicas"
                )
            elif replica_rank < 0 or replica_rank >= replica_world_size:
                logger.error(
                    f"Impossible setting up distributed HSDP environment with replica world size {replica_world_size} and replica rank {replica_rank}"
                )
            else:
                self.replica_local_rank = replica_local_rank
                self.replica_local_size = replica_local_size
                self.replica_rank = replica_rank
                self.replica_world_size = replica_world_size
        else:
            logger.error(
                f"Impossible setting up local distributed environment configuration: the distributed environment is not initialized"
            )

    def set_hsdp_mesh(self) -> None:
        if torch.distributed.is_initialized():
            self.hsdp_mesh = init_device_mesh(
                device_type=str(self.device),
                mesh_shape=(self.replica_world_size, self.replica_local_size),
                mesh_dim_names=("replica", "shard"),
            )
        else:
            logger.error(
                f"Impossible setting up HSDP: the distributed environment is not initialized"
            )

    def is_hsdp_setup(self) -> bool:
        return (
            self.replica_local_rank is not None
            and self.replica_local_size is not None
            and self.replica_rank is not None
            and self.replica_world_size is not None
        )

    def unset_hsdp(self) -> None:
        self.replica_local_rank = None
        self.replica_local_size = None
        self.replica_rank = None
        self.replica_world_size = None
        self.hsdp_mesh = None

    def set_symmetric_federated_scaling(
        self,
        federated_local_rank: int,
        federated_local_size: int,
        federated_rank: int,
        federated_world_size: int,
    ) -> None:
        """Create the federated scaling rank groups

        This process groups bring together all the ranks handling corresponding model's shards.
        E.g.: if a model is sharded among four processes and replicated across two process groups (i.e., device_mesh=[[0,1,2,3],[4,5,6,7]])
        then the federated scaling process groups correspond to the groups of processes having the same local rank (i.e., [[0,4][1,5][2,6][3,7]])

        """
        if torch.distributed.is_initialized():
            if federated_local_rank < 0 or federated_local_rank >= federated_local_size:
                logger.error(
                    f"Impossible setting up distributed Federated Scaling environment with local rank {federated_local_rank} and local world size {federated_local_size}"
                )
            elif (
                self.world_size % federated_local_size != 0
                or self.world_size // federated_local_size != federated_world_size
            ):
                logger.error(
                    f"Impossible setting up distributed Federated Scaling environment with global world size {self.world_size} and local world size {federated_local_size}: global world size is not divisible by the local world size into {federated_world_size} federated groups"
                )
            elif federated_rank < 0 or federated_rank >= federated_world_size:
                logger.error(
                    f"Impossible setting up distributed Federated Scaling environment with federated world size {federated_world_size} and federated rank {federated_rank}"
                )
            else:
                self.federated_local_rank = federated_local_rank
                self.federated_local_size = federated_local_size
                self.federated_rank = federated_rank
                self.federated_world_size = federated_world_size

                # Check that HSDP and federated configurations are interoperable
                if self.is_hsdp_setup():
                    if self.federated_local_size % self.replica_local_size != 0:
                        logger.error(
                            f"Impossible setting up distributed symmetric HSDP Federated Scaling environment with federated local size {self.federated_local_size} and replica local size {self.replica_local_size} - falling back to FSDP"
                        )
                        self.unset_hsdp()
                    elif self.replica_world_size % self.federated_world_size:
                        logger.error(
                            f"Impossible setting up distributed symmetric HSDP Federated Scaling environment with replica world size {self.replica_world_size} and federated world size {self.federated_world_size} - falling back to FSDP"
                        )
                        self.unset_hsdp()
                    else:
                        self.set_hsdp(
                            replica_local_rank=self.federated_local_rank
                            % self.replica_local_size,
                            replica_local_size=self.replica_local_size,
                            replica_rank=self.federated_local_rank
                            // self.replica_local_size,
                            replica_world_size=self.federated_local_size
                            // self.replica_local_size,
                        )

                # HSDP federation
                if self.is_hsdp_setup():
                    mesh: torch.Tensor = create_device_mesh(
                        mesh_shape=(
                            self.federated_world_size,
                            self.replica_world_size,
                            self.replica_local_size,
                        )
                    )

                    self.hsdp_mesh = DeviceMesh.from_group(
                        device_type=str(self.device),
                        group=[
                            self.create_process_group(
                                ranks=tuple(
                                    mesh[
                                        self.federated_rank,
                                        :,
                                        self.replica_local_rank,
                                    ].tolist()
                                ),
                                group_desc=f"Federated replica group #{self.federated_rank}",  # Share the same model's shard
                            ),
                            self.create_process_group(
                                ranks=tuple(
                                    mesh[
                                        self.federated_rank, self.replica_rank
                                    ].tolist()
                                ),
                                group_desc=f"Federated sharding group #{self.federated_rank}",  # Share all the model's shards
                            ),
                        ],
                        mesh=tuple(mesh[self.federated_rank].tolist()),
                        mesh_dim_names=("replica", "shard"),
                    )

                    # Group of processes sharing the same shard across the federated groups
                    self.federated_group = self.create_process_group(
                        ranks=tuple(
                            mesh[:, self.replica_rank, self.replica_local_rank].tolist()
                        ),
                        group_desc=f"Federated shard averaging group #{self.federated_rank}",
                    )

                    # Group of processes sharing the same shard inside the federated group
                    self.replica_group = self.create_process_group(
                        ranks=tuple(
                            mesh[
                                self.federated_rank, :, self.replica_local_rank
                            ].tolist()
                        ),
                        group_desc=f"Federated replica averaging group #{self.federated_rank}",
                    )
                else:  # FSDP federation
                    mesh: torch.Tensor = create_device_mesh(
                        mesh_shape=(
                            self.federated_world_size,
                            self.federated_local_size,
                        )
                    )

                    self.fsdp_mesh = DeviceMesh.from_group(
                        device_type=str(self.device),
                        group=self.create_process_group(
                            ranks=tuple(mesh[self.federated_rank].tolist()),
                            group_desc=f"Federated sharding group #{self.federated_rank}",
                        ),
                        mesh_dim_names=("shard",),
                    )

                    # Group of processes sharing the same shard across the federated groups
                    self.federated_group = self.create_process_group(
                        ranks=tuple(mesh[:, self.federated_local_rank].tolist()),
                        group_desc=f"Federated shard averaging group #{self.federated_rank}",
                    )
        else:
            logger.error(
                f"Impossible setting up local distributed environment configuration: the distributed environment is not initialized"
            )

    def set_asymmetric_federated_scaling(
        self,
        federated_local_rank: int,
        federated_local_size: Tuple[int],
        federated_rank: int,
        federated_world_size: int,
    ) -> None:
        """Create the federated scaling rank groups

        This process groups bring together all the ranks handling corresponding model's shards.
        E.g.: if a model is sharded among four processes and replicated across two process groups (i.e., device_mesh=[[0,1,2,3],[4,5,6,7]])
        then the federated scaling process groups correspond to the groups of processes having the same local rank (i.e., [[0,4][1,5][2,6][3,7]])

        """
        if torch.distributed.is_initialized():
            if (
                federated_local_rank < 0
                or federated_local_rank >= federated_local_size[federated_rank]
            ):
                logger.error(
                    f"Impossible setting up distributed Federated Scaling environment with local rank {federated_local_rank} and local world size {federated_local_size[federated_rank]}"
                )
            elif (
                self.world_size != sum(federated_local_size)
                or len(federated_local_size) != federated_world_size
            ):
                logger.error(
                    f"Impossible setting up distributed Federated Scaling environment with global world size {self.world_size} and local world sizes {federated_local_size}: global world size is not divisible by the local world size into {federated_world_size} federated groups"
                )
            elif federated_rank < 0 or federated_rank >= federated_world_size:
                logger.error(
                    f"Impossible setting up distributed Federated Scaling environment with federated world size {federated_world_size} and federated rank {federated_rank}"
                )
            else:
                self.federated_local_rank = federated_local_rank
                self.federated_local_size = federated_local_size
                self.federated_rank = federated_rank
                self.federated_world_size = federated_world_size

                # Check that HSDP and federated configurations are interoperable
                if self.is_hsdp_setup():
                    if not all(
                        [
                            federated_local_size % self.replica_local_size == 0
                            for federated_local_size in self.federated_local_size
                        ]
                    ):
                        logger.error(
                            f"Impossible setting up distributed asymmetric HSDP Federated Scaling environment with federated local sizes {self.federated_local_size} and replica local size {self.replica_local_size} - falling back to HSDP"
                        )
                        self.unset_federated_scaling()
                    elif self.replica_world_size != sum(
                        [
                            federated_local_size // self.replica_local_size
                            for federated_local_size in self.federated_local_size
                        ]
                    ):
                        logger.error(
                            f"Impossible setting up distributed asymmetric HSDP Federated Scaling environment with replica world size {self.replica_world_size} and federated local world sizes {self.federated_local_size} - falling back to HSDP"
                        )
                        self.unset_federated_scaling()
                    else:  # Federated local world size update
                        replica_world_size: List[int] = []
                        for rank in range(self.federated_world_size):
                            replica_world_size.append(
                                self.federated_local_size[rank]
                                // self.replica_local_size
                            )
                            if (
                                sum(replica_world_size[:-1])
                                <= self.replica_rank
                                < sum(replica_world_size)
                            ):
                                self.replica_rank = self.replica_rank - sum(
                                    replica_world_size[:-1]
                                )
                        self.replica_world_size = tuple(replica_world_size)

                # HSDP asymmetric federation
                if self.is_federated_scaling_setup():
                    mesh: List[torch.Tensor] = []
                    offset: int = 0
                    for rank in range(self.federated_world_size):
                        mesh.append(
                            create_device_mesh(
                                mesh_shape=(
                                    self.replica_world_size[rank],
                                    self.replica_local_size,
                                )
                            )
                            + offset
                        )
                        offset += self.federated_local_size[rank]

                    self.hsdp_mesh = DeviceMesh.from_group(
                        device_type=str(self.device),
                        group=[
                            self.create_process_group(
                                ranks=tuple(
                                    mesh[self.federated_rank][
                                        :,
                                        self.replica_local_rank,
                                    ].tolist()
                                ),
                                group_desc=f"Federated replica group #{self.federated_rank}",  # Share the same model's shard
                            ),
                            self.create_process_group(
                                ranks=tuple(
                                    mesh[self.federated_rank][
                                        self.replica_rank
                                    ].tolist()
                                ),
                                group_desc=f"Federated sharding group #{self.federated_rank}",  # Share all the model's shards
                            ),
                        ],
                        mesh=tuple(mesh[self.federated_rank].tolist()),
                        mesh_dim_names=("replica", "shard"),
                    )

                    # Group of processes sharing the same shard across the federated groups
                    self.federated_group = self.create_process_group(
                        ranks=tuple(
                            [
                                int(
                                    federated_mesh[
                                        self.replica_rank, self.replica_local_rank
                                    ]
                                )
                                for federated_mesh in mesh
                                if self.replica_rank < len(federated_mesh)
                            ]
                        ),
                        group_desc=f"Federated shard averaging group #{self.federated_rank}",
                    )

                    # Group of processes sharing the same shard inside the federated group
                    self.replica_group = self.create_process_group(
                        ranks=tuple(
                            mesh[self.federated_rank][
                                :, self.replica_local_rank
                            ].tolist()
                        ),
                        group_desc=f"Federated replica averaging group #{self.federated_rank}",
                    )
        else:
            logger.error(
                f"Impossible setting up local distributed environment configuration: the distributed environment is not initialized"
            )

    def unset_federated_scaling(self) -> None:
        self.federated_local_rank = None
        self.federated_local_size = None
        self.federated_rank = None
        self.federated_world_size = None

    def is_federated_scaling_setup(self) -> bool:
        return (
            self.federated_local_rank is not None
            and self.federated_local_size is not None
            and self.federated_rank is not None
            and self.federated_world_size is not None
        )

    def create_process_group(
        self,
        ranks: Tuple[int, ...] | torch.Tensor,
        group_desc: Optional[str],
    ) -> ProcessGroup:
        """Creates a new process group with the specified ranks

        Only the interested rank can enter this method

        :param ranks: Ranks making up the group
        :type ranks: Tuple[int, ...]
        :param group_desc: Description of the process group
        :type group_desc: Optional[str]
        :return: Process group handle
        :rtype: ProcessGroup
        """
        return dist.new_group(
            ranks=ranks,
            timeout=get_timeout(),
            backend=self.backend,
            pg_options=get_default_nccl_process_group_options(),
            use_local_synchronization=True,
            group_desc=group_desc,
        )


def create_device_mesh(mesh_shape: Tuple[int, ...]) -> torch.Tensor:
    """Creates a Tensor of distributed process ranks with the specified dimensions

    :param mesh_shape: Dimensions of the mesh
    :type mesh_shape: Tuple[int, ...]
    :return: Tensor of ranks
    :rtype: torch.Tensor
    """
    return torch.arange(math.prod(mesh_shape), dtype=torch.int).view(mesh_shape)
