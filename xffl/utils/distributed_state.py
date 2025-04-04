import math
from dataclasses import dataclass
from logging import Logger, getLogger
from typing import List, Literal, Optional, Tuple

import torch
import torch.cuda as cuda
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

    # LOCAL NODE
    # These are local information about the distributed processes on the same node (invariant to the other groups)
    # The current xFFL software assumes homogeneous nodes - no node differences are modelled of specifiable
    node_local_rank: Optional[int] = None
    """Rank of the process inside the local computing node"""
    node_local_size: Optional[int] = None
    """node size of a computing node"""
    node_rank: Optional[int] = None
    """Rank of the computing node with respect to all the other ones"""
    node_world_size: Optional[int] = None
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
    replica_world_size: Optional[Tuple[int, ...]] = None
    """Global number of replica sharding groups involved in the training process (eventually, inside the federated groups)"""

    # FEDERATED GROUP
    # These are local information about the distributed processes part of the same federated group (if FederatedScaling is active)
    federated_local_rank: Optional[int] = None
    """Rank of the process inside the local federated group"""
    federated_local_size: Optional[Tuple[int, ...]] = None
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
    federated_group: Optional[Tuple[ProcessGroup, ...]] = None
    """Process group collecting ranks holding the same model's shard across federated groups"""
    replica_group: Optional[Tuple[ProcessGroup, ...]] = None
    """Process group collecting ranks holding the same model's shard inside federated groups"""
    federation: Optional[ProcessGroup] = None
    """Process group collecting all ranks participating in the same federated group"""

    # TECHNICAL
    backend: Optional[Literal["nccl", "gloo", "mpi"]] = None
    """Communication backend"""
    master_addr: Optional[str] = None
    """Rendez-vous address"""
    master_port: Optional[int] = None
    """Rendez-vous port"""
    device: Optional[Literal["cpu", "cuda"]] = None
    """Chosen deployment device"""
    streams: Optional[Tuple[cuda.Stream, ...]] = None
    """Pool of available CUDA streams"""

    def __str__(self):
        federated_group: Optional[List[int]] = (
            dist.get_process_group_ranks(self.federated_group[0])
            if self.federated_group is not None
            else None
        )
        replica_group: Optional[List[int]] = (
            dist.get_process_group_ranks(self.replica_group[0])
            if self.replica_group is not None
            else None
        )
        federation: Optional[List[int]] = (
            dist.get_process_group_ranks(self.federation)
            if self.federation is not None
            else None
        )
        return f"""\n
                GLOBAL:
                    Rank={self.rank}
                    World size={self.world_size}
                NODE:
                    Node local rank={self.node_local_rank}
                    Node local size={self.node_local_size}
                    Node rank={self.node_rank}
                    Node world size={self.node_world_size}
                REPLICA:
                    Replica local rank={self.replica_local_rank}
                    Replica local size={self.replica_local_size}
                    Replica rank={self.replica_rank}
                    Replica world size={self.replica_world_size}
                FEDERATION:
                    Federated local rank={self.federated_local_rank}
                    Federated local size={self.federated_local_size}
                    Federated rank={self.federated_rank}
                    Federated world size={self.federated_world_size}
                MESHES:
                    FSDP={self.fsdp_mesh}
                    HSDP={self.hsdp_mesh}
                    Federated group={federated_group}
                    Replica group={replica_group}
                    Federation={federation}
                TECHNICAL:
                    Backend={self.backend}
                    Master address={self.master_addr}
                    Master port={self.master_port}
                    Device={self.device}
                    Streams={self.streams}
                """

    ### Methods ###

    def set_global(self, rank: int, world_size: int) -> None:
        """
        Set global process group information.

        :param rank: Global process rank
        :type rank: int
        :param world_size: Global world size
        :type world_size: int
        """
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
        streams: int = 4,
    ) -> None:
        """
        Set the technical properties of the distributed process group.

        :param backend: Communication backend to use
        :type backend: Literal["nccl", "gloo", "mpi"]
        :param master_addr: Address of the master node for the rendez-vous
        :type master_addr: str
        :param master_port: Port of the master node for the rendez-vous
        :type master_port: int
        :param device: Type of device to use
        :type device: Literal["cpu", "cuda"]
        :param streams: Number of CUDA streams to instantiate, defaults to 4
        :type streams: int
        """
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
            self.streams = tuple(cuda.Stream() for _ in range(streams))

    def set_node(
        self,
        node_local_rank: int,
        node_local_size: int,
        node_rank: int,
        node_world_size: int,
    ) -> None:
        """
        Set the process' information relative to the local node.

        :param node_local_rank: Local compute node rank
        :type node_local_rank: int
        :param node_local_size: World size of the local compute node
        :type node_local_size: int
        :param node_rank: Rank of the local compute node among all the available nodes in the training
        :type node_rank: int
        :param node_world_size: Number of compute nodes involved in the training process
        :type node_world_size: int
        """
        if torch.distributed.is_initialized():
            if node_local_rank < 0 or node_local_rank >= node_local_size:
                logger.error(
                    f"Impossible setting up distributed environment on local node with local rank {node_local_rank} and local world size {node_local_size}"
                )
            elif (
                self.world_size % node_local_size != 0
                or self.world_size // node_local_size != node_world_size
            ):
                logger.error(
                    f"Impossible setting up distributed environment on local node with global world size {self.world_size} and local world size {node_local_size}: global world size is not divisible by the local world size into {node_world_size} nodes"
                )
            elif node_rank < 0 or node_rank >= node_world_size:
                logger.error(
                    f"Impossible setting up distributed environment on local node with node world size {node_world_size} and node rank {node_rank}"
                )
            else:
                self.node_local_rank = node_local_rank
                self.node_local_size = node_local_size
                self.node_rank = node_rank
                self.node_world_size = node_world_size
        else:
            logger.error(
                f"Impossible setting up local distributed environment configuration: the distributed environment is not initialized"
            )

    def is_node_setup(self) -> bool:
        """
        Checks if the local compute node information is set up.

        :return: True if the local compute node information is set up, False otherwise
        :rtype: bool
        """
        return (
            self.node_local_rank is not None
            and self.node_local_size is not None
            and self.node_rank is not None
            and self.node_world_size is not None
        )

    def _get_global_fsdp_mesh(self) -> Optional[DeviceMesh]:
        """
        Returns a standard global FSDP device mesh.
        Do not call this method if global FSDP is not required.

        :return: A global FSDP device mesh if the distributed PyTorch environment is initialized, None otherwise
        :rtype: Optional[DeviceMesh]
        """
        mesh: Optional[DeviceMesh] = None
        if torch.distributed.is_initialized():
            mesh = init_device_mesh(
                device_type=str(self.device),
                mesh_shape=(self.world_size,),
                mesh_dim_names=("shard",),
            )
        else:
            logger.error(
                f"Impossible setting up FSDP: the distributed environment is not initialized"
            )

        return mesh

    def set_fsdp(self, mesh: Optional[DeviceMesh] = None) -> None:
        """
        Enable PyTorch's FSDP functionality.
        If no mesh specified, FSDP will be enabled on the global process group.

        :param mesh: An FSDP device mesh, defaults to None
        :type mesh: Optional[DeviceMesh]
        """
        self.fsdp_mesh = self._get_global_fsdp_mesh() if mesh is None else mesh

    def is_fsdp_setup(self) -> bool:
        """
        Checks if FSDP is set up.

        :return: True if FSDP is set up, False otherwise
        :rtype: bool
        """
        return self.fsdp_mesh is not None

    def _get_global_hsdp_mesh(self) -> Optional[DeviceMesh]:
        """
        Returns a global HSD device mesh.
        Do not call this method if global HSD device is not required.

        :return: A global HSDP device mesh if the distributed PyTorch environment is initialized, None otherwise
        :rtype: Optional[DeviceMesh]
        """
        mesh: Optional[DeviceMesh] = None
        if torch.distributed.is_initialized():
            self.hsdp_mesh = init_device_mesh(
                device_type=str(self.device),
                mesh_shape=(self.replica_world_size[0], self.replica_local_size),
                mesh_dim_names=("replica", "shard"),
            )
        else:
            logger.error(
                f"Impossible setting up HSDP: the distributed environment is not initialized"
            )
        return mesh

    def set_hsdp(self, hsdp: int) -> None:
        """
        Enable global PyTorch's HSDP functionality.

        :param hsdp: Size of an HSDP replica
        :type hsdp: int
        """
        self._partial_hsdp_setup(hsdp=hsdp)
        self.hsdp_mesh = self._get_global_hsdp_mesh()

    def _partial_hsdp_setup(self, hsdp: int) -> None:
        """
        Initialize PyTorch's HSDP parameters without creating the device mesh.

        :param hsdp: Size of an HSDP replica
        :type hsdp: int
        """
        self._partial_hsdp_setup_manual(
            replica_local_rank=self.rank % hsdp,
            replica_local_size=hsdp,
            replica_rank=self.rank // hsdp,
            replica_world_size=(self.world_size // hsdp,),
        )

    def _partial_hsdp_setup_manual(
        self,
        replica_local_rank: int,
        replica_local_size: int,
        replica_rank: int,
        replica_world_size: Tuple[int, ...],
    ) -> None:
        """
        Partial set up of PyTorch's HSDP functionality; to complete it is necessary to instantiate also the HSDP device mesh.

        :param replica_local_rank: Rank of the current process within its model replica
        :type replica_local_rank: int
        :param replica_local_size: Local world size of a model replica
        :type replica_local_size: int
        :param replica_rank: Rank of the current model replica among the current federated group replica world size
        :type replica_rank: int
        :param replica_world_size: Number of replicas available for each federated group
        :type replica_world_size: Tuple[int,...]
        """
        if torch.distributed.is_initialized():

            _world_size: int
            _replica_world_size: int
            if self.is_federated_scaling_setup():
                _world_size = self.federated_local_size[self.federated_rank]
                _replica_world_size = replica_world_size[self.federated_rank]
            else:
                _world_size = self.world_size
                _replica_world_size = replica_world_size[0]

            if replica_local_rank < 0 or replica_local_rank >= replica_local_size:
                logger.error(
                    f"Impossible setting up distributed HSDP environment with local rank {replica_local_rank} and local world size {replica_local_size}"
                )
            elif (
                _world_size % replica_local_size != 0
                or _world_size // replica_local_size != _replica_world_size
            ):
                logger.error(
                    f"Impossible setting up distributed HSDP environment with global world size {_world_size} and local world size {replica_local_size}: global world size is not divisible by the local world size into {_replica_world_size} replicas"
                )
            elif replica_rank < 0 or replica_rank >= _replica_world_size:
                logger.error(
                    f"Impossible setting up distributed HSDP environment with replica world size {_replica_world_size} and replica rank {replica_rank}"
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

    def is_hsdp_setup(self) -> bool:
        """
        Checks if HSDP is set up.
        Does not check the HSDP device mesh.

        :return: True if HSDP is set up, False otherwise
        :rtype: bool
        """
        return (
            self.replica_local_rank is not None
            and self.replica_local_size is not None
            and self.replica_rank is not None
            and self.replica_world_size is not None
        )

    def unset_hsdp(self) -> None:
        """
        Unsets all HSDP related variables.
        """
        self.replica_local_rank = None
        self.replica_local_size = None
        self.replica_rank = None
        self.replica_world_size = None
        self.hsdp_mesh = None

    def set_federated_scaling(
        self, federated_group_size: Tuple[int], hsdp: Optional[int] = None
    ) -> None:
        if hsdp is not None:  # Setting HSDP if needed
            self._partial_hsdp_setup(hsdp=hsdp)

        if len(set(federated_group_size)) == 1:
            logger.debug(
                f"Setting Symmetric Federated Scaling with sizes {federated_group_size}"
            )
            self._set_symmetric_federated_scaling(
                federated_group_size=federated_group_size
            )
        else:
            logger.debug(
                f"Setting Asymmetric Federated Scaling with sizes {federated_group_size}"
            )
            self._set_asymmetric_federated_scaling(
                federated_group_size=federated_group_size
            )

    def _set_symmetric_federated_scaling(
        self, federated_group_size: Tuple[int]
    ) -> None:
        """
        Create the federated scaling process groups

        :param federated_group_size: Number of processes making up one federated group
        :type federated_group_size: int
        """
        if torch.distributed.is_initialized():

            federated_local_rank: int = self.rank % federated_group_size[0]
            federated_local_size: Tuple[int, ...] = federated_group_size
            federated_rank: int = self.rank // federated_group_size[0]
            federated_world_size: int = self.world_size // federated_group_size[0]

            _federated_local_size: int = federated_local_size[federated_rank]

            if (
                federated_local_rank < 0
                or federated_local_rank >= _federated_local_size
            ):
                logger.error(
                    f"Impossible setting up distributed Federated Scaling environment with local rank {federated_local_rank} and local world size {_federated_local_size}"
                )
            elif (
                self.world_size % _federated_local_size != 0
                or self.world_size // _federated_local_size != federated_world_size
            ):
                logger.error(
                    f"Impossible setting up distributed Federated Scaling environment with global world size {self.world_size} and local world size {_federated_local_size}: global world size is not divisible by the local world size into {federated_world_size} federated groups"
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
                    if (
                        self.federated_local_size[self.federated_rank]
                        % self.replica_local_size
                        != 0
                    ):
                        logger.error(
                            f"Impossible setting up distributed symmetric HSDP Federated Scaling environment with federated local size {self.federated_local_size[self.federated_rank]} and replica local size {self.replica_local_size} - falling back to FSDP"
                        )
                        self.unset_hsdp()
                    elif (
                        self.replica_world_size[0]
                        % self.federated_world_size  # The replica_world_size is still not divided among the federated groups
                    ):
                        logger.error(
                            f"Impossible setting up distributed symmetric HSDP Federated Scaling environment with replica world size {self.replica_world_size} and federated world size {self.federated_world_size} - falling back to FSDP"
                        )
                        self.unset_hsdp()
                    else:
                        self._partial_hsdp_setup_manual(
                            replica_local_rank=self.federated_local_rank
                            % self.replica_local_size,
                            replica_local_size=self.replica_local_size,
                            replica_rank=self.federated_local_rank
                            // self.replica_local_size,
                            replica_world_size=tuple(
                                self.federated_local_size[federated_rank]
                                // self.replica_local_size
                                for federated_rank in range(self.federated_world_size)
                            ),
                        )

                if self.is_hsdp_setup():  # HSDP federation
                    mesh: torch.Tensor = create_device_mesh(
                        mesh_shape=(
                            self.federated_world_size,
                            self.replica_world_size[self.federated_rank],
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
                    self.federated_group = tuple(
                        self.create_process_group(
                            ranks=tuple(
                                mesh[
                                    :, self.replica_rank, self.replica_local_rank
                                ].tolist()
                            ),
                            group_desc=f"Federated shard averaging group #{self.federated_rank} instance #{i}",
                        )
                        for i in range(
                            len(self.streams)
                        )  # Multiple ProcessGroup handles are needed to communicate with multiple Streams
                    )

                    # Group of processes sharing the same shard inside the federated group
                    self.replica_group = tuple(
                        self.create_process_group(
                            ranks=tuple(
                                mesh[
                                    self.federated_rank, :, self.replica_local_rank
                                ].tolist()
                            ),
                            group_desc=f"Federated replica averaging group #{self.federated_rank} instance #{i}",
                        )
                        for i in range(
                            len(self.streams)
                        )  # Multiple ProcessGroup handles are needed to communicate with multiple Streams
                    )

                else:  # FSDP federation
                    mesh: torch.Tensor = create_device_mesh(
                        mesh_shape=(
                            self.federated_world_size,
                            self.federated_local_size[self.federated_rank],
                        )
                    )

                    self.set_fsdp(
                        DeviceMesh.from_group(
                            device_type=str(self.device),
                            group=self.create_process_group(
                                ranks=tuple(mesh[self.federated_rank].tolist()),
                                group_desc=f"Federated sharding group #{self.federated_rank}",
                            ),
                            mesh_dim_names=("shard",),
                        )
                    )

                    # Group of processes sharing the same shard across the federated groups
                    self.federated_group = tuple(
                        self.create_process_group(
                            ranks=tuple(mesh[:, self.federated_local_rank].tolist()),
                            group_desc=f"Federated shard averaging group #{self.federated_rank} instance #{i}",
                        )
                        for i in range(
                            len(self.streams)
                        )  # Multiple ProcessGroup handles are needed to communicate with multiple Streams
                    )

            # Group of processes participating in the same federated group
            self.federation = self.create_process_group(
                ranks=tuple(
                    [
                        rank
                        for rank in range(
                            self.federated_local_size[federated_rank]
                            * self.federated_rank,
                            self.federated_local_size[federated_rank]
                            * (self.federated_rank + 1),
                        )
                    ]
                ),
                group_desc=f"Federated group #{self.federated_rank}",
            )
        else:
            logger.error(
                f"Impossible setting up local distributed environment configuration: the distributed environment is not initialized"
            )

    def _set_asymmetric_federated_scaling(
        self, federated_group_size: Tuple[int]
    ) -> None:
        """
        Create the federated scaling process groups

        This process groups bring together all the ranks handling corresponding model's shards.
        E.g.: if a model is sharded among four processes and replicated across two process groups (i.e., device_mesh=[[0,1,2,3],[4,5,6,7]])
        then the federated scaling process groups correspond to the groups of processes having the same local rank (i.e., [[0,4][1,5][2,6][3,7]])

        :param federated_group_size: Number of processes making up one federated group
        :type federated_group_size: int
        """
        if torch.distributed.is_initialized():

            _federated_rank: int = -1
            index: int = 0
            while _federated_rank < 0:
                if self.rank < sum(federated_group_size[: index + 1]):
                    _federated_rank = index
                index += 1

            federated_local_rank = self.rank - sum(
                federated_group_size[:_federated_rank]
            )
            federated_local_size = federated_group_size
            federated_rank = _federated_rank
            federated_world_size = len(federated_group_size)

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
                    elif sum(self.replica_world_size) != sum(
                        [
                            federated_local_size // self.replica_local_size
                            for federated_local_size in self.federated_local_size
                        ]
                    ):
                        logger.error(
                            f"Impossible setting up distributed asymmetric HSDP Federated Scaling environment with replica local size {self.replica_local_size} and federated local world sizes {self.federated_local_size} - falling back to HSDP"
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
                    self.federated_group = tuple(
                        self.create_process_group(
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
                            group_desc=f"Federated shard averaging group #{self.federated_rank} instance #{i}",
                        )
                        for i in range(len(self.streams))
                    )

                    # Group of processes sharing the same shard inside the federated group
                    self.replica_group = tuple(
                        self.create_process_group(
                            ranks=tuple(
                                mesh[self.federated_rank][
                                    :, self.replica_local_rank
                                ].tolist()
                            ),
                            group_desc=f"Federated replica averaging group #{self.federated_rank} instance #{i}",
                        )
                        for i in range(len(self.streams))
                    )

                    self.federation = self.create_process_group(
                        ranks=tuple(
                            [
                                rank
                                for rank in range(
                                    sum(
                                        self.federated_local_size[: self.federated_rank]
                                    ),
                                    sum(
                                        self.federated_local_size[
                                            : self.federated_rank + 1
                                        ]
                                    ),
                                )
                            ]
                        ),
                        group_desc=f"Federated group #{self.federated_rank}",
                    )
        else:
            logger.error(
                f"Impossible setting up local distributed environment configuration: the distributed environment is not initialized"
            )

    def unset_federated_scaling(self) -> None:
        """
        Unset Federated Scaling parameters
        """
        self.federated_local_rank = None
        self.federated_local_size = None
        self.federated_rank = None
        self.federated_world_size = None
        self.federated_group = None
        self.replica_group = None
        self.federation = None

    def is_federated_scaling_setup(self) -> bool:
        """
        Checks if Federated Scaling is set up.

        :return: True if Federated Scaling is set up, False otherwise
        :rtype: bool
        """
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


### TESTING ###

processes: int = 8
