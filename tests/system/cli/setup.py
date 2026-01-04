import asyncio
import json
import logging
import socket
import sys
import tempfile

import asyncssh
import docker
from huggingface_hub import snapshot_download

logger = logging.getLogger("docker_deployer")
logger.setLevel(logging.INFO)


class PortManager:
    def __init__(self):
        self.reserved_ports = set()
        self.lock = asyncio.Lock()

    async def get_unique_port(self):
        async with self.lock:
            while True:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", 0))
                    port = s.getsockname()[1]
                if port not in self.reserved_ports:
                    self.reserved_ports.add(port)
                    return port


async def launch_single_container(index, d_client, port_manager):
    skey = asyncssh.generate_private_key(
        alg_name="ssh-rsa",
        comment=f"streamflow-test-{index}",
        key_size=4096,
    )
    public_key = skey.export_public_key().decode("utf-8")
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        skey.write_private_key(f.name)
        private_key_path = f.name

    ssh_port = await port_manager.get_unique_port()
    container_name = f"ssh-server-{index}-{ssh_port}"
    container_env = [f"PUBLIC_KEY={public_key}"]
    docker_image = "lscr.io/linuxserver/openssh-server"
    try:
        container = d_client.containers.run(
            image=docker_image,
            name=container_name,
            environment=container_env,
            ports={"2222/tcp": ssh_port},
            detach=True,
            remove=True,
        )
        logger.info(f"  [+] Started {container_name} on port {ssh_port}")
        return {"id": container.id[:12], "port": ssh_port, "key": private_key_path}
    except Exception as e:
        logger.error(f"  [!] Failed to start {container_name}: {e}")
        return None


async def main():
    if len(sys.argv) < 2:
        logger.error(f"Usage: python {sys.argv[0]} <number_of_containers>")
        sys.exit(1)

    try:
        num_containers = int(sys.argv[1])
    except ValueError:
        logger.error(f"Error: '{sys.argv[1]}' is not a valid number.")
        sys.exit(1)
    try:
        docker_client = docker.from_env()
    except Exception as e:
        logger.error(f"Could not connect to Docker: {e}")
        sys.exit(1)
    port_manager = PortManager()
    logger.info(f"Deploying {num_containers} instances...")
    results = [
        res
        for res in await asyncio.gather(
            *(
                launch_single_container(i, docker_client, port_manager)
                for i in range(num_containers)
            )
        )
        if res is not None
    ]
    if len(results) != num_containers:
        logger.info(f"Something went wrong while deploying {num_containers} instances.")
        logger.info(f"Undeploying {len(results)} instances...")
        for res in results:
            docker_client.containers.get(res["id"]).stop()
        raise Exception(f"Could not deploy {num_containers} instances.")

    print(json.dumps(results, indent=2))

    model_path = snapshot_download(
        repo_id="llamafactory/tiny-random-Llama-3",
        local_dir="tiny-random-llama-3",
        local_dir_use_symlinks=False,
    )

    xffl_answers = (
        f"../../../examples/intra-silo/03_LLM/training.py\n"
        f"../../../examples/intra-silo/03_LLM/config.py\n"
        f"{model_path}\n"
        f"output\n"
        f"/tmp\n"
        f"2\n"
    )
    with open("config.txt", "w") as fd:
        fd.write(xffl_answers)
        for i, res in enumerate(results):
            docker_client.containers.get(res["id"]).exec_run(
                f"mkdir -p /tmp/user-{res['port']}/datasets1 && "
                f"touch /tmp/user-{res['port']}/placeholder.sif",
                user="linuxserver.io",
            )
            fd.write(
                f"dummy{res['port']}\ndummy\n127.0.0.1:{res['port']}\n"
                f"linuxserver.io\n{res['key']}\n \n/tmp/user-{res['port']}/workdir\n"
                f"/tmp/user-{res['port']}/placeholder.sif\n"
                f"/tmp/user-{res['port']}/datasets1\n"
            )
            if i < len(results) - 1:
                fd.write("y\n")
            else:
                fd.write("n\n")


if __name__ == "__main__":
    asyncio.run(main())
