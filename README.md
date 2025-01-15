# Cross-Facility Federated Learning (xFFL)

[Cross-Facility Federated Learning (xFFL)](https://hpc4ai.unito.it/hpc-federation/) is a federated learning (FL) framework based on the [StreamFlow workflow management system (WMS)](https://streamflow.di.unito.it) developed in the [Parallel Computing \[Alpha\] research group](https://alpha.di.unito.it) at the University of Turin, Italy.

xFFL is designed to be a research-oriented software, enabling FL workloads deployment on geographically distributed computing infrastructures including, but not limited to, high-performance computers (HPCs).

Our aim is to democratize large ML workloads development, allowing researchers and SMEs to be competitive with BigTech companies through the exploitation of sparse computing power.

<table width="100%">
  <tr>
    <td style='text-align:center; vertical-align:middle; border:none!important;'><img src="figures/unito.png" alt="UniTo" width="200" height=110/></td>
    <td style='text-align:center; vertical-align:middle; border:none!important;'><img src="figures/alpha.png" alt="Alpha" width="100" height=100/></td>
    <td style='text-align:center; vertical-align:middle; border:none!important;'><img src="figures/streamflow.png" alt="StreamFlow" width="200" height="70"/></td>
  </tr>
</table>


## Get started

### Setup
Clone this repository on the computing infrastructure that will act as the federation orchestrator:
```bash
git clone --depth 1 --branch main https://github.com/alpha-unito/xffl.git
```
Note that this machine should have internet access and be capable of reaching throgh ssh the federation clients.

Create and activate a Python virtual environment (both venv and conda are valid options):
```bash
python3 -m venv xFFL
source xFFL/bin/activate
```

Install the requirements necessary to run the StreamFlow WMS:
```bash
cd xffl/
python3 -m pip install -r requirements.txt
```

### Configuration creation
Execute the `config.py` script to create a xFFL configuration file in a guided way:
```bash
python3 config.py --workdir [workdir] --project [project]
```
where `workdir` is the name of the working directory to create, and `project` is the name of the created deployment.
This will produce a directory containing the necessary StreamFlow and xFFL configuration files, together with the choosen model.

## Requirements

Python>3.8  
Singularity/Docker  

## Architecture

xFFL implements a centralized FL schema, in which a central server coordinates, distributes and aggregates the learning tasks deployed on the clients. Such process is orchestrated by StreamFlow, which takes care of handling the data movement between the computing infrastructures and deploying the workloads.

![xFFL](figures/xffl.png)


## Contributors

|                  |                             |                          |
| ---------------: | :-------------------------- | :----------------------- |
| [Gianluca Mittone](https://alpha.di.unito.it/gianluca-mittone/) | <gianluca.mittone@unito.it> | creator and maintainer   |
| [Alberto Mulone](https://alpha.di.unito.it/alberto-mulone/)   | <alberto.mulone@unito.it>   | developer and maintainer |  
| [Iacopo Colonnelli](https://alpha.di.unito.it/iacopo-colonnelli/)| <iacopo.colonnelli@unito.it>| investigator             | 
| [Robert Birke](https://alpha.di.unito.it/robert-rene-maria-birke/)     | <robert.birke@unito.it>     | investigator             |   
| [Marco Aldinucci](https://alpha.di.unito.it/marco-aldinucci/)  | <marco.aldinucci@unito.it>  | principal investigator   |