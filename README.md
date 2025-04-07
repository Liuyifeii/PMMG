# PMMG

<div align="center">
  <img src="https://github.com/Liuyifeii/PMMG/blob/main/img/For%20Table%20of%20Contents%20Only.png" width="95%">
</div>


## How to setup :pushpin:

### Requirements :memo:
<details>
  <summary>Click to show/hide requirements</summary>

1. python: 3.7
2. rdkit: 2021.03.5
3. tensorflow: 2.5.0
4. pyyaml
5. pandas
6. joblib

#### (a) Installation on a server WITH a MPI environment

```bash
cd YOUR_WORKSPACE
python3.7 -m venv .venv
source .venv/bin/activate
pip install --upgrade chemtsv2
pip install mpi4py==3.0.3
```

#### (b) Installation on a server WITHOUT a MPI environment

```bash
conda create -n mpchem python=3.7
# swith to the `mpchem` environment
conda install -c conda-forge openmpi cxx-compiler mpi mpi4py=3.0.3
pip install --upgrade chemtsv2
```
</details>

## How to run PMMG :pushpin:

### 1. Prepare a reward file
Please refer to `reward/README.md`.

### 2. Prepare a config file

### 3. Generate molecules

#### ChemTSv2 with single process mode :red_car:

```bash
chemtsv2 -c config/setting.yaml
```

#### ChemTSv2 with massive parallel mode :airplane:

```bash
mpiexec -n 4 chemtsv2-mp --config config/setting_mp.yaml
```

## Advanced usege :pushpin:

### Extend user-specified SMILES

You can extend the SMILES string you input.
In this case, you need to put the atom you want to extend at the end of the string and run ChemTS with `--input_smiles` argument as follows.

```bash
chemtsv2 -c config/setting.yaml --input_smiles 'C1=C(C)N=CC(N)=C1C'
```

### GPU acceleration

If you want to use GPU, run ChemTS with `--gpu GPU_ID` argument as follows.

```bash
chemtsv2 -c config/setting.yaml --gpu 0
```

## License :pushpin:

This package is distributed under the MIT License.

## Contact :pushpin:

- Yifei Liu (yifeiliu@zju.edu.cn)
