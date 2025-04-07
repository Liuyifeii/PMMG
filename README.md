# PMMG

<div align="center">
  <img src="https://github.com/Liuyifeii/PMMG/blob/main/img/For%20Table%20of%20Contents%20Only.png" width="95%">
</div>


## Setup :environment:
```bash
conda create -n PMMG python=3.7
```
### Requirements :memo:

<details>
  <summary>Click to show/hide requirements</summary>

1. python: 3.7
2. rdkit-pypi==2021.03.5
3. numpy~=1.19.2
4. tensorflow: 2.5.0
5. protobuf~=3.9.2
6. pyyaml
7. pandas
8. joblib

</details>

## Run PMMG :

### 1. Prepare a reward file
Please refer to `reward/README.md`.

### 2. Prepare a config file

### 3. Generate molecules

#### ChemTSv2 with single process mode :red_car:

```bash
python run.py -c config/setting.yaml
```


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
