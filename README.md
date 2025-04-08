# PMMG：A Multi-Objective Molecular Generation Method Based on Pareto Algorithm and Monte Carlo Tree Search

<div align="center">
  <img src="https://github.com/Liuyifeii/PMMG/blob/main/img/For%20Table%20of%20Contents%20Only.png" width="95%">
</div>



## 🛠️ Setup

Recommend using 'conda' to manage project environments:

```bash
conda create -n PMMG python=3.7
conda activate PMMG
```
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
```bash
class MyReward:
    def __call__(self, smiles):
        # User defined
        return score
```
### 2. Prepare a config file
please refer to config/7goals_set
### 3. Generate molecules
```bash
python run.py -c config/setting.yaml
```
you can also use -- input_stiles to specify the starting molecule for directional optimization
```bash
python run.py -c config/setting.yaml
```
### 4. If you want to train the RNN model yourself
you can change the setting in the TRAIN_setting.yaml and then:
```bash
python train.py -c TRAIN_setting.yaml
```
## License :pushpin:
This package is distributed under the MIT License.

## Contact :pushpin:
- Yifei Liu (yifeiliu@zju.edu.cn)
