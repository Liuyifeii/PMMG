# PMMG

<div align="center">
  <img src="https://github.com/molecule-generator-collection/ChemTSv2/blob/master/img/logo.png" width="95%">
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
An example of reward definition for LogP maximization task is as follows.
```python
from rdkit.Chem import Descriptors
import numpy as np
from reward.reward import Reward

class LogP_reward(Reward):
    def get_objective_functions(conf):
        def LogP(mol):
            return Descriptors.MolLogP(mol)
        return [LogP]
    
    def calc_reward_from_objective_values(objective_values, conf):
        logp = objective_values[0]
        return np.tanh(logp/10)
```

### 2. Prepare a config file

The explanation of options are described in the [Support option/function](#support-optionfunction-pushpin) section. 
The prepared reward file needs to be specified in `reward_setting`.
For details, please refer to a sample file ([config/setting.yaml](config/setting.yaml)). 
If you want to pass any value to `calc_reward_from_objective_values` (e.g., weights for each value), add it in the config file.

### 3. Generate molecules

#### ChemTSv2 with single process mode :red_car:

```bash
chemtsv2 -c config/setting.yaml
```

#### ChemTSv2 with massive parallel mode :airplane:

```bash
mpiexec -n 4 chemtsv2-mp --config config/setting_mp.yaml
```

## Example usage :pushpin:

|Target|Reward|Config|Additional requirement|Ref.|
|---|---|---|---|---|
|LogP|[logP_reward.py](reward/logP_reward.py)|[setting.yaml](config/setting.yaml)|-|-|
|Jscore|[Jscore_reward.py](reward/Jscore_reward.py)|[setting_jscore.yaml](config/setting_jscore.yaml)|-|[^1]|
|Absorption wavelength|[chro_reward.py](reward/chro_reward.py)|[setting_chro.yaml](config/setting_chro.yaml)|Gaussian 16[^3]<br> via QCforever[^10]|[^4]|
|Upper-absorption & fluorescence<br> wavelength|[fluor_reward.py](reward/fluor_reward.py)|[setting_fluor.yaml](config/setting_fluor.yaml)|Gaussian 16[^3]<br> via QCforever[^10]|[^5]|
|Kinase inhibitory activities|[dscore_reward.py](reward/dscore_reward.py)|[setting_dscore.yaml](config/setting_dscore.yaml)|LightGBM[^6]|[^7]|
|Docking score|[Vina_binary_reward.py](reward/Vina_binary_reward.py)|[setting_vina_binary.yaml](config/setting_vina_binary.yaml)|AutoDock Vina[^8]|[^9]|
|Pharmacophore|[pharmacophore_reward.py](reward/pharmacophore_reward.py)|[setting_pharmacophore.yaml](config/setting_pharmacophore.yaml)|-|[^11]|

[^3]: Frisch, M. J. et al. Gaussian 16 Revision C.01. 2016; Gaussian Inc. Wallingford CT.
[^4]: Sumita, M., Yang, X., Ishihara, S., Tamura, R., & Tsuda, K. (2018). Hunting for Organic Molecules with Artificial Intelligence: Molecules Optimized for Desired Excitation Energies. ACS Central Science, 4(9), 1126–1133. https://doi.org/10.1021/acscentsci.8b00213
[^5]: Sumita, M., Terayama, K., Suzuki, N., Ishihara, S., Tamura, R., Chahal, M. K., Payne, D. T., Yoshizoe, K., & Tsuda, K. (2022). De novo creation of a naked eye–detectable fluorescent molecule based on quantum chemical computation and machine learning. Science Advances, 8(10). https://doi.org/10.1126/sciadv.abj3906
[^6]: Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., … Liu, T.-Y. (2017). Lightgbm: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.
[^7]: Yoshizawa, T., Ishida, S., Sato, T., Ohta, M., Honma, T., & Terayama, K. (2022). Selective Inhibitor Design for Kinase Homologs Using Multiobjective Monte Carlo Tree Search. Journal of Chemical Information and Modeling, 62(22), 5351–5360. https://doi.org/10.1021/acs.jcim.2c00787
[^8]: Eberhardt, J., Santos-Martins, D., Tillack, A. F., & Forli, S. (2021). AutoDock Vina 1.2.0: New Docking Methods, Expanded Force Field, and Python Bindings. Journal of Chemical Information and Modeling, 61(8), 3891–3898. https://doi.org/10.1021/acs.jcim.1c00203
[^9]: Ma, B., Terayama, K., Matsumoto, S., Isaka, Y., Sasakura, Y., Iwata, H., Araki, M., & Okuno, Y. (2021). Structure-Based de Novo Molecular Generator Combined with Artificial Intelligence and Docking Simulations. Journal of Chemical Information and Modeling, 61(7), 3304–3313. https://doi.org/10.1021/acs.jcim.1c00679
[^10]: Sumita, M., Terayama, K., Tamura, R., & Tsuda, K. (2022). QCforever: A Quantum Chemistry Wrapper for Everyone to Use in Black-Box Optimization. Journal of Chemical Information and Modeling, 62(18), 4427–4434. https://doi.org/10.1021/acs.jcim.2c00812
[^11]: 石田祥一, 吉澤竜哉, 寺山慧 (2023). 深層学習と木探索に基づくde novo分子設計, SAR News, 44.


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
