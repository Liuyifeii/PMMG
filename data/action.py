from rdkit import Chem
import warnings
warnings.filterwarnings("ignore")
import pickle
import sys
from rdkit.Chem import Descriptors, rdMolDescriptors
#from filter.filter import Filter
import pandas as pd
import sys
#sys.path.append("./data/")
import sascorer
from rdkit.Chem import AllChem
import numpy as np
import csv
import logging

# 配置日志
#logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


LGB_MODELS_PATH = 'model/lgb_models.pickle'
SURE_CHEMBL_ALERTS_PATH = 'sure_chembl_alerts.txt'

with open(LGB_MODELS_PATH, mode='rb') as models,\
    open(SURE_CHEMBL_ALERTS_PATH, mode='rb') as alerts:
    lgb_models = pickle.load(models)
    smarts = pd.read_csv(alerts, header=None, sep='\t')[1].tolist()
    alert_mols = [Chem.MolFromSmarts(smart) for smart in smarts if Chem.MolFromSmarts(smart) is not None]

def max_gauss(x, a=1, mu=8, sigma=2):
    if x > mu:
        return 1
    else :
        return a * np.exp(-(x-mu)**2 / (2*sigma**2))


def min_gauss(x, a=1, mu=2, sigma=2):
    if x < mu:
        return 1
    else :
        return a * np.exp(-(x-mu)**2 / (2*sigma**2))

def EGFR(mol):
    if mol is None:
        return None
    fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
    true_egfr = lgb_models["EGFR"].predict(fp)[0]
    scaled_egfr =  max_gauss(true_egfr, a=1, mu=9, sigma=3)
    return scaled_egfr

def ERBB2(mol):
     if mol is None:
        return None
     fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
     true_erbb2 =  lgb_models["ERBB2"].predict(fp)[0]
     scaled_erbb2 =  min_gauss(true_erbb2, a=1, mu=3, sigma=2)
     return scaled_erbb2

def SAScore(mol):
    sa = sascorer.calculateScore(mol)
    scaled_sa = (10. - sa) / 9.
    return scaled_sa

def Solubility(mol):
    if mol is None:
        return None
    fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
    true_Solubility = lgb_models["Sol"].predict(fp)[0]
    scaled_Solubility = max_gauss(true_Solubility, a=1, mu=0.3, sigma=1.7)
    return scaled_Solubility

def Permeability(mol):
    if mol is None:
        return None
    fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
    true_Permeability = lgb_models["Perm"].predict(fp)[0]
    scaled_Permeability = max_gauss(true_Permeability, a=1, mu=1.8, sigma=0.5)
    return scaled_Permeability

def Metabolic_stability(mol):
    if mol is None:
        return None
    fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
    true_Metabolic_stability = lgb_models["Meta"].predict(fp)[0]
    scaled_Metabolic_stability = max_gauss(true_Metabolic_stability, a=1, mu=110, sigma=32)
    return scaled_Metabolic_stability

def Toxicity(mol):
    if mol is None:
        return None
    fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
    true_Toxicity = lgb_models["Tox"].predict(fp)[0]
    scaled_Toxicity = min_gauss(true_Toxicity, a=1, mu=1.4, sigma=1)
    return scaled_Toxicity

def check(mol):
     weight = round(rdMolDescriptors._CalcMolWt(mol), 2)
     logp = Descriptors.MolLogP(mol)
     donor = rdMolDescriptors.CalcNumLipinskiHBD(mol)
     acceptor = rdMolDescriptors.CalcNumLipinskiHBA(mol)
     rotbonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
     cond = weight <= 500 and logp <= 5 and donor <= 5 and acceptor <= 10
     print(cond)

def LogP(mol):
    return Descriptors.MolLogP(mol)

def QED(mol):
    try:
        return Chem.QED.qed(mol)
    except (Chem.rdchem.AtomValenceException, Chem.rdchem.KekulizeException):
        return None

def tox_alert(mol):
    if np.any([mol.HasSubstructMatch(alert) for alert in alert_mols]):
     score = 0
    else:
     score = 1
    return score

def remove_last_character(input_string):
    if input_string:
        return input_string[1:-2]
    else:
        return input_string
    
excel_file_path = 'smiles_ga.csv'

column_name = 'SMILES'  # 或者使用列索引，例如：column_index = 0

df = pd.read_csv(excel_file_path)

column_data = df[column_name]  # 或者 df.iloc[:, column_index]
for i in range(376):
     s = column_data.iloc[i]
     smiles = remove_last_character(s)

     with open("smiles-output.txt", "a") as file:
         import sys
         sys.stdout = file
         print(smiles)

         sys.stdout = sys.__stdout__




#print(value1)
# 现在，变量 column_data 包含了Excel文件中指定列的数据
# 你可以使用 column_data 变量来处理该列的数据

#LGB_MODELS_PATH = 'lgb_models.pickle'
#with open(LGB_MODELS_PATH, mode='rb') as models:
#   lgb_models = pickle.load(models)


#a = 'C/C=C1/CNC2=NC3=C(C[C@H]2C1)C(=O)c1ccc(COC[C@H](C)Nc2cn[nH]c(=O)c2C(F)(F)F)c(=O)n1CC3'

