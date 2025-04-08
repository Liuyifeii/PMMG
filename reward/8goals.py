import os
import pickle
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append("./data/")
import sascorer

from chemtsv2.misc.scaler import minmax, max_gauss, min_gauss, rectangular
from reward.reward import Reward

# === Load pretrained LGB models ===
LGB_MODELS_PATH = 'data/model/lgb_models.pickle'
with open(LGB_MODELS_PATH, 'rb') as models:
    lgb_models = pickle.load(models)

# === Utility: Scaling dispatcher ===
def scale_objective_value(params, value):
    scaling_type = params.get("type", "identity")
    if scaling_type == "max_gauss":
        return max_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling_type == "min_gauss":
        return min_gauss(value, params["alpha"], params["mu"], params["sigma"])
    elif scaling_type == "minmax":
        return minmax(value, params["min"], params["max"])
    elif scaling_type == "rectangular":
        return rectangular(value, params["min"], params["max"])
    elif scaling_type == "identity":
        return value
    else:
        raise ValueError(f"Unsupported scaling type: {scaling_type}")


# === Core Class ===
class eight_goals(Reward):

    @staticmethod
    def _predict_scaled(mol, target, scale_fn):
        if mol is None:
            return None
        fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
        raw = lgb_models[target].predict(fp)[0]
        return scale_fn(raw)

    @staticmethod
    def get_objective_functions(conf):
        def EGFR(mol): return eight_goals._predict_scaled(mol, "EGFR", lambda x: max_gauss(x, a=1, mu=9, sigma=3))
        def ERBB2(mol): return eight_goals._predict_scaled(mol, "ERBB2", lambda x: max_gauss(x, a=1, mu=9, sigma=3))
        def Solubility(mol): return eight_goals._predict_scaled(mol, "Sol", lambda x: max_gauss(x, a=1, mu=0.3, sigma=1.7))
        def Permeability(mol): return eight_goals._predict_scaled(mol, "Perm", lambda x: max_gauss(x, a=1, mu=1.8, sigma=0.5))
        def Metabolic_stability(mol): return eight_goals._predict_scaled(mol, "Meta", lambda x: max_gauss(x, a=1, mu=110, sigma=32))
        def QED(mol): return eight_goals._safe_qed(mol)
        def Toxicity(mol): return eight_goals._predict_scaled(mol, "Tox", lambda x: min_gauss(x, a=1, mu=1.4, sigma=1))
        def SAScore(mol): return eight_goals._safe_sascore(mol)

        return [EGFR, ERBB2, Solubility, Permeability, Metabolic_stability, QED, Toxicity, SAScore]

    @staticmethod
    def get_true_objective_functions(conf):
        def raw_predict(mol, target):
            if mol is None:
                return None
            fp = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)]
            return lgb_models[target].predict(fp)[0]

        def EGFR(mol): return raw_predict(mol, "EGFR")
        def ERBB2(mol): return raw_predict(mol, "ERBB2")
        def Solubility(mol): return raw_predict(mol, "Sol")
        def Permeability(mol): return raw_predict(mol, "Perm")
        def Metabolic_stability(mol): return raw_predict(mol, "Meta")
        def QED(mol): return eight_goals._safe_qed(mol)
        def Toxicity(mol): return raw_predict(mol, "Tox")
        def SAScore(mol): return sascorer.calculateScore(mol) if mol else None

        return [EGFR, ERBB2, Solubility, Permeability, Metabolic_stability, QED, Toxicity, SAScore]

    @staticmethod
    def _safe_qed(mol):
        try:
            return Chem.QED.qed(mol)
        except Exception:
            return None

    @staticmethod
    def _safe_sascore(mol):
        if mol is None:
            return None
        sa = sascorer.calculateScore(mol)
        return (10.0 - sa) / 9.0

    @staticmethod
    def calc_reward_from_objective_values(values, conf):
        if None in values:
            return [0] * len(values)
        return np.array(values)
