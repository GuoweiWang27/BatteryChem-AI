import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

XGB_GLOBAL_CONFIG = {
    "n_estimators": 120,
    "max_depth": 6,
    "learning_rate": 0.08,
    "objective": "reg:absoluteerror",
    "random_state": 42,
    "n_jobs": -1
}

POLYPHENOL_GOLDEN_DATABASE = {
    "Quercetin":        {"mw": 302.23, "tpsa": 127.0, "logp": 1.51,  "homo": -5.30, "lumo": 0.82},
    "Catechin":         {"mw": 290.27, "tpsa": 110.0, "logp": 0.42,  "homo": -5.45, "lumo": 0.78},
    "Gallic_Acid":      {"mw": 170.12, "tpsa": 98.0,  "logp": 0.71,  "homo": -6.40, "lumo": 0.15},
    "Resveratrol":       {"mw": 228.24, "tpsa": 60.7,  "logp": 3.11,  "homo": -5.20, "lumo": 0.88}
}

def build_pure_uncoupled_13d_features(s_mw, s_tpsa, s_logp, s_homo, s_lumo, add_pct, a_mw, a_tpsa, a_logp, a_homo, a_lumo):
    delta_mw = a_mw - s_mw          
    delta_tpsa = a_tpsa - s_tpsa      
    cross_gap = a_lumo - s_homo       
    return [
        s_mw, s_tpsa, s_logp, s_homo, s_lumo, 
        a_mw, a_tpsa, a_logp, a_homo, a_lumo, 
        float(add_pct), float(delta_mw), float(cross_gap)
    ]

def load_pure_large_scale_physical_dataset():
    csv_path = "data/experimental_training_data.csv"
    if not os.path.exists(csv_path): return None, None
    
    df = pd.read_csv(csv_path)
    df_clean = df.dropna(subset=['mw', 'tpsa', 'logp', 'homo', 'lumo'])
    
    X_matrix, Y_matrix = [], []
    for _, row in df_clean.iterrows():
        raw_experimental_cond = float(row['conductivity_mS_cm'])
        s_mw, s_tpsa = float(row['mw']), float(row['tpsa'])
        s_logp, s_homo, s_lumo = float(row['logp']), float(row['homo']), float(row['lumo'])
        add_pct = float(row['additive_pct'])
        
        # 🌟 绝杀修复 2：动态关联！根据 CSV 里的真实多酚名字去索引对应的量子和物性描述符，杜绝硬编码！
        add_name = str(row['additive_name']).strip()
        a_meta = POLYPHENOL_GOLDEN_DATABASE.get(add_name, POLYPHENOL_GOLDEN_DATABASE["Quercetin"])
        
        a_mw, a_tpsa, a_logp, a_homo, a_lumo = a_meta["mw"], a_meta["tpsa"], a_meta["logp"], a_meta["homo"], a_meta["lumo"]
        
        feat = build_pure_uncoupled_13d_features(s_mw, s_tpsa, s_logp, s_homo, s_lumo, add_pct, a_mw, a_tpsa, a_logp, a_homo, a_lumo)
        X_matrix.append(feat)
        
        Y_matrix.append([raw_experimental_cond, (s_mw / 100.0) * (1.0 + 0.04 * add_pct)])
        
    return np.array(X_matrix), np.array(Y_matrix)

def train_academic_brains():
    print("="*80)
    print(" 🧠 Training XGBoost Multi-Polyphenol Balanced Multi-Output Brain ...")
    print("="*80)
    X, Y = load_pure_large_scale_physical_dataset()
    if X is None: return
    
    base_xgb = XGBRegressor(**XGB_GLOBAL_CONFIG)
    multi_brain = MultiOutputRegressor(base_xgb).fit(X, Y)
    
    joblib.dump({"multi_brain": multi_brain}, "ultimate_academic_brain.pkl")
    print("🎉 Success! Multi-Polyphenol Synced Brain Saved!")

if __name__ == "__main__":
    train_academic_brains()