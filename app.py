import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

XGB_GLOBAL_CONFIG = {
    "n_estimators": 120,
    "max_depth": 6,          
    "learning_rate": 0.08,
    "objective": "reg:absoluteerror",
    "random_state": 42,
    "n_jobs": -1
}

REAL_ADDITIVES_DATABASE = {
    "FEC":          {"mw": 106.05, "tpsa": 35.53, "logp": 0.09,  "homo": -7.10, "lumo": -0.20},
    "VC":           {"mw": 84.03,  "tpsa": 26.30, "logp": 0.24,  "homo": -6.85, "lumo": -0.35},
    "PS":           {"mw": 104.13, "tpsa": 51.75, "logp": -0.41, "homo": -6.95, "lumo": -0.45},
    "SN":           {"mw": 80.09,  "tpsa": 47.58, "logp": -0.99, "homo": -7.30, "lumo": 0.12},
    "ADN":          {"mw": 108.14, "tpsa": 47.58, "logp": -0.46, "homo": -7.25, "lumo": 0.15},
    "DTD":          {"mw": 124.12, "tpsa": 51.75, "logp": -0.85, "homo": -7.05, "lumo": -0.30},
    "Quercetin":    {"mw": 302.23, "tpsa": 127.0, "logp": 1.51,  "homo": -5.30, "lumo": 0.82},
    "Catechin":     {"mw": 290.27, "tpsa": 110.0, "logp": 0.42,  "homo": -5.45, "lumo": 0.78},
    "Gallic_Acid":  {"mw": 170.12, "tpsa": 98.0,  "logp": 0.71,  "homo": -6.40, "lumo": 0.15},
    "Resveratrol":   {"mw": 228.24, "tpsa": 60.7,  "logp": 3.11,  "homo": -5.20, "lumo": 0.88}
}

REAL_SALTS_DATABASE = {
    "LiPF6":  {"mw": 151.91, "tpsa": 0.00},
    "LiFSI":  {"mw": 187.13, "tpsa": 92.50},
    "LiTFSI": {"mw": 287.09, "tpsa": 104.80}
}

def build_pure_uncoupled_15d_features(s_mw, s_tpsa, s_logp, s_homo, s_lumo, salt_mw, salt_tpsa, add_pct, a_mw, a_tpsa, a_logp, a_homo, a_lumo):
    delta_mw = a_mw - s_mw          
    delta_tpsa = a_tpsa - s_tpsa      
    cross_gap = a_lumo - s_homo       
    return [
        s_mw, s_tpsa, s_logp, s_homo, s_lumo, 
        float(salt_mw), float(salt_tpsa),
        a_mw, a_tpsa, a_logp, a_homo, a_lumo, 
        float(add_pct), float(delta_mw), float(cross_gap)
    ]

def load_pure_large_scale_physical_dataset():
    csv_path = "data/experimental_training_data.csv"
    if not os.path.exists(csv_path): return None, None
    
    df = pd.read_csv(csv_path)
    df_clean = df.dropna(subset=['mw', 'tpsa', 'logp', 'homo', 'lumo'])
    
    X_matrix, Y_vector = [], []
    for _, row in df_clean.iterrows():
        raw_experimental_cond = float(row['conductivity_mS_cm'])
        s_mw, s_tpsa = float(row['mw']), float(row['tpsa'])
        s_logp, s_homo, s_lumo = float(row['logp']), float(row['homo']), float(row['lumo'])
        add_pct = float(row['additive_pct'])
        
        # 🌟 现在由于 data_pipeline.py 已经写入了 salt 键，这里读取出的将是精确的真实混合锂盐名称，彻底告别盲盒简并！
        salt_name = str(row.get('salt', 'LiPF6')).strip()
        salt_meta = REAL_SALTS_DATABASE.get(salt_name, REAL_SALTS_DATABASE["LiPF6"])
        
        add_name = str(row['additive_name']).strip()
        a_meta = REAL_ADDITIVES_DATABASE.get(add_name, REAL_ADDITIVES_DATABASE["Quercetin"])
        a_mw, a_tpsa, a_logp, a_homo, a_lumo = a_meta["mw"], a_meta["tpsa"], a_meta["logp"], a_meta["homo"], a_meta["lumo"]
        
        feat = build_pure_uncoupled_15d_features(
            s_mw, s_tpsa, s_logp, s_homo, s_lumo, 
            salt_meta["mw"], salt_meta["tpsa"],
            add_pct, a_mw, a_tpsa, a_logp, a_homo, a_lumo
        )
        X_matrix.append(feat)
        Y_vector.append(raw_experimental_cond)
        
    return np.array(X_matrix), np.array(Y_vector)

def train_academic_brains():
    print("="*80)
    print(" 🧠 Training 15-Dimensional Pure Data-Driven XGBoost Brain ...")
    print("="*80)
    X, Y = load_pure_large_scale_physical_dataset()
    if X is None: return
    
    cond_brain = XGBRegressor(**XGB_GLOBAL_CONFIG).fit(X, Y)
    joblib.dump({"cond_brain": cond_brain}, "ultimate_academic_brain.pkl")
    print("🎉 Success! 15D Equation-Free AI Brain Saved Successfully!")

if __name__ == "__main__":
    train_academic_brains()