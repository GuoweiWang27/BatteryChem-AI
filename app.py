import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor  # 🚀 斩断森林，全面换血升级为高级梯度提升树

# =====================================================================
# 📚 1. 黄金真分子描述符资产库 (对齐 data_pipeline)
# =====================================================================
MOLECULAR_GOLDEN_DATABASE = {
    "DMC":  {"mw": 90.08,  "tpsa": 26.30, "logp": 0.23,  "homo": -6.50, "lumo": 0.50,  "is_solvent": True},
    "DEC":  {"mw": 118.13, "tpsa": 26.30, "logp": 0.80,  "homo": -6.45, "lumo": 0.52,  "is_solvent": True},
    "EC":   {"mw": 88.06,  "tpsa": 35.53, "logp": 0.11,  "homo": -7.21, "lumo": 0.45,  "is_solvent": True}, # 彻底对齐
    "EMC":  {"mw": 104.11, "tpsa": 26.30, "logp": 0.47,  "homo": -6.40, "lumo": 0.53,  "is_solvent": True},
    "PC":   {"mw": 102.09, "tpsa": 26.30, "logp": 0.06,  "homo": -6.65, "lumo": 0.42,  "is_solvent": True},
    "DOL":  {"mw": 74.08,  "tpsa": 9.23,  "logp": 0.22,  "homo": -6.15, "lumo": 0.65,  "is_solvent": True},
    "DME":  {"mw": 90.12,  "tpsa": 18.46, "logp": -0.21, "homo": -6.20, "lumo": 0.60,  "is_solvent": True},
    "MP":   {"mw": 88.11,  "tpsa": 26.30, "logp": 0.84,  "homo": -6.35, "lumo": 0.55,  "is_solvent": True},
    "EP":   {"mw": 102.13, "tpsa": 26.30, "logp": 1.30,  "homo": -6.30, "lumo": 0.57,  "is_solvent": True},
    "THF":  {"mw": 72.11,  "tpsa": 9.23,  "logp": 1.75,  "homo": -6.05, "lumo": 0.70,  "is_solvent": True},
    "GBL":  {"mw": 86.09,  "tpsa": 26.30, "logp": -0.64, "homo": -6.55, "lumo": 0.48,  "is_solvent": True},
    "AN":   {"mw": 41.05,  "tpsa": 23.79, "logp": -0.34, "homo": -7.20, "lumo": 0.05,  "is_solvent": True},
    "FEC":  {"mw": 106.05, "tpsa": 35.53, "logp": 0.09,  "homo": -7.10, "lumo": -0.20, "is_solvent": False},
    "VC":   {"mw": 84.03,  "tpsa": 26.30, "logp": 0.24,  "homo": -6.85, "lumo": -0.35, "is_solvent": False},
    "PS":   {"mw": 104.13, "tpsa": 51.75, "logp": -0.41, "homo": -6.95, "lumo": -0.45, "is_solvent": False},
    "SN":   {"mw": 80.09,  "tpsa": 47.58, "logp": -0.22, "homo": -7.50, "lumo": -0.10, "is_solvent": False},
    "ADN":  {"mw": 108.14, "tpsa": 47.58, "logp": 0.79,  "homo": -7.42, "lumo": -0.12, "is_solvent": False},
    "DTD":  {"mw": 108.12, "tpsa": 51.75, "logp": -1.55, "homo": -7.05, "lumo": -0.40, "is_solvent": False},
    "Quercetin":        {"mw": 302.23, "tpsa": 127.0, "logp": 1.51,  "homo": -5.30, "lumo": 0.82,  "is_solvent": False},
    "Catechin":         {"mw": 290.27, "tpsa": 110.0, "logp": 0.42,  "homo": -5.45, "lumo": 0.78,  "is_solvent": False},
    "Epigallocatechin":  {"mw": 306.27, "tpsa": 129.0, "logp": 0.22,  "homo": -5.35, "lumo": 0.80,  "is_solvent": False},
    "Gallic_Acid":      {"mw": 170.12, "tpsa": 98.0,  "logp": 0.71,  "homo": -6.40, "lumo": 0.15,  "is_solvent": False},
    "Resveratrol":       {"mw": 228.24, "tpsa": 60.7,  "logp": 3.11,  "homo": -5.20, "lumo": 0.88,  "is_solvent": False}
}

def build_advanced_13d_features(s_mw, s_tpsa, s_logp, s_homo, s_lumo, add_pct, a_mw, a_tpsa, a_logp, a_homo, a_lumo):
    cross_gap = a_lumo - s_homo
    combined_mass = s_mw * 0.9 + a_mw * (add_pct / 100.0)
    approx_viscosity = 0.1 * np.exp((0.0073 * combined_mass + 0.0115 * s_tpsa))
    inv_viscosity = 1.0 / approx_viscosity
    dielectric_field = (s_tpsa + a_tpsa * (add_pct / 100.0)) / (s_mw + 1e-5)
    return [s_mw, s_tpsa, s_logp, s_homo, s_lumo, a_mw, a_tpsa, a_logp, a_homo, a_lumo, cross_gap, inv_viscosity, dielectric_field]

def load_pure_data_driven_dataset():
    csv_path = "data/experimental_training_data.csv"
    if not os.path.exists(csv_path): return None, None
    df = pd.read_csv(csv_path)
    df_clean = df.dropna(subset=['mw', 'tpsa', 'logp', 'homo', 'lumo'])
    
    X_matrix, y_cond = [], []
    for _, row in df_clean.iterrows():
        base_experimental_cond = float(row['conductivity_mS_cm'])
        s_mw, s_tpsa = float(row['mw']), float(row['tpsa'])
        s_logp, s_homo, s_lumo = float(row['logp']), float(row['homo']), float(row['lumo'])
        add_pct = float(row['additive_pct'])
        
        add_name = str(row.get('additive_name', 'Quercetin')).split(" ")[0].strip()
        if add_name not in MOLECULAR_GOLDEN_DATABASE: add_name = "Quercetin"
        a_meta = MOLECULAR_GOLDEN_DATABASE[add_name]
        a_mw, a_tpsa, a_logp, a_homo, a_lumo = a_meta["mw"], a_meta["tpsa"], a_meta["logp"], a_meta["homo"], a_meta["lumo"]
        
        # 🌟 恢复标准的宏观 5% 扰动相空间（不需要人工委屈缩减扰动了！）
        for k in range(100):
            perturb = 0.95 + (k / 100.0) * 0.10  
            p_s_mw, p_s_tpsa, p_add_pct = s_mw * perturb, s_tpsa * (2.0 - perturb), add_pct * perturb
            
            feat = build_advanced_13d_features(p_s_mw, p_s_tpsa, s_logp, s_homo, s_lumo, p_add_pct, a_mw, a_tpsa, a_logp, a_homo, a_lumo)
            X_matrix.append(feat)
            
            approx_viscosity_ratio = np.exp(0.0073 * (p_s_mw - s_mw) + 0.0115 * (p_s_tpsa - s_tpsa))
            raw_target_cond = base_experimental_cond / approx_viscosity_ratio
            
            # 🌟 绝杀点：对数变换（Log10 变换），强行压缩指数级非线性波动
            y_cond.append(np.log10(raw_target_cond))
            
    return np.array(X_matrix), np.array(y_cond)

def train_academic_brains():
    print("="*80)
    print(" 🧠 Re-Training 13D Brain with XGBoost + Log-Transform (Scheme B) ...")
    print("="*80)
    X, y_cond = load_pure_data_driven_dataset()
    if X is None: return
    
    # 工业级高轻量化 XGBoost 超参数底座配置
    xgb_config = dict(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    print(f"⏳ Fitting Pure XGBoost Conductivity Engine over {len(X)} samples...")
    cond_brain = XGBRegressor(**xgb_config).fit(X, y_cond)
    
    joblib.dump({"cond_brain": cond_brain}, "ultimate_academic_brain.pkl")
    print("="*80)
    print("🎉 Success! XGBoost Log-Driven Brain Saved Successfully!")
    print("="*80)

if __name__ == "__main__":
    train_academic_brains()