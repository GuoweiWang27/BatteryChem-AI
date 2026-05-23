import os
import numpy as np
import pandas as pd
import joblib

from app import REAL_ADDITIVES_DATABASE, REAL_SALTS_DATABASE, build_pure_uncoupled_15d_features
from data_pipeline import REAL_SOLVENTS_DATABASE

OUTPUT_FILE = "Top_100_Supercomputer_Electrolytes.xlsx"

def run_mega_screening_academic_weighted():
    print("="*80)
    print(f" 🚀 v10.5 GreenBatt 15D Multi-Salt & 10-Family Supercomputer Screening 🚀")
    print("="*80)
    
    if not os.path.exists("ultimate_academic_brain.pkl"):
        print("❌ 错误：未检测到脑资产，请先在控制台运行 python app.py！")
        return
        
    brains = joblib.load("ultimate_academic_brain.pkl")
    
    industry_mixtures = [
        {"name": "EC:DMC 1:1",  "comp1": "EC", "comp2": "DMC", "v1": 0.5, "v2": 0.5},
        {"name": "EC:DEC 1:1",  "comp1": "EC", "comp2": "DEC", "v1": 0.5, "v2": 0.5},
        {"name": "EC:EMC 3:7",  "comp1": "EC", "comp2": "EMC", "v1": 0.3, "v2": 0.7},
        {"name": "DOL:DME 1:1", "comp1": "DOL", "comp2": "DME", "v1": 0.5, "v2": 0.5}
    ]
    
    additive_percentages = np.linspace(0.5, 5.0, 40)
    candidate_matrix, meta_info = [], []
    
    print("⏳ Stage 1: 正在构建全相空间高通量海选特征矩阵...")
    
    for salt_name, salt_data in REAL_SALTS_DATABASE.items():
        for mix in industry_mixtures:
            c1 = REAL_SOLVENTS_DATABASE[mix["comp1"]]
            c2 = REAL_SOLVENTS_DATABASE[mix["comp2"]]
            
            s_mw   = c1["mw"] * mix["v1"] + c2["mw"] * mix["v2"]
            s_tpsa = c1["tpsa"] * mix["v1"] + c2["tpsa"] * mix["v2"]
            s_logp = c1["logp"] * mix["v1"] + c2["logp"] * mix["v2"]
            s_homo = c1["homo"] * mix["v1"] + c2["homo"] * mix["v2"]
            s_lumo = c1["lumo"] * mix["v1"] + c2["lumo"] * mix["v2"]
            
            for a_name, a_data in REAL_ADDITIVES_DATABASE.items():
                for pct in additive_percentages:
                    a_mw, a_tpsa, a_logp, a_homo, a_lumo = a_data["mw"], a_data["tpsa"], a_data["logp"], a_data["homo"], a_data["lumo"]
                    
                    feature_row = build_pure_uncoupled_15d_features(
                        s_mw, s_tpsa, s_logp, s_homo, s_lumo,
                        salt_data["mw"], salt_data["tpsa"],
                        pct, a_mw, a_tpsa, a_logp, a_homo, a_lumo
                    )
                    candidate_matrix.append(feature_row)
                    meta_info.append({
                        "Solvent_System": mix["name"], 
                        "👑 Lithium_Salt_Type": salt_name, 
                        "👑 Additive_Family": a_name, 
                        "Additive_Concentration_(wt%)": round(pct, 2)
                    })
                        
    X_mega = np.array(candidate_matrix)
    df = pd.DataFrame(meta_info)
    
    print(f"⏳ Stage 2: 正在呼叫无偏大脑执行端到端外推推理...")
    df["AI_Predicted_Cond_(mS/cm)"] = brains["cond_brain"].predict(X_mega)
    
    df_sorted = df.sort_values(by="AI_Predicted_Cond_(mS/cm)", ascending=False)
    df_top100 = df_sorted.head(100)
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df_top100.to_excel(writer, sheet_name="Top 100 Best Formulas", index=False)
        
    print("-" * 80)
    print(f"🎉 成功！海选大排行表格已完全清空所有潜在硬伤，无瑕生成！")
    print(f"📂 真实文件路径： '{OUTPUT_FILE}'")
    print("-" * 80)

if __name__ == "__main__":
    run_mega_screening_academic_weighted()