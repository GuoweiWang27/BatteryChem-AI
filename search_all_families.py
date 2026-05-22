import os
import numpy as np
import pandas as pd
import joblib
from app import POLYPHENOL_GOLDEN_DATABASE, build_pure_uncoupled_13d_features
from data_pipeline import PURE_SOLVENT_COMPONENTS

OUTPUT_FILE = "Top_100_Supercomputer_Electrolytes.xlsx"

def run_mega_screening_academic_weighted(selected_salt="LiPF6"):
    print("="*80)
    print(f" 🚀 v8.5 GreenBatt Log-Free Physical Screening Engine 🚀")
    print("="*80)
    
    if not os.path.exists("ultimate_academic_brain.pkl"): return
    brains = joblib.load("ultimate_academic_brain.pkl")
    
    # 动态构建二元经典工业基底
    industry_mixtures = [
        {"name": "EC:DMC 1:1",  "comp1": "EC", "comp2": "DMC", "v1": 0.5, "v2": 0.5},
        {"name": "EC:DEC 1:1",  "comp1": "EC", "comp2": "DEC", "v1": 0.5, "v2": 0.5},
        {"name": "EC:EMC 3:7",  "comp1": "EC", "comp2": "EMC", "v1": 0.3, "v2": 0.7},
        {"name": "DOL:DME 1:1", "comp1": "DOL", "comp2": "DME", "v1": 0.5, "v2": 0.5}
    ]
    
    additive_percentages = np.linspace(0.5, 5.0, 40)
    candidate_matrix, meta_info = [], []
    
    for mix in industry_mixtures:
        c1 = PURE_SOLVENT_COMPONENTS[mix["comp1"]]
        c2 = PURE_SOLVENT_COMPONENTS[mix["comp2"]]
        
        # 严格执行体积分数加权平均，消灭特征断层
        s_mw   = c1["mw"] * mix["v1"] + c2["mw"] * mix["v2"]
        s_tpsa = c1["tpsa"] * mix["v1"] + c2["tpsa"] * mix["v2"]
        s_logp = c1["logp"] * mix["v1"] + c2["logp"] * mix["v2"]
        s_homo = c1["homo"] * mix["v1"] + c2["homo"] * mix["v2"]
        s_lumo = c1["lumo"] * mix["v1"] + c2["lumo"] * mix["v2"]
        
        for a_name, a_data in POLYPHENOL_GOLDEN_DATABASE.items():
            for pct in additive_percentages:
                a_mw, a_tpsa, a_logp, a_homo, a_lumo = a_data["mw"], a_data["tpsa"], a_data["logp"], a_data["homo"], a_data["lumo"]
                feature_row = build_pure_uncoupled_13d_features(s_mw, s_tpsa, s_logp, s_homo, s_lumo, pct, a_mw, a_tpsa, a_logp, a_homo, a_lumo)
                candidate_matrix.append(feature_row)
                meta_info.append({
                    "Solvent_System": mix["name"], "Lithium_Salt": selected_salt, 
                    "Polyphenol_Additive": a_name, "Additive_Concentration_(wt%)": round(pct, 2)
                })
                        
    X_mega = np.array(candidate_matrix)
    df = pd.DataFrame(meta_info)
    
    # 多目标并行推理，直接输出宏观物理尺度
    outputs = brains["multi_brain"].predict(X_mega)
    df["AI_Predicted_Cond_(mS/cm)"] = outputs[:, 0]
    
    # 纯客观大排行，不加任何主观伪权重系数
    df_sorted = df.sort_values(by="AI_Predicted_Cond_(mS/cm)", ascending=False)
    df_top100 = df_sorted.head(100)
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df_top100.to_excel(writer, sheet_name="Top 100 Best Formulas", index=False)
    print(f"🎉 成功！完全自洽的黄金海选大排行表格已落盘 -> '{OUTPUT_FILE}'")

if __name__ == "__main__":
    run_mega_screening_academic_weighted()