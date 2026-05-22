import os
import numpy as np
import pandas as pd
import joblib
from app import MOLECULAR_GOLDEN_DATABASE

OUTPUT_FILE = "Top_100_Supercomputer_Electrolytes.xlsx"

def run_mega_screening(selected_salt="LiPF6"):
    print("="*80)
    print(f" 🚀 v8.1 GreenBatt XGBoost screening Engine ({selected_salt}) 🚀")
    print("="*80)
    
    if not os.path.exists("ultimate_academic_brain.pkl"): return
    brains = joblib.load("ultimate_academic_brain.pkl")
    
    solvents = {k: v for k, v in MOLECULAR_GOLDEN_DATABASE.items() if v["is_solvent"]}
    additives = {k: v for k, v in MOLECULAR_GOLDEN_DATABASE.items() if not v["is_solvent"]}
    additive_percentages = np.linspace(0.5, 5.0, 40)
    
    candidate_matrix, meta_info = [], []
    for s_name, s_data in solvents.items():
        for a_name, a_data in additives.items():
            for pct in additive_percentages:
                cross_gap = float(a_data["lumo"]) - float(s_data["homo"])
                combined_mass = float(s_data["mw"]) * 0.9 + float(a_data["mw"]) * (float(pct) / 100.0)
                approx_viscosity = 0.1 * np.exp((0.0073 * combined_mass + 0.0115 * float(s_data["tpsa"])))
                inv_viscosity = 1.0 / approx_viscosity
                dielectric_field = (float(s_data["tpsa"]) + float(a_data["tpsa"]) * (float(pct) / 100.0)) / (float(s_data["mw"]) + 1e-5)
                
                feature_row = [
                    float(s_data["mw"]), float(s_data["tpsa"]), float(s_data["logp"]), float(s_data["homo"]), float(s_data["lumo"]),
                    float(a_data["mw"]), float(a_data["tpsa"]), float(a_data["logp"]), float(a_data["homo"]), float(a_data["lumo"]),
                    cross_gap, inv_viscosity, dielectric_field
                ]
                candidate_matrix.append(feature_row)
                meta_info.append({"Solvent_System": s_name, "Lithium_Salt": selected_salt, "Polyphenol_Additive": a_name, "Additive_Concentration_(wt%)": round(pct, 2)})
                        
    X_mega = np.array(candidate_matrix)
    df = pd.DataFrame(meta_info)
    
    # 🚀 预测结果进行 10** 逆变换
    p_log_cond = brains["cond_brain"].predict(X_mega)
    df["AI_Predicted_Cond_(mS/cm)"] = 10 ** p_log_cond
    
    df_sorted = df.sort_values(by="AI_Predicted_Cond_(mS/cm)", ascending=False)
    df_top100 = df_sorted.head(100)
    
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df_top100.to_excel(writer, sheet_name="Top 100 Best Formulas", index=False)
    print(f"🎉 Success! Top 100 table saved: '{OUTPUT_FILE}'")

if __name__ == "__main__":
    run_mega_screening()