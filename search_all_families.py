import os
import numpy as np
import pandas as pd
import joblib

# 🌟 从升级后的大脑和管道中动态对齐 15 维全要素特征指纹构造器
from app import REAL_ADDITIVES_DATABASE, REAL_SALTS_DATABASE, build_pure_uncoupled_15d_features
from data_pipeline import REAL_SOLVENTS_DATABASE # 🚀 修复点：修改为最新的标准溶剂资产库，防止数据打架

OUTPUT_FILE = "Top_100_Supercomputer_Electrolytes.xlsx"

def run_mega_screening_academic_weighted():
    print("="*80)
    print(f" 🚀 v9.0 GreenBatt 15D Multi-Salt & 10-Family Supercomputer Screening 🚀")
    print("="*80)
    
    if not os.path.exists("ultimate_academic_brain.pkl"):
        print("❌ 错误：未检测到 15D 脑资产，请先在控制台运行 python app.py！")
        return
        
    brains = joblib.load("ultimate_academic_brain.pkl")
    
    # 工业标准经典二元混合溶剂基底
    industry_mixtures = [
        {"name": "EC:DMC 1:1",  "comp1": "EC", "comp2": "DMC", "v1": 0.5, "v2": 0.5},
        {"name": "EC:DEC 1:1",  "comp1": "EC", "comp2": "DEC", "v1": 0.5, "v2": 0.5},
        {"name": "EC:EMC 3:7",  "comp1": "EC", "comp2": "EMC", "v1": 0.3, "v2": 0.7},
        {"name": "DOL:DME 1:1", "comp1": "DOL", "comp2": "DME", "v1": 0.5, "v2": 0.5}
    ]
    
    # 0.5wt% 到 5.0wt% 的高密度连续浓度切片
    additive_percentages = np.linspace(0.5, 5.0, 40)
    candidate_matrix, meta_info = [], []
    
    print("⏳ Stage 1: 正在全量构建 [3大工业锂盐 × 10大要素添加剂 × 4大经典配方] 的非线性交叉海选矩阵...")
    
    # 🌟 引爆笛卡尔积交叉海选，让 3 种锂盐与 10 种全要素添加剂在混合溶剂里全面合流
    for salt_name, salt_data in REAL_SALTS_DATABASE.items():
        for mix in industry_mixtures:
            c1 = REAL_SOLVENTS_DATABASE[mix["comp1"]] # 🚀 修正点：对齐完全自洽的溶剂数据源
            c2 = REAL_SOLVENTS_DATABASE[mix["comp2"]]
            
            # 计算主溶剂的热力学体积分数加权描述符
            s_mw   = c1["mw"] * mix["v1"] + c2["mw"] * mix["v2"]
            s_tpsa = c1["tpsa"] * mix["v1"] + c2["tpsa"] * mix["v2"]
            s_logp = c1["logp"] * mix["v1"] + c2["logp"] * mix["v2"]
            s_homo = c1["homo"] * mix["v1"] + c2["homo"] * mix["v2"]
            s_lumo = c1["lumo"] * mix["v1"] + c2["lumo"] * mix["v2"]
            
            for a_name, a_data in REAL_ADDITIVES_DATABASE.items():
                for pct in additive_percentages:
                    a_mw, a_tpsa, a_logp, a_homo, a_lumo = a_data["mw"], a_data["tpsa"], a_data["logp"], a_data["homo"], a_data["lumo"]
                    
                    # 🌟 100% 锁死并对齐 app.py 的 15 维完全体特征指纹流
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
    
    print(f"⏳ Stage 2: 正在呼叫全新 15D 无偏大脑对全空间 {len(X_mega)} 组配方执行高通量推理...")
    raw_preds = brains["cond_brain"].predict(X_mega)
    
    # 完美融入源头训练集的连续流体力学衰减效应，实现训练-预测高度对称
    adjusted_preds = []
    for i, row in enumerate(meta_info):
        a_meta = REAL_ADDITIVES_DATABASE[row["👑 Additive_Family"]]
        dosage = row["Additive_Concentration_(wt%)"]
        salt_name = row["👑 Lithium_Salt_Type"]
        
        salt_visc_factor = 1.3 if salt_name == "LiTFSI" else (1.1 if salt_name == "LiFSI" else 1.0)
        modifier = 1.0 - 0.038 * (dosage - 0.5) * (a_meta["mw"] / 200.0) * salt_visc_factor
        adjusted_preds.append(float(raw_preds[i] * modifier))
        
    df["AI_Predicted_Cond_(mS/cm)"] = adjusted_preds
    
    # 纯客观大排行：筛选出综合性能最顶尖的 Top 100 黄金配方
    df_sorted = df.sort_values(by="AI_Predicted_Cond_(mS/cm)", ascending=False)
    df_top100 = df_sorted.head(100)
    
    # 采用 xlsxwriter 干净利落落盘，杜绝过载
    with pd.ExcelWriter(OUTPUT_FILE, engine='xlsxwriter') as writer:
        df_top100.to_excel(writer, sheet_name="Top 100 Best Formulas", index=False)
        
    print("-" * 80)
    print(f"🎉 成功！15D 超算高高通量海选表格已完全清除报错、无暇生成！")
    print(f"📂 真实文件已安全落盘 -> 桌面的 AI4S 文件夹内： '{OUTPUT_FILE}'")
    print("-" * 80)

if __name__ == "__main__":
    run_mega_screening_academic_weighted()