import os
import numpy as np
import pandas as pd
import joblib

MODEL_FILE = "ultimate_academic_brain.pkl"
OUTPUT_FILE = "Top_100_Supercomputer_Electrolytes.xlsx"

# =====================================================================
# 🔬 1. 纯英文核心映射池 (与 app.py 100% 绝对对齐)
# =====================================================================
SOLVENTS_POOL = {
    "COC(=O)OC": "DMC", "CCOC(=O)OCC": "DEC", "C1COCO1": "DOL", 
    "COCCO": "DME", "C1COC(=O)O1": "EC", "CCC(=O)OC": "MP", 
    "CCC(=O)OCC": "EP", "CC1DOC(=O)O1": "PC", "O=C(OC)OCC": "EMC"
}

ADDITIVE_POOL = {
    "FC1COC(=O)O1": "FEC", "O=C1C=CO1": "VC", "O=S1(=O)CCO1": "PS", 
    "O=S1(=O)CCCO1": "BS", "CCS(=O)(=O)C": "EMS", "CS(=O)(=O)C": "DMS", 
    "FS(=O)(=O)F": "SO2F2", "N#CCCC#N": "SN", "N#CCCCCC#N": "ADN", 
    "CC#N": "AN", "N#CC=CC#N": "DCB", "O=S1(=O)CC(F)O1": "FPS", 
    "O=S1(=O)CC(CF3)O1": "TFPS", "FC(F)(F)CO": "TFE", "O=S1(=O)OCCO1": "DTD",
    "Oc1cc(O)cc(C)c1": "TP", "Oc1ccc(O)cc1": "HQ", "Oc1c(O)cccc1": "CC"
}

def mutate_features(base_mw, base_tpsa, base_logp, base_homo, base_lumo, base_f, base_n, base_s, modifier):
    mw, tpsa, logp, homo, lumo, f, n, s = base_mw, base_tpsa, base_logp, base_homo, base_lumo, base_f, base_n, base_s
    if modifier == "Methyl": mw += 15.03; logp += 0.50; homo += 0.05; lumo += 0.02
    elif modifier == "Methoxy": mw += 31.03; tpsa += 9.23; logp += 0.15; homo += 0.20; lumo += 0.05
    elif modifier == "Fluorine": mw += 18.99; logp += 0.14; homo -= 0.15; lumo -= 0.60; f += 1
    elif modifier == "Nitrile": mw += 25.02; tpsa += 23.79; logp -= 0.30; homo -= 0.20; lumo -= 0.80; n += 1
    elif modifier == "Sulfone": mw += 64.06; tpsa += 34.44; logp -= 0.80; homo -= 0.10; lumo -= 1.10; s += 1
    return mw, tpsa, logp, homo, lumo, f, n, s

# =====================================================================
# 🚀 3. 百万级多目标【大矩阵向量化】超算筛选引擎 (Version 6.0 闪电破局)
# =====================================================================
def run_mega_screening():
    print("="*80)
    print(" 🪐 BatteryChem-AI: 百万级高通量虚拟筛选系统 6.0 (大矩阵向量化超算版) 🪐")
    print("="*80)
    
    if not os.path.exists(MODEL_FILE):
        print("❌ 错误：未检测到大脑模型，请先运行一次 app.py！")
        return
        
    brains = joblib.load(MODEL_FILE)
    from app import SOLVENT_DATABASE, ADDITIVE_DATABASE
    
    sol_modifiers = ["None", "Methyl", "Fluorine", "Methoxy"]
    add_modifiers = ["None", "Methyl", "Fluorine", "Nitrile", "Sulfone"]
    halogen_clones = [1.0, 1.1, 1.2]
    
    # 构建内存超速缓冲区
    input_features_list = []
    meta_info_list = []
    
    print("⏳ 第一阶段：正在闪电组装 10,800 组高维特征相空间大矩阵...")
    
    for sol_smiles, sol_name in SOLVENTS_POOL.items():
        for add_smiles, add_name in ADDITIVE_POOL.items():
            b_sol = SOLVENT_DATABASE[sol_smiles]
            b_add = ADDITIVE_DATABASE[add_smiles]
            
            for s_mod in sol_modifiers:
                for a_mod in add_modifiers:
                    for h_clone in halogen_clones:
                        if add_smiles == "CCO": continue
                        
                        s_mw, s_tpsa, s_logp, s_homo, s_lumo, _, _, _ = mutate_features(b_sol["mw"], b_sol["tpsa"], b_sol["logp"], b_sol["homo"], b_sol["lumo"], 0, 0, 0, s_mod)
                        a_mw, a_tpsa, a_logp, a_homo, a_lumo, a_f, a_n, a_s = mutate_features(b_add["mw"], b_add["tpsa"], b_add["logp"], b_add["homo"], b_add["lumo"], b_add["f"], b_add["n"], b_add["s"], a_mod)
                        
                        a_mw *= h_clone
                        a_lumo *= (2.0 - h_clone)
                        
                        # 压入矩阵池
                        input_features_list.append([s_mw, s_tpsa, s_logp, s_homo, s_lumo, a_mw, a_tpsa, a_logp, a_homo, a_lumo, a_f, a_n, a_s])
                        
                        # 暂存人类可读元标签
                        sol_label = f"{sol_name}-{s_mod}" if s_mod != "None" else sol_name
                        add_label = f"{add_name}-{a_mod}(x{h_clone:.1f})" if a_mod != "None" else f"{add_name}(x{h_clone:.1f})"
                        meta_info_list.append((sol_label, add_label, a_f, a_s, a_n))
                        
    # 💥 降维打击的核心：把列表转化为一个整块的 NumPy 连续内存大矩阵
    X_all = np.array(input_features_list)
    print(f"  |-- 成功组装结构大矩阵！矩阵形状 (Shape): {X_all.shape}，正在移交 AI 核心...")
    
    # ⏳ 第二阶段：让 5 大回归大脑各自进行【仅仅 1 次】全量离散大矩阵吞噬（用 C++ 速度并行解决！）
    print("⏳ 第二阶段：超级AI大脑正在启动矩阵群全量一次性闪电吞噬...")
    all_cond = brains["cond_brain"].predict(X_all)
    all_wind = brains["wind_brain"].predict(X_all)
    all_flash = brains["flash_brain"].predict(X_all)
    all_desolv = brains["desolv_brain"].predict(X_all)
    all_coord = brains["coord_brain"].predict(X_all)
    
    print("⏳ 第三阶段：回归数值大对齐，正在批量编译双语学术白名单...")
    records = []
    
    # 仅在最后组装表格时做一次 O(N) 线性遍历，速度极快
    for i in range(len(meta_info_list)):
        sol_label, add_label, a_f, a_s, a_n = meta_info_list[i]
        pred_cond = all_cond[i]
        pred_wind = all_wind[i]
        pred_flash = all_flash[i]
        pred_desolv = all_desolv[i]
        pred_coord = all_coord[i]
        
        sei_component = "聚碳酸酯/烷基锂富集型有机 SEI 膜"
        if a_f >= 1.5: sei_component = "高致密无机 LiF 富集型 SEI 膜 (强力抗高压、抗锂枝晶)"
        elif a_s >= 1.0: sei_component = "含硫高导电无机盐富集型 SEI 膜 (加速离子传质)"
        elif a_n >= 1.0: sei_component = "含氮正极自钝化过渡金属保护 CEI 膜"
        
        material_score = (pred_cond * 5.0) + (pred_wind * 18.0) + (pred_flash * 0.25) - (pred_desolv * 0.12)
        
        if pred_cond >= 7.0 and pred_wind >= 3.5 and pred_flash >= 100: scientific_rating = "🏆 满分顶刊神仙配方"
        elif pred_wind >= 4.2 and "LiF" in sei_component: scientific_rating = "🛡️ 超高电压耐受型配方"
        elif pred_desolv <= 42.0 and pred_cond >= 9.5: scientific_rating = "⚡ 极速快充动力学配方"
        else: scientific_rating = "⚠️ 性能表现平庸配方"
        
        records.append({
            "Solvent_Name": sol_label, "Additive_Name": add_label,
            "AI_Cond_(mS/cm)": round(pred_cond, 3), "AI_Voltage_Window_(V)": round(pred_wind, 3),
            "AI_Flash_Point_(°C)": round(pred_flash, 1), "AI_Desolvation_(kJ/mol)": round(pred_desolv, 2),
            "Coordination_Index": round(pred_coord, 1), "Microscopic_SEI_Component": sei_component,
            "Comprehensive_Fitness_Score": round(material_score, 2), "AI4S_Scientific_Rating": scientific_rating
        })
        
    df_results = pd.DataFrame(records).sort_values(by="Comprehensive_Fitness_Score", ascending=False).reset_index(drop=True)
    df_results.head(100).to_excel(OUTPUT_FILE, index=False, sheet_name="Top 100 Best Formulas")
    
    print("-" * 80)
    print(f"🎉 终极完全胜利！矩阵向量化高通量虚拟筛选全闭环圆满成功！")
    print(f"💾 前沿大排行 Excel 报表已在本地毫秒级生成：'{os.path.abspath(OUTPUT_FILE)}'")
    print("=" * 80)

if __name__ == "__main__":
    import sys; sys.path.append(os.getcwd())
    run_mega_screening()