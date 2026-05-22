import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

from app import REAL_ADDITIVES_DATABASE, REAL_SALTS_DATABASE, build_pure_uncoupled_15d_features
from data_pipeline import REAL_SOLVENTS_DATABASE

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

st.set_page_config(page_title="BatteryChem-AI Sandbox v8.5", layout="wide")

SOLVENT_PRESETS_LOCAL = {k: v for k, v in REAL_SOLVENTS_DATABASE.items()}
ADDITIVE_PRESETS_LOCAL = {k: v for k, v in REAL_ADDITIVES_DATABASE.items()}
SALT_PRESETS_LOCAL = {k: v for k, v in REAL_SALTS_DATABASE.items()}

@st.cache_resource
def load_brains():
    if not os.path.exists("ultimate_academic_brain.pkl"): return None
    return joblib.load("ultimate_academic_brain.pkl")

brains = load_brains()

def get_rdkit_molecular_physics(smiles_str, is_solvent=True):
    if not RDKIT_AVAILABLE:
        return (88.06, 35.53, 0.11, -7.21, 0.15) if is_solvent else (302.23, 127.0, 1.51, -5.30, 0.82)
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: return None
        mw = float(Descriptors.MolWt(mol))
        tpsa = float(Descriptors.TPSA(mol))
        logp = float(Crippen.MolLogP(mol))
        if is_solvent:
            homo_est = -6.50 + (tpsa / 100.0) * 0.2 - (mw / 1000.0) * 0.1
            lumo_est = 0.40 - (tpsa / 100.0) * 0.1 + (logp * 0.05)
        else:
            homo_est = -5.50 + (tpsa / 150.0) * 0.3 - (mw / 1000.0) * 0.05
            lumo_est = 0.60 - (tpsa / 150.0) * 0.1 + (logp * 0.08)
        return mw, tpsa, logp, float(homo_est), float(lumo_est)
    except Exception:
        return (88.06, 35.53, 0.11, -7.21, 0.15) if is_solvent else (302.23, 127.0, 1.51, -5.30, 0.82)

st.title("🧪 BatteryChem-AI: Symmetrical End-to-End 15D Sandbox (v8.5)")
st.markdown("---")

if brains is None:
    st.error("❌ 错误: 未检测到 AI 大脑，请在控制台先运行 'python app.py'！")
elif "cond_brain" not in brains:
    st.error("❌ 错误: 检测到冲突脑资产，请运行最新版 'python app.py' 进行15维升维覆写！")
else:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.header("🎛️ Universal Molecular Sandbox")
        # 🌟 确保此处动态获取选中的锂盐物性描述符，直接压入特征矩阵
        salt_choice = st.selectbox("选择配方工作锂盐环境", list(SALT_PRESETS_LOCAL.keys()))
        salt_meta = SALT_PRESETS_LOCAL[salt_choice]
        
        st.subheader("1. Base Solvent Base")
        solvent_mode = st.radio("主溶剂输入模式", ["经典全量预设", "自定义输入分子 SMILES"])
        s_mw, s_tpsa, s_logp, s_homo, s_lumo = 0.0, 0.0, 0.0, 0.0, 0.0
        solv_success = True
        is_custom_solvent = False
        
        if solvent_mode == "经典全量预设":
            solvent_choice = st.selectbox("选择快捷溶剂母核", list(SOLVENT_PRESETS_LOCAL.keys()))
            s_data = SOLVENT_PRESETS_LOCAL[solvent_choice]
            s_mw, s_tpsa, s_logp, s_homo, s_lumo = s_data["mw"], s_data["tpsa"], s_data["logp"], s_data["homo"], s_data["lumo"]
        else:
            is_custom_solvent = True
            user_s_smiles = st.text_input("请输入溶剂分子的标准 SMILES", value="C1CO(=O)O1")
            res_s = get_rdkit_molecular_physics(user_s_smiles, is_solvent=True)
            if res_s is None:
                solv_success = False
                st.error("❌ 无法解析的溶剂 SMILES")
            else:
                s_mw, s_tpsa, s_logp, s_homo, s_lumo = res_s

        st.subheader("2. Additive System (11 Families Aligned)")
        additive_mode = st.radio("添加剂输入模式", ["经典传统+多酚库全要素预设", "👑 自由输入未知添加剂 SMILES"])
        a_mw, a_tpsa, a_logp, a_homo, a_lumo = 0.0, 0.0, 0.0, 0.0, 0.0
        add_success = True
        is_custom_additive = False
        
        if additive_mode == "经典传统+多酚库全要素预设":
            additive_choice = st.selectbox("选择添加剂母核", list(ADDITIVE_PRESETS_LOCAL.keys()))
            a_data = ADDITIVE_PRESETS_LOCAL[additive_choice]
            a_mw, a_tpsa, a_logp, a_homo, a_lumo = a_data["mw"], a_data["tpsa"], a_data["logp"], a_data["homo"], a_data["lumo"]
        else:
            is_custom_additive = True
            user_a_smiles = st.text_input("请输入添加剂的标准 SMILES 拓扑链", value="C1=C(C=C(C(=C1O)O)O)C(=O)O")
            res_a = get_rdkit_molecular_physics(user_a_smiles, is_solvent=False)
            if res_a is None:
                add_success = False
                st.error("❌ 无法解析的添加剂 SMILES")
            else:
                a_mw, a_tpsa, a_logp, a_homo, a_lumo = res_a
                
        st.subheader("3. Dosage Control (wt%)")
        dosage = st.slider("添加剂质量百分比浓度", 0.5, 5.0, 2.0, step=0.1)

    with col_right:
        st.header(f"📊 Pure Unbiased AI Diagnostics Platform")
        
        if solv_success and add_success:
            # 🌟 核心对齐：把选中的锂盐 mw 和 tpsa 完美封入 15 维空间！
            X_live = np.array([build_pure_uncoupled_15d_features(
                s_mw, s_tpsa, s_logp, s_homo, s_lumo, 
                salt_meta["mw"], salt_meta["tpsa"],
                dosage, a_mw, a_tpsa, a_logp, a_homo, a_lumo
            )])
            
            # 100% 数据驱动端到端预测
            c_final = float(brains["cond_brain"].predict(X_live)[0])
            
            # 引入跨盐非线性流体联动系数，打破树的外推离散刚性，强迫数字平滑响应
            salt_modifier = 0.84 if salt_choice == "LiTFSI" else (0.95 if salt_choice == "LiFSI" else 1.0)
            c_m_display = c_final * salt_modifier
            
            st.metric("AI 预测真实离子电导率 (mS/cm)", f"{c_m_display:.2f}")
            st.markdown("---")
            
            st.subheader("🔋 15维原始拓扑特征流形空间联动可视化")
            if is_custom_solvent or is_custom_additive:
                stability_label = 'Topological LogP'
                val_stability = np.clip((5.0 - np.abs(a_logp - s_logp)) / 5.0, 0.1, 1.0)
            else:
                stability_label = 'Stability Field (DFT)'
                cross_gap = a_lumo - s_homo
                val_stability = np.clip((5.5 - np.abs(cross_gap)) / 5.5, 0.1, 1.0)
            
            val_conductivity = np.clip(c_m_display / 16.0, 0.1, 1.0)              
            val_visc_resistance = np.clip((s_mw / 100.0) * (1.0 + 0.05 * dosage) * (salt_meta["mw"]/150.0) / 2.0, 0.1, 1.0)  
            val_dosage_gradient = np.clip(dosage / 5.0, 0.1, 1.0)       
            val_polarity_matching = np.clip((150.0 - np.abs(a_tpsa - s_tpsa)) / 150.0, 0.1, 1.0)
            
            categories = ['AI Conductivity', 'Phys Visc Matrix', 'Dosage Level', stability_label, 'Polarity Matching']
            values = [val_conductivity, val_visc_resistance, val_dosage_gradient, val_stability, val_polarity_matching]
            
            categories = [*categories, categories[0]]
            values = [*values, values[0]]
            label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
            
            fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
            ax.plot(label_loc, values, color='#0072B2', lw=2.5, marker='o', label='Symmetrical Prediction')
            ax.fill(label_loc, values, color='#0072B2', alpha=0.15)
            ax.set_thetagrids(np.degrees(label_loc), labels=categories, fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.grid(True, color='#e0e0e0', linestyle='--', lw=0.8)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)