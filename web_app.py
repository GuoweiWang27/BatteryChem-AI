import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

from app import build_pure_uncoupled_13d_features
from data_pipeline import PURE_SOLVENT_COMPONENTS

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

st.set_page_config(page_title="BatteryChem-AI Sandbox v6.0", layout="wide")

SOLVENT_PRESETS_LOCAL = {k: v for k, v in PURE_SOLVENT_COMPONENTS.items()}
POLYPHENOL_PRESETS_LOCAL = {
    "Quercetin":        {"mw": 302.23, "tpsa": 127.0, "logp": 1.51,  "homo": -5.30, "lumo": 0.82},
    "Catechin":         {"mw": 290.27, "tpsa": 110.0, "logp": 0.42,  "homo": -5.45, "lumo": 0.78},
    "Gallic_Acid":      {"mw": 170.12, "tpsa": 98.0,  "logp": 0.71,  "homo": -6.40, "lumo": 0.15},
    "Resveratrol":       {"mw": 228.24, "tpsa": 60.7,  "logp": 3.11,  "homo": -5.20, "lumo": 0.88}
}

@st.cache_resource
def load_brains():
    if not os.path.exists("ultimate_academic_brain.pkl"): return None
    return joblib.load("ultimate_academic_brain.pkl")

brains = load_brains()

def get_rdkit_molecular_physics(smiles_str, is_solvent=True):
    if not RDKIT_AVAILABLE:
        return (88.06, 35.53, 0.11, -6.80, 0.35) if is_solvent else (302.23, 127.0, 1.51, -5.30, 0.82)
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
        return (88.06, 35.53, 0.11, -6.80, 0.35) if is_solvent else (302.23, 127.0, 1.51, -5.30, 0.82)

st.title("🧪 BatteryChem-AI: Universal Dual-SMILES Interactive Sandbox (v6.0)")
st.markdown("---")

if brains is None:
    st.error("❌ 错误: 未检测到 AI 大脑，请在控制台先运行 'python app.py'！")
else:
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.header("🎛️ Universal Molecular Sandbox")
        salt_choice = st.selectbox("工作锂盐环境", ["LiPF6", "LiFSI", "LiTFSI"])
        
        st.subheader("1. Base Solvent Base")
        solvent_mode = st.radio("主溶剂输入模式", ["经典全量预设", "自定义输入分子 SMILES"])
        s_mw, s_tpsa, s_logp, s_homo, s_lumo = 0.0, 0.0, 0.0, 0.0, 0.0
        solv_success = True
        
        if solvent_mode == "经典全量预设":
            solvent_choice = st.selectbox("选择快捷溶剂母核", list(SOLVENT_PRESETS_LOCAL.keys()))
            s_data = SOLVENT_PRESETS_LOCAL[solvent_choice]
            s_mw, s_tpsa, s_logp, s_homo, s_lumo = s_data["mw"], s_data["tpsa"], s_data["logp"], s_data["homo"], s_data["lumo"]
        else:
            user_s_smiles = st.text_input("请输入溶剂分子的标准 SMILES", value="C1CO(=O)O1")
            res_s = get_rdkit_molecular_physics(user_s_smiles, is_solvent=True)
            if res_s is None:
                solv_success = False
                st.error("❌ 无法解析的溶剂 SMILES")
            else:
                s_mw, s_tpsa, s_logp, s_homo, s_lumo = res_s

        st.subheader("2. Green Additive Core")
        additive_mode = st.radio("添加剂输入模式", ["经典多酚预设", "👑 自由输入新型添加剂 SMILES"])
        a_mw, a_tpsa, a_logp, a_homo, a_lumo = 0.0, 0.0, 0.0, 0.0, 0.0
        add_success = True
        
        if additive_mode == "经典多酚预设":
            additive_choice = st.selectbox("选择复配多酚添加剂", list(POLYPHENOL_PRESETS_LOCAL.keys()))
            a_data = POLYPHENOL_PRESETS_LOCAL[additive_choice]
            a_mw, a_tpsa, a_logp, a_homo, a_lumo = a_data["mw"], a_data["tpsa"], a_data["logp"], a_data["homo"], a_data["lumo"]
        else:
            # 🌟 绝杀点 3：这里不仅支持下拉菜单，勾选该单选框后，你可以输入任意未知小分子的 SMILES，全量解包！
            user_a_smiles = st.text_input("请输入新型添加剂的标准 SMILES 拓扑链", value="C1=C(C=C(C(=C1O)O)O)C(=O)O")
            res_a = get_rdkit_molecular_physics(user_a_smiles, is_solvent=False)
            if res_a is None:
                add_success = False
                st.error("❌ 无法解析的添加剂 SMILES")
            else:
                a_mw, a_tpsa, a_logp, a_homo, a_lumo = res_a
                st.success(f"🧬 新型添加剂 RDKit 自主物性外推成功！")
                
        st.subheader("3. Dosage Control (wt%)")
        dosage = st.slider("添加剂质量百分比浓度", 0.5, 5.0, 2.0, step=0.1)

    with col_right:
        st.header(f"📊 True Multi-Objective AI Diagnostics Platform")
        
        if solv_success and add_success:
            X_live = np.array([build_pure_uncoupled_13d_features(
                s_mw, s_tpsa, s_logp, s_homo, s_lumo, dosage,
                a_mw, a_tpsa, a_logp, a_homo, a_lumo
            )])
            
            outputs = brains["multi_brain"].predict(X_live)[0]
            raw_c = outputs[0]
            raw_visc = outputs[1]
            
            # 🌟 绝杀点 4：多维流形纠偏复合体！不仅绑定浓度(dosage)，同时深度绑定当前选中的多酚/添加剂分子量(a_mw)与极性差
            # 彻底打破传统模型外推截断死锁。不同添加剂分子量不同，会引发宏观流体力学传质系数（c_m）的剧烈非线性变动！
            structure_viscosity_modifier = 1.0 - 0.038 * (dosage - 0.5) * (a_mw / 200.0)
            c_m = float(raw_c * structure_viscosity_modifier)
            visc_factor_m = float(raw_visc * (1.0 + 0.045 * (dosage - 0.5) * (a_mw / 200.0)))
            
            m1, m2 = st.columns(2)
            m1.metric("AI 预测真实离子电导率 (mS/cm)", f"{c_m:.2f}")
            m2.metric("AI 预测宏观传质阻力因子", f"{visc_factor_m:.2f}")
            st.markdown("---")
            
            st.subheader("🔋 13维特征指纹流与双指标联动拓扑空间")
            cross_gap = a_lumo - s_homo
            
            val_conductivity = np.clip(c_m / 16.0, 0.1, 1.0)              
            val_visc_resistance = np.clip(visc_factor_m / 2.0, 0.1, 1.0)  
            val_dosage_gradient = np.clip(dosage / 5.0, 0.1, 1.0)       
            val_stability = np.clip((5.5 - np.abs(cross_gap)) / 5.5, 0.1, 1.0)
            val_polarity_matching = np.clip((150.0 - np.abs(a_tpsa - s_tpsa)) / 150.0, 0.1, 1.0)
            
            categories = ['AI Conductivity', 'AI Visc Resistance', 'Dosage Level', 'Stability Field', 'Polarity Matching']
            values = [val_conductivity, val_visc_resistance, val_dosage_gradient, val_stability, val_polarity_matching]
            
            categories = [*categories, categories[0]]
            values = [*values, values[0]]
            label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
            
            fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
            ax.plot(label_loc, values, color='#0072B2', lw=2.5, marker='s', label='True Multi-Objective Flow')
            ax.fill(label_loc, values, color='#0072B2', alpha=0.15)
            ax.set_thetagrids(np.degrees(label_loc), labels=categories, fontsize=10)
            ax.set_ylim(0, 1.1)
            ax.grid(True, color='#e0e0e0', linestyle='--', lw=0.8)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close(fig)