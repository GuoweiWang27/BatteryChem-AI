import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# 🛡️ 引入 RDKit 核心化学信息学组件，用于现场肢解并计算任意分子的结构指纹
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# 设置高规格宽屏学术布局
st.set_page_config(page_title="BatteryChem-AI Sandbox", layout="wide")

# 固化主溶剂快捷选项
SOLVENT_PRESETS = {
    "EC (Carbonate)":  {"mw": 88.06,  "tpsa": 35.53, "logp": -0.38, "homo": -7.21, "lumo": 0.45},
    "DMC (Carbonate)": {"mw": 90.08,  "tpsa": 26.30, "logp": -0.42, "homo": -6.85, "lumo": 0.72},
    "EMC (Carbonate)": {"mw": 104.11, "tpsa": 26.30, "logp": -0.15, "homo": -6.78, "lumo": 0.68},
    "DEC (Carbonate)": {"mw": 118.13, "tpsa": 26.30, "logp": 0.12,  "homo": -6.72, "lumo": 0.65},
    "PC (Carbonate)":  {"mw": 102.09, "tpsa": 35.53, "logp": -0.22, "homo": -7.15, "lumo": 0.42}
}

POLYPHENOL_OPTIONS = {
    "Quercetin (槲皮素)":        {"homo": -5.30, "lumo": 0.82},
    "Catechin (儿茶素)":         {"homo": -5.45, "lumo": 0.78},
    "Epigallocatechin (表儿茶素)": {"homo": -5.35, "lumo": 0.80},
    "Gallic_Acid (没食子酸)":      {"homo": -6.40, "lumo": 0.15},
    "Resveratrol (白藜芦醇)":     {"homo": -5.20, "lumo": 0.88}
}

@st.cache_resource
def load_brains():
    if not os.path.exists("ultimate_academic_brain.pkl"):
        return None
    return joblib.load("ultimate_academic_brain.pkl")

brains = load_brains()

def predict_with_bootstrap(model, X_in):
    """ 提取随机森林里 100 棵决策树的独立投票，现场推演 90% 物理置信边界 """
    all_tree_preds = np.array([tree.predict(X_in)[0] for tree in model.estimators_])
    p_mean = np.mean(all_tree_preds)
    p_lower = np.percentile(all_tree_preds, 5)
    p_upper = np.percentile(all_tree_preds, 95)
    return p_mean, p_lower, p_upper

# =====================================================================
# 🎨 前端 UI 渲染展示层
# =====================================================================
st.title("🧪 BatteryChem-AI: High-Throughput Polyphenol Discovery Platform")
st.markdown("---")

if brains is None:
    st.error("❌ 错误: 未检测到训练好的 AI 大脑（ultimate_academic_brain.pkl）。请先在控制台运行 'python app.py'！")
else:
    # 划分左右控制台
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.header("🎛️ Sandbox Molecular Designer")
        st.subheader("1. Base Solvent Select")
        
        # 🌟 核心升级：增加一个“自定义输入”的切换开关
        solvent_mode = st.radio("主溶剂输入模式", ["经典快捷预设", "自定义输入分子 SMILES (RDKit 实时解析)"])
        
        s_mw, s_tpsa, s_logp, s_homo, s_lumo = 0.0, 0.0, 0.0, 0.0, 0.0
        input_success = True
        
        if solvent_mode == "经典快捷预设":
            solvent_choice = st.selectbox("选择快捷溶剂母核", list(SOLVENT_PRESETS.keys()))
            s_data = SOLVENT_PRESETS[solvent_choice]
            s_mw, s_tpsa, s_logp, s_homo, s_lumo = s_data["mw"], s_data["tpsa"], s_data["logp"], s_data["homo"], s_data["lumo"]
        else:
            # 开启终极自定义自由捏脸模式
            user_smiles = st.text_input("请输入溶剂分子的标准 SMILES 字符串", value="C1COC(=O)O1", help="例如: DMC输入 COC(=O)OC，EA输入 CC(=O)OCC")
            
            if not RDKIT_AVAILABLE:
                st.warning("⚠️ 警告: 检测到当前本地环境未安装 RDKit 库。请在 Anaconda 窗口运行 `pip install rdkit` 以便现场解析拓扑特征！")
                # 未安装时进行安全的静态兜底降级
                s_mw, s_tpsa, s_logp, s_homo, s_lumo = 88.06, 35.53, -0.38, -7.21, 0.45
            else:
                try:
                    # 🚀 RDKit 现场拆解分子拓扑结构并进行第一性原理物性硬算！
                    mol = Chem.MolFromSmiles(user_smiles)
                    if mol is None:
                        st.error("❌ 无法解析的 SMILES 字符串，请检查输入格式是否有误。")
                        input_success = False
                    else:
                        s_mw = float(Descriptors.MolWt(mol))
                        s_tpsa = float(Descriptors.TPSA(mol))
                        s_logp = float(Descriptors.MolLogP(mol))
                        # HOMO/LUMO 采用经典非共轭碳酸酯结构能级偏置进行线性经验映射映射
                        s_homo = -6.5 - 0.01 * s_tpsa
                        s_lumo = 0.5 + 0.002 * s_mw
                        
                        # 极其拉满学术自豪感的实时物理特征公示板
                        st.success("🎉 RDKit 现场解析成功！")
                        st.info(f"🧬 **实时特征指纹**: MW={s_mw:.2f} | TPSA={s_tpsa:.2f} | LogP={s_logp:.2f} | HOMO(est)={s_homo:.2f}")
                except Exception as e:
                    st.error(f"解析报错: {e}")
                    input_success = False

        st.subheader("2. Green Additive Select")
        additive_choice = st.selectbox("选择复配多酚添加剂", list(POLYPHENOL_OPTIONS.keys()))
        a_data = POLYPHENOL_OPTIONS[additive_choice]
        
        st.subheader("3. Dosage Control (wt%)")
        dosage = st.slider("添加剂质量百分比浓度", 0.5, 5.0, 2.0, step=0.1)
        
        # 实时拼装 9 维输入矩阵并准备喂给随机森林
        if input_success:
            X_live = np.array([[
                float(s_mw), float(s_tpsa), float(s_logp), float(s_homo), float(s_lumo),
                float(dosage), float(dosage) * 0.1, float(s_homo) - 0.5, float(s_lumo) + 0.5
            ]])

    with col_right:
        st.header("📊 Real-time AI Multi-Objective Diagnostics")
        
        if input_success:
            # 调用随机森林现场多目标轰鸣预测
            c_m, c_l, c_u = predict_with_bootstrap(brains["cond_brain"], X_live)
            w_m, w_l, w_u = predict_with_bootstrap(brains["wind_brain"], X_live)
            f_m, f_l, f_u = predict_with_bootstrap(brains["flash_brain"], X_live)
            d_m, d_l, d_u = predict_with_bootstrap(brains["desolv_brain"], X_live)
            co_m, co_l, co_u = predict_with_bootstrap(brains["coord_brain"], X_live)
            
            # 渲染误差波动仪表盘
            m1, m2, m3 = st.columns(3)
            m1.metric("AI 传质电导率 (mS/cm)", f"{c_m:.2f}", f"90% Range: [{c_l:.2f} ~ {c_u:.2f}]", delta_color="off")
            m2.metric("AI 电化学稳定窗口 (V)", f"{w_m:.2f}", f"90% Range: [{w_l:.2f} ~ {w_u:.2f}]", delta_color="off")
            m3.metric("AI 经验安全闪点 (°C)", f"{f_m:.1f}", f"90% Range: [{f_l:.1f} ~ {f_u:.1f}]", delta_color="off")
            
            st.markdown("---")
            
            # 渲染带阴影环的置信度雷达图
            categories = ['Conductivity', 'Voltage Window', 'Flash Point', 'Desolvation Barrier', 'Coordination Field']
            values_mean = [c_m/18.0, w_m/6.0, f_m/150.0, d_m/100.0, co_m/100.0]
            values_lower = [c_l/18.0, w_l/6.0, f_l/150.0, d_l/100.0, co_l/100.0]
            values_upper = [c_u/18.0, w_u/6.0, f_u/150.0, d_u/100.0, co_u/100.0]
            
            categories = [*categories, categories[0]]
            values_mean = [*values_mean, values_mean[0]]
            values_lower = [*values_lower, values_lower[0]]
            values_upper = [*values_upper, values_upper[0]]
            
            label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories))
            
            fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
            ax.plot(label_loc, values_mean, label='AI Predicted Mode', color='#1f77b4', lw=2)
            ax.fill(label_loc, values_mean, color='#1f77b4', alpha=0.1)
            ax.plot(label_loc, values_lower, color='#7f7f7f', linestyle='--', lw=1, alpha=0.7)
            ax.plot(label_loc, values_upper, color='#7f7f7f', linestyle='--', lw=1, alpha=0.7)
            ax.fill_between(label_loc, values_lower, values_upper, color='#7f7f7f', alpha=0.15, label='90% Bootstrap Confidence Band')
            
            ax.set_thetagrids(np.degrees(label_loc), labels=categories)
            ax.set_ylim(0, 1.1)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            st.pyplot(fig)
        else:
            st.info("💡 请在左侧输入合法正确的化学分子 SMILES 结构式以唤醒后台诊断矩阵。")