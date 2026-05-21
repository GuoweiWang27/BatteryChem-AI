import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

MODEL_FILE = "ultimate_academic_brain.pkl"
OUTPUT_FILE = "Top_100_Supercomputer_Electrolytes.xlsx"

# 🌟 设置网页的高大上学术主题与宽屏布局
st.set_page_config(page_title="BatteryChem-AI Platform", page_icon="🔋", layout="wide")

st.title("🪐 BatteryChem-AI: High-Throughput Screening & Discovery Platform")
st.markdown("---")

# 检查大脑是否存在
if not os.path.exists(MODEL_FILE):
    st.error("❌ Error: 'ultimate_academic_brain.pkl' not found. Please run 'python app.py' first to train the AI Brain!")
else:
    # 异步加载 AI 大脑与核心数据库
    brains = joblib.load(MODEL_FILE)
    from app import SOLVENT_DATABASE, ADDITIVE_DATABASE
    
    # =====================================================================
    # 🔬 1. 网页左侧：【双模自由切换】材料设计沙盒
    # =====================================================================
    st.sidebar.header("🔬 Custom Molecule Sandbox")
    st.sidebar.markdown("Configure core molecular descriptors to query the multi-objective AI Brain.")

    # 核心升级：允许用户在“预设母核组合”与“纯手动自由输入未知分子参数”之间一键切换
    input_mode = st.sidebar.radio(
        "Input Domain Mode (输入模式选择):", 
        ["Use Database Presets (使用库内预设)", "Custom Molecule Input (纯手动输入量化参数)"]
    )

    if input_mode == "Use Database Presets (使用库内预设)":
        selected_sol = st.sidebar.selectbox("Select Core Solvent Base:", list(SOLVENT_DATABASE.keys()))
        selected_add = st.sidebar.selectbox("Select Functional Additive:", list(ADDITIVE_DATABASE.keys()))
        sol_mod = st.sidebar.selectbox("Solvent Substituent Modifier:", ["None", "Methyl", "Fluorine", "Methoxy"])
        add_mod = st.sidebar.selectbox("Additive Substituent Modifier:", ["None", "Methyl", "Fluorine", "Nitrile", "Sulfone"])
        h_clone = st.sidebar.slider("Halogen Loading Index 微扰系数:", 1.0, 1.2, 1.0, step=0.1)
        
        # 调用 6.0 版本的特征突变逻辑
        from search_all_families import mutate_features
        b_sol = SOLVENT_DATABASE[selected_sol]
        b_add = ADDITIVE_DATABASE[selected_add]
        
        s_mw, s_tpsa, s_logp, s_homo, s_lumo, _, _, _ = mutate_features(b_sol["mw"], b_sol["tpsa"], b_sol["logp"], b_sol["homo"], b_sol["lumo"], 0, 0, 0, sol_mod)
        a_mw, a_tpsa, a_logp, a_homo, a_lumo, a_f, a_n, a_s = mutate_features(b_add["mw"], b_add["tpsa"], b_add["logp"], b_add["homo"], b_add["lumo"], b_add["f"], b_add["n"], b_add["s"], add_mod)
        
        a_mw *= h_clone
        a_lumo *= (2.0 - h_clone)

    else:
        # 🌟 自由输入模式：用户可以输入任意在实验室新合成/量子化学模拟出的未知分子参数
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧪 Novel Solvent Properties")
        s_mw = st.sidebar.number_input("Solvent Molecular Weight (分子量):", min_value=30.0, max_value=250.0, value=90.0)
        s_tpsa = st.sidebar.number_input("Solvent TPSA (极性表面积):", min_value=0.0, max_value=150.0, value=26.3)
        s_logp = st.sidebar.number_input("Solvent LogP (脂水分配系数):", min_value=-3.0, max_value=5.0, value=-0.4)
        s_homo = st.sidebar.number_input("Solvent HOMO (eV):", min_value=-10.0, max_value=-2.0, value=-6.5)
        s_lumo = st.sidebar.number_input("Solvent LUMO (eV):", min_value=-2.0, max_value=3.0, value=0.5)

        st.sidebar.markdown("---")
        st.sidebar.subheader("🧪 Novel Additive Properties")
        a_mw = st.sidebar.number_input("Additive Molecular Weight (分子量):", min_value=30.0, max_value=300.0, value=106.0)
        a_tpsa = st.sidebar.number_input("Additive TPSA (极性表面积):", min_value=0.0, max_value=150.0, value=35.5)
        a_logp = st.sidebar.number_input("Additive LogP (脂水分配系数):", min_value=-3.0, max_value=5.0, value=-0.15)
        a_homo = st.sidebar.number_input("Additive HOMO (eV):", min_value=-10.0, max_value=-2.0, value=-7.1)
        a_lumo = st.sidebar.number_input("Additive LUMO (eV):", min_value=-3.0, max_value=3.0, value=-0.2)
        
        st.sidebar.markdown("🧪 Functional Atom Counts")
        a_f = st.sidebar.slider("Fluorine (F) Atom Count:", 0, 5, 1)
        a_n = st.sidebar.slider("Nitrile (N) Atom Count:", 0, 3, 0)
        a_s = st.sidebar.slider("Sulfur/Sulfone (S) Atom Count:", 0, 2, 0)

    # 统一将两模输入的特征，打包进大矩阵馈送给机器学习模型
    combined_input = np.array([[s_mw, s_tpsa, s_logp, s_homo, s_lumo, a_mw, a_tpsa, a_logp, a_homo, a_lumo, a_f, a_n, a_s]])
    
    # 实时调用 5 大回归大脑进行毫秒级串行预测
    pred_cond = brains["cond_brain"].predict(combined_input)[0]
    pred_wind = brains["wind_brain"].predict(combined_input)[0]
    pred_flash = brains["flash_brain"].predict(combined_input)[0]
    pred_desolv = brains["desolv_brain"].predict(combined_input)[0]
    pred_coord = brains["coord_brain"].predict(combined_input)[0]
    
    # 计算综合材料学评分
    material_score = (pred_cond * 5.0) + (pred_wind * 18.0) + (pred_flash * 0.25) - (pred_desolv * 0.12)
    
    # 微观界面成膜机理逻辑映射
    sei_component = "聚碳酸酯/烷基锂富集型有机 SEI 膜"
    if a_f >= 1.5: sei_component = "高致密无机 LiF 富集型 SEI 膜 (强力抗高压、抗锂枝晶)"
    elif a_s >= 1.0: sei_component = "含硫高导电无机盐富集型 SEI 膜 (加速离子传质)"
    elif a_n >= 1.0: sei_component = "含氮正极自钝化过渡金属保护 CEI 膜"
    
    # =====================================================================
    # 📊 2. 网页右侧主面板：【实时指标仪表盘】+【Plotly 交互雷达图】
    # =====================================================================
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚡ Real-Time Material Performance Metrics")
        
        # 实时渲染硬核数据标签
        st.metric(label="AI Predicted Ionic Conductivity (离子电导率)", value=f"{pred_cond:.3f} mS/cm")
        st.metric(label="AI Predicted Voltage Window (电化学工作窗口)", value=f"{pred_wind:.3f} V")
        st.metric(label="AI Predicted Thermal Flash Point (安全闪点阈值)", value=f"{pred_flash:.1f} °C")
        st.metric(label="AI Predicted Desolvation Barrier (去溶剂化能垒)", value=f"{pred_desolv:.2f} kJ/mol")
        
        st.info(f"🔮 **Predicted Microscopic Interphase Chemistry (微观成膜预测):** \n{sei_component}")
        st.success(f"🏆 **Global Comprehensive Fitness Score (综合材料学积分):** {material_score:.2f}")

    with col2:
        st.subheader("📊 Dynamic Multi-Objective Pareto Radar")
        
        # 构建 Plotly 高级网页交互型动态雷达
        metrics = ['Conductivity', 'Voltage Window', 'Flash Point', 'Fast-Charging', 'Coordination']
        # 针对当前的沙盒参数进行归一化映射，拉开雷达张力
        norm_vals = [
            min(1.0, max(0.1, pred_cond / 12.0)),
            min(1.0, max(0.1, pred_wind / 6.0)),
            min(1.0, max(0.1, pred_flash / 180.0)),
            min(1.0, max(0.1, (100.0 - pred_desolv) / 80.0)), # 反转能垒，使其越大越好
            min(1.0, max(0.1, pred_coord / 100.0))
        ]
        norm_vals += norm_vals[:1]
        categories = metrics + [metrics[0]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=norm_vals, theta=categories, fill='toself',
            name='Current Sandbox Blend', line_color='#d62728', fillcolor='rgba(214, 39, 40, 0.2)'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=False)),
            showlegend=False, margin=dict(l=40, r=40, t=40, b=40), height=380
        )
        st.plotly_chart(fig, use_container_width=True)

    # =====================================================================
    # 🪐 3. 网页底部：【一键高通量海选白名单交互展示大厅】
    # =====================================================================
    st.markdown("---")
    st.subheader("🪐 Million-Scale Virtual Screening Data Matrix Hub")
    
    if st.button("🚀 Execute High-Throughput Screening Pipeline (一键扫描全域相空间)"):
        with st.spinner("Supercomputer algorithms parsing phase space loops linearly..."):
            # 如果检测到本地还没生成 Excel 表格，会自动现场调用海选引擎秒级生成
            if not os.path.exists(OUTPUT_FILE):
                from search_all_families import run_mega_screening
                run_mega_screening()
            
            # 读取 Excel 并直观展示在网页前端，支持滑动和排序
            df_display = pd.read_excel(OUTPUT_FILE)
            st.dataframe(df_display.head(50), use_container_width=True)
            st.balloons() # 放飞庆祝成功的彩色气球！
            st.success("🎉 Top 50 Academic Champions unrolled successfully below! Use the generated Excel for physical bench validation.")