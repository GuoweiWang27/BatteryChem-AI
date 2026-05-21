import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import shap

# 100% 对齐 5.91/6.0 版本的全量特征底座
FEATURE_NAMES = [
    "Solvent_MW", "Solvent_TPSA", "Solvent_LogP", "Solvent_HOMO", "Solvent_LUMO",
    "Additive_MW", "Additive_TPSA", "Additive_LogP", "Additive_HOMO", "Additive_LUMO",
    "F_Count", "N_Count", "S_Count"
]

def run_academic_validation():
    print("="*80)
    print(" 📊 BatteryChem-AI: Part 1 & Part 2 Academic Evaluation Pipeline 📊")
    print("="*80)
    
    # 从你的 app.py 顺畅提取那组 56,250 样本的工业基础数据
    print("⏳ Loading physical training dataset matrix...")
    from app import generate_industrial_dataset
    raw_data = generate_industrial_dataset()
    
    X = np.array([item[0] for item in raw_data])
    y_cond = np.array([item[1] for item in raw_data]) # 拿最具代表性的离子电导率做大考
    y_wind = np.array([item[2] for item in raw_data]) # 拿电压窗口做大考
    
    # =====================================================================
    # 📊 落地【第二点】：5折交叉验证（5-Fold Cross-Validation）与残差散点图
    # =====================================================================
    print("\n⏳ [Part 2] Initiating 5-Fold Cross-Validation Academic Exam...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model_eval = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=1)
    
    # 模拟真实世界大考：轮流 5 次闭卷考试，收集全量预测值
    y_cond_pred = cross_val_predict(model_eval, X, y_cond, cv=kf)
    
    # 计算学术界最认可的两个核心数学指标
    r2 = r2_score(y_cond, y_cond_pred)
    rmse = np.sqrt(mean_squared_error(y_cond, y_cond_pred))
    
    print(f"  |-- 🎯 Cross-Validation Results for Ionic Conductivity:")
    print(f"  |   ▶ R-squared (R²) Score : {r2:.4f} (Model captures {r2*100:.1f}% of continuous variance)")
    print(f"  |   ▶ Root Mean Squared Error (RMSE): {rmse:.4f} mS/cm")
    
    # 自动化绘制顶刊标配：Predicted vs. Experimental 残差拟合图
    plt.figure(figsize=(6, 5))
    plt.scatter(y_cond, y_cond_pred, alpha=0.3, color='#1f77b4', label=f'Candidate Formulations (N={len(X)})')
    plt.plot([y_cond.min(), y_cond.max()], [y_cond.min(), y_cond.max()], 'r--', lw=2, label='Perfect Fit Diagonal')
    plt.title('AI Predicted vs. Experimental Benchmark (Ionic Conductivity)')
    plt.xlabel('True Physical Benchmark (mS/cm)')
    plt.ylabel('AI Cross-Validated Prediction (mS/cm)')
    plt.text(y_cond.min() + 1, y_cond.max() - 2, f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.3f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    cv_plot_path = "Academic_Cross_Validation_Residuals.png"
    plt.savefig(cv_plot_path, dpi=300)
    plt.close()
    print(f"  |-- 💾 Residual plot successfully saved to local directory: '{cv_plot_path}'")
    
    # =====================================================================
    # 🔬 落地【第一点】：SHAP (Shapley Additive exPlanations) 物理可解释性分析
    # =====================================================================
    print("\n⏳ [Part 1] Initiating Quantum Space SHAP Interpretation Engine...")
    
    # 现场快速拟合一个解释器大脑
    model_eval.fit(X, y_wind) # 用电压窗口(Voltage Window)来剖析化学键的秘密
    
    # 抽取典型的 500 组异构配方子集进行高维博弈论对齐计算（防止内存溢出）
    sample_idx = np.random.choice(X.shape[0], 500, replace=False)
    X_sample = X[sample_idx]
    
    # 调用 SHAP 树状解释器
    explainer = shap.TreeExplainer(model_eval)
    shap_values = explainer.shap_values(X_sample)
    
    # 自动化绘制学术界最高规格的特征贡献度 summary_plot 散点图
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_NAMES, show=False)
    plt.title('SHAP Explanations: Feature Contributions to Electrochemical Voltage Window', fontsize=12, pad=20)
    plt.tight_layout()
    
    shap_plot_path = "Academic_SHAP_Feature_Importance.png"
    plt.savefig(shap_plot_path, dpi=300)
    plt.close()
    print(f"  |-- 💾 SHAP contribution plot successfully saved: '{shap_plot_path}'")
    print("-" * 80)
    print("🎉 Academic Part 1 & Part 2 upgrade completed with stellar visual outputs!")
    print("=" * 80)

if __name__ == "__main__":
    run_academic_validation()