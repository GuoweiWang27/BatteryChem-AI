import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
import shap

# 🌟 绝杀对齐：100% 契合你最新的天然多酚数据管道与 9 维物理增强底座
FEATURE_NAMES = [
    "Solvent_MW", "Solvent_TPSA", "Solvent_LogP", "Solvent_HOMO", "Solvent_LUMO",
    "Additive_Pct", "Additive_Perturb_Field", "Additive_HOMO_Field", "Additive_LUMO_Field"
]

def run_academic_validation():
    print("="*80)
    print(" 📊 BatteryChem-AI: Polyphenol Experimental Evaluation Pipeline v7.0 📊")
    print("="*80)
    
    # 🌟 绝杀修复：优雅接通最新的多酚实验增强数据集
    print("⏳ Loading high-fidelity polyphenol training dataset matrix...")
    from app import load_augmented_experimental_dataset
    X, y_cond, y_wind, y_flash, y_desolv, y_coord = load_augmented_experimental_dataset()
    
    # =====================================================================
    # 📊 落地 Part 2：5折交叉验证（5-Fold Cross-Validation）与残差散点图
    # =====================================================================
    print("\n⏳ [Part 2] Initiating 5-Fold Cross-Validation Academic Exam...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # 将 n_jobs 改为 -1，全核拉满并发，考试速度飙升数倍！
    model_eval = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    
    # 轮流 5 次闭卷大考，收集模型在【真实实验衍生相空间】中的预测值
    y_cond_pred = cross_val_predict(model_eval, X, y_cond, cv=kf)
    
    r2 = r2_score(y_cond, y_cond_pred)
    rmse = np.sqrt(mean_squared_error(y_cond, y_cond_pred))
    
    print(f"  |-- 🎯 Cross-Validation Results for Polyphenol Ionic Conductivity:")
    print(f"  |   ▶ R-squared (R²) Score : {r2:.4f} (Model captures {r2*100:.1f}% of real continuous variance)")
    print(f"  |   ▶ Root Mean Squared Error (RMSE): {rmse:.4f} mS/cm")
    
    # 自动化绘制顶刊标配：Predicted vs. Experimental 残差拟合图
    plt.figure(figsize=(6, 5))
    plt.scatter(y_cond, y_cond_pred, alpha=0.4, color='#2ca02c', edgecolors='k', label=f'Polyphenol Formulations (N={len(X)})')
    plt.plot([y_cond.min(), y_cond.max()], [y_cond.min(), y_cond.max()], 'r--', lw=2, label='Perfect Fit Diagonal')
    plt.title('Polyphenol AI predicted vs. Experimental Benchmark')
    plt.xlabel('True Experimental Benchmark (mS/cm)')
    plt.ylabel('AI Cross-Validated Prediction (mS/cm)')
    plt.text(y_cond.min() + 0.5, y_cond.max() - 1.5, f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.3f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    cv_plot_path = "Academic_Cross_Validation_Residuals.png"
    plt.savefig(cv_plot_path, dpi=300)
    plt.close()
    print(f"  |-- 💾 Residual plot successfully saved: '{cv_plot_path}'")
    
    # =====================================================================
    # 🔬 落地 Part 1：SHAP (Shapley Additive exPlanations) 物理可解释性分析
    # =====================================================================
    print("\n⏳ [Part 1] Initiating Quantum Space SHAP Interpretation Engine...")
    
    # 现场快速拟合一个多酚电化学稳定窗口的大脑
    model_eval.fit(X, y_wind)
    
    # 抽取典型的 500 组异构配方子集进行高维博弈论对齐计算
    sample_idx = np.random.choice(X.shape[0], min(500, X.shape[0]), replace=False)
    X_sample = X[sample_idx]
    
    # 调用博弈论 SHAP 解释器，强行给黑盒模型脱下面纱
    explainer = shap.TreeExplainer(model_eval)
    shap_values = explainer.shap_values(X_sample)
    
    # 自动化绘制学术界最高规格的特征贡献度 summary_plot 散点图
    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_NAMES, show=False)
    plt.title('SHAP Explanations: Feature Contributions to Polyphenol Voltage Window', fontsize=12, pad=20)
    plt.tight_layout()
    
    shap_plot_path = "Academic_SHAP_Feature_Importance.png"
    plt.savefig(shap_plot_path, dpi=300)
    plt.close()
    print(f"  |-- 💾 SHAP contribution plot successfully saved: '{shap_plot_path}'")
    print("-" * 80)
    print("🎉 Academic Upgrades Successful! You are 100% ready for lab presentations!")
    print("=" * 80)

if __name__ == "__main__":
    run_academic_validation()