import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor  # 引入提升树结构
import shap

FEATURE_NAMES = [
    "Solvent_MW", "Solvent_TPSA", "Solvent_LogP", "Solvent_HOMO", "Solvent_LUMO",
    "Additive_MW", "Additive_TPSA", "Additive_LogP", "Additive_HOMO", "Additive_LUMO",
    "Derived_Cross_Gap", "Derived_Inv_Viscosity", "Derived_Dielectric_Field"
]

def run_academic_validation():
    print("="*80)
    print(" 📊 BatteryChem-AI: XGBoost Log-Inverse Validation Pipeline 📊")
    print("="*80)
    
    from app import load_pure_data_driven_dataset
    X, y_log_cond = load_pure_data_driven_dataset()
    if X is None: return
    
    # 手动建立 5 折闭卷大考循环，以便在逆变换下计算最清白的 R2
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_true_physical = 10 ** y_log_cond  # 还原出真金白银的实验观测值
    y_pred_physical = np.zeros_like(y_true_physical)
    
    print("⏳ [Part 2] Initiating 5-Fold Cross-Validation with Log-Inverse Correction...")
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_log_cond[train_idx], y_log_cond[val_idx]
        
        # 实例化完全同源的 XGBoost 考试大脑
        model = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # 预测出对数空间的值后，现场通过 10** 逆变换强行映射回宏观物理电导率
        y_pred_log = model.predict(X_val)
        y_pred_physical[val_idx] = 10 ** y_pred_log
        
    r2 = r2_score(y_true_physical, y_pred_physical)
    rmse = np.sqrt(mean_squared_error(y_true_physical, y_pred_physical))
    
    print(f"  |-- 🎯 Final Restored Physical Space Cross-Validation Results:")
    print(f"  |   ▶ R-squared (R²) Score : {r2:.4f} (XGBoost Decoded Flow!)")
    print(f"  |   ▶ Root Mean Squared Error (RMSE): {rmse:.4f} mS/cm")
    
    # 绘图
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true_physical, y_pred_physical, alpha=0.5, color='#9467bd', edgecolors='k', label=f'Formulations (N={len(X)})')
    plt.plot([y_true_physical.min(), y_true_physical.max()], [y_true_physical.min(), y_true_physical.max()], 'r--', lw=2, label='Perfect Fit Diagonal')
    plt.title('XGBoost + Log-Inverse Transform Performance')
    plt.xlabel('True Experimental Dataset (mS/cm)')
    plt.ylabel('AI Restored Prediction (mS/cm)')
    plt.text(y_true_physical.min() + 0.5, y_true_physical.max() - 1.5, f'$R^2$ = {r2:.4f}\nRMSE = {rmse:.3f}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("Academic_Cross_Validation_Residuals.png", dpi=300)
    plt.close()
    
    # =====================================================================
    # 🔬 Part 1: SHAP 解释
    # =====================================================================
    print("\n⏳ [Part 1] Running TreeSHAP Explainer for XGBoost Brain...")
    model_eval = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    model_eval.fit(X, y_log_cond)
    
    sample_idx = np.random.choice(X.shape[0], min(500, X.shape[0]), replace=False)
    X_sample = X[sample_idx]
    
    explainer = shap.TreeExplainer(model_eval)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(11, 6))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_NAMES, show=False)
    plt.title('XGBoost SHAP: Feature Contributions to Log-Conductivity', fontsize=11, pad=20)
    plt.tight_layout()
    plt.savefig("Academic_SHAP_Feature_Importance.png", dpi=300)
    plt.close()
    print("🎉 [通关!] XGBoost and Log-Inverse Evaluation Complete!")

if __name__ == "__main__":
    run_academic_validation()