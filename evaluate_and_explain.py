#!/usr/bin/env python3
"""
evaluate_and_explain.py — 15维全要素纯数据驱动统计学无偏交叉验证大考 (完全体 v10.5)
=============================================================================
100% 格式化并对齐无公式自循环泄露的 15 维真实物理特征流形。
隔离 15% 封闭盲测集，吐出最严谨的双色残差拟合图与第一性原理 SHAP 归因解释图。
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
import shap

# 🌟 从核心大脑中同步导入统一的超参配置与数据加载管道
from app import XGB_GLOBAL_CONFIG, load_pure_large_scale_physical_dataset

# 🌟 核心重构：将特征工程字典全面升维对齐 15 维完备物理指纹流
FEATURE_NAMES = [
    "Solvent_MW", "Solvent_TPSA", "Solvent_LogP", "Solvent_HOMO", "Solvent_LUMO",
    "Salt_MW", "Salt_TPSA",  # 🚀 新增：主工作锂盐的第一性原理拓扑指纹
    "Additive_MW", "Additive_TPSA", "Additive_LogP", "Additive_HOMO", "Additive_LUMO",
    "Dosage_Level", "Delta_Structure_MW", "Quantum_Cross_Gap"
]

def run_academic_validation():
    print("="*80)
    print(" 📊 BatteryChem-AI: 15-Dimensional Unbiased Validation Pipeline 📊")
    print("="*80)
    
    # 加载 100% 剔除了人工公式污染的纯驱动数据集
    X, y_physical_cond = load_pure_large_scale_physical_dataset()
    if X is None or len(X) == 0:
        print("❌ 错误：未检测到规范的数据集，请先运行 python data_pipeline.py！")
        return
    
    # 🔒 强行在外部隔离 15% 完全不参与训练的封闭 Holdout 独立盲测集，用于验证真实外推尊严
    X_cv_train, X_holdout, y_cv_train, y_holdout = train_test_split(
        X, y_physical_cond, test_size=0.15, random_state=42, shuffle=True
    )
    
    # 构建严格的 5 折交叉验证洗牌器
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_r2_scores, fold_rmse_scores = [], []
    y_pred_all_cv = np.zeros_like(y_cv_train)
    fold_idx = 1
    
    print("⏳ Stage 1: 正在 15 维高维空间中进行 5-Fold 交叉验证闭卷大考...")
    for train_idx, val_idx in kf.split(X_cv_train):
        X_tr, X_val = X_cv_train[train_idx], X_cv_train[val_idx]
        y_tr, y_val = y_cv_train[train_idx], y_cv_train[val_idx]
        
        # 100% 锁死单源超参
        model = XGBRegressor(**XGB_GLOBAL_CONFIG)
        model.fit(X_tr, y_tr)
        
        pred_physical = model.predict(X_val)
        y_pred_all_cv[val_idx] = pred_physical
        
        fold_r2_scores.append(r2_score(y_val, pred_physical))
        fold_rmse_scores.append(np.sqrt(mean_squared_error(y_val, pred_physical)))
        fold_idx += 1
        
    mean_r2, std_r2 = np.mean(fold_r2_scores), np.std(fold_r2_scores)
    mean_rmse, std_rmse = np.mean(fold_rmse_scores), np.std(fold_rmse_scores)
    
    # 盲测考卷投递
    final_brain = XGBRegressor(**XGB_GLOBAL_CONFIG)
    final_brain.fit(X_cv_train, y_cv_train)
    holdout_pred = final_brain.predict(X_holdout)
    final_holdout_r2 = r2_score(y_holdout, holdout_pred)
    
    print("⏳ Stage 2: 正在绘制纯净高斯数据分布下的双色学术拟合残差图...")
    plt.figure(figsize=(6, 5))
    # 蓝色散点：交叉验证拟合流形
    plt.scatter(y_cv_train, y_pred_all_cv, alpha=0.4, color='#1f77b4', edgecolors='k', label='5-Fold CV Pool')
    # 红色三角：15% 绝对隔离的未知材料盲测点
    plt.scatter(y_holdout, holdout_pred, alpha=0.7, color='#d62728', marker='^', edgecolors='k', label='🔒 Isolated Test')
    
    # 绘制理想对角线
    all_min = min(y_cv_train.min(), y_pred_all_cv.min()) - 0.5
    all_max = max(y_cv_train.max(), y_pred_all_cv.max()) + 0.5
    plt.plot([all_min, all_max], [all_min, all_max], 'r--', lw=2, label='Perfect Fit')
    
    plt.title('15D Equation-Free Residuals Space')
    plt.xlabel('True Instrument Value (mS/cm)')
    plt.ylabel('AI Directly Predicted Value (mS/cm)')
    
    stat_text = f'CV $R^2$: {mean_r2:.3f} $\pm$ {std_r2:.3f}\nCV RMSE: {mean_rmse:.2f}\nTest $R^2$: {final_holdout_r2:.3f}'
    plt.text(all_min + 0.5, all_max - 3.5, stat_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.85))
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    output_img1 = "Academic_Cross_Validation_Residuals.png"
    plt.savefig(output_img1, dpi=300)
    plt.close()
    
    print("⏳ Stage 3: 正在呼叫 SHAP 基因组归因解释器，透视 15 维物理特征的隐式贡献...")
    explainer = shap.TreeExplainer(final_brain)
    shap_values = explainer.shap_values(X_cv_train)
    
    plt.figure(figsize=(11, 6))
    shap.summary_plot(shap_values, X_cv_train, feature_names=FEATURE_NAMES, show=False)
    plt.tight_layout()
    
    output_img2 = "Academic_SHAP_Feature_Importance.png"
    plt.savefig(output_img2, dpi=300)
    plt.close()
    
    print("-" * 80)
    print(f"🎉 恭喜！15维全要素大考验证与归因图表完美吐出，无公式污染！")
    print(f"📂 散点图已安全落盘 -> '{output_img1}'")
    print(f"📂 可解释性 SHAP 图已落盘 -> '{output_img2}'")
    print("-" * 80)

if __name__ == "__main__":
    run_academic_validation()