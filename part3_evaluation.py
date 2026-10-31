"""
Part 3: 模型性能评估与最优模型选择
- 对所有模型进行全面评估
- 准确率、精确率、召回率、F1分数、ROC-AUC
- 选出最优模型并分析特征重要性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Part 3: 模型性能评估与最优模型选择")
print("=" * 60)

# ============================================
# 1. 加载数据和模型
# ============================================
print("\n[1.1] 加载数据和模型...")

# 加载处理后的数据
df = pd.read_csv('output1/03_data_cleaned.csv')
target = 'Diabetes_012'
y = df[target]

# 加载特征选择结果
selected_features = pd.read_csv('output2/07_selected_features.csv').columns.tolist()
X = df[selected_features]

# 加载scaler
with open('output2/08_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# 加载训练/测试集索引确保一致性
train_indices = np.load('output2/09_train_indices.npy', allow_pickle=True).tolist()
test_indices = np.load('output2/09_test_indices.npy', allow_pickle=True).tolist()

# 使用原始索引获取数据
X_train = X.loc[train_indices]
X_test = X.loc[test_indices]
y_train = y.loc[train_indices]
y_test = y.loc[test_indices]

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")

# ============================================
# 2. 重新训练模型获取概率预测
# ============================================
print("\n[2.1] 重新训练模型...")

# 逻辑回归 (使用part2中的最佳参数)
lr_best = LogisticRegression(C=0.01, class_weight='balanced', max_iter=500, random_state=42, solver='lbfgs')
lr_best.fit(X_train_scaled, y_train)

# 随机森林 (使用part2中的最佳参数)
rf_best = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
rf_best.fit(X_train, y_train)

# 梯度提升树 (使用part2中的最佳参数)
gb_best = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                      learning_rate=0.1, min_samples_split=5, random_state=42)
gb_best.fit(X_train, y_train)

# Stacking
base_estimators = [
    ('lr', LogisticRegression(C=0.01, class_weight='balanced', max_iter=500, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                                  class_weight='balanced', random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                      learning_rate=0.1, min_samples_split=5, random_state=42))
]
meta_learner = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
stacking = StackingClassifier(estimators=base_estimators, final_estimator=meta_learner, cv=5, n_jobs=-1)
stacking.fit(X_train_scaled, y_train)

print("模型训练完成!")

# ============================================
# 3. 计算评估指标
# ============================================
print("\n" + "=" * 60)
print("3. 模型性能评估")
print("=" * 60)

def calculate_metrics(y_true, y_pred, y_proba):
    """计算所有评估指标"""
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['Recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['F1'] = f1_score(y_true, y_pred, average='weighted')

    # 多类别ROC-AUC (OvR)
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    try:
        metrics['ROC-AUC'] = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='weighted')
    except:
        metrics['ROC-AUC'] = np.nan

    return metrics

# 存储所有结果
all_metrics = {}
all_predictions = {}
all_probas = {}

model_data = {
    'LogisticRegression': (lr_best, X_test_scaled),
    'RandomForest': (rf_best, X_test),
    'GradientBoosting': (gb_best, X_test),
    'Stacking': (stacking, X_test_scaled)
}

for name, (model, X_test_data) in model_data.items():
    y_pred = model.predict(X_test_data)
    y_proba = model.predict_proba(X_test_data)

    all_predictions[name] = y_pred
    all_metrics[name] = calculate_metrics(y_test.values, y_pred, y_proba)
    all_probas[name] = y_proba

# ============================================
# 4. 结果汇总与可视化
# ============================================
print("\n[4.1] 模型性能汇总...")
print("-" * 60)

# 创建汇总表
metrics_df = pd.DataFrame(all_metrics).T
metrics_df = metrics_df[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']]
metrics_df = metrics_df.sort_values('Accuracy', ascending=False)

print("\n模型性能比较:")
print(metrics_df.round(4))

# 保存评估结果
metrics_df.to_csv('output3/01_model_evaluation_metrics.csv')
print("\n  - 已保存: output3/01_model_evaluation_metrics.csv")

# ============================================
# 5. 混淆矩阵可视化
# ============================================
print("\n[5.1] 生成混淆矩阵...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
model_names = list(all_predictions.keys())

for idx, name in enumerate(model_names):
    y_true = y_test.values
    y_pred = all_predictions[name]
    cm = confusion_matrix(y_true, y_pred)

    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Diab', 'Pre-Diab', 'Diab'],
                yticklabels=['No Diab', 'Pre-Diab', 'Diab'])
    ax.set_title(f'{name} - Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('output3/02_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/02_confusion_matrices.png")

# ============================================
# 6. ROC曲线可视化
# ============================================
print("\n[6.1] 生成ROC曲线...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

for idx, name in enumerate(model_names):
    y_true = y_test.values
    y_proba = all_probas[name]
    y_bin = label_binarize(y_true, classes=[0, 1, 2])

    ax = axes[idx]
    for i, color in zip(range(3), colors):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f'Class {i} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{name} - ROC Curves (OvR)')
    ax.legend(loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('output3/03_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/03_roc_curves.png")

# ============================================
# 7. 模型比较柱状图
# ============================================
print("\n[7.1] 生成模型比较图...")

fig, ax = plt.subplots(figsize=(12, 6))
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
x = np.arange(len(metrics_to_plot))
width = 0.2

for i, name in enumerate(model_names):
    values = [all_metrics[name][m] for m in metrics_to_plot]
    ax.bar(x + i * width, values, width, label=name)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_to_plot)
ax.legend()
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('output3/04_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/04_model_comparison.png")

# ============================================
# 8. 最优模型选择与特征重要性分析
# ============================================
print("\n" + "=" * 60)
print("8. 最优模型选择与特征重要性分析")
print("=" * 60)

# 找出最优模型
best_model_name = metrics_df['Accuracy'].idxmax()
best_score = metrics_df.loc[best_model_name, 'Accuracy']
print(f"\n最优模型: {best_model_name}")
print(f"准确率: {best_score:.4f}")

# 特征重要性分析
print(f"\n[8.1] 特征重要性分析 (所有模型)...")

# 存储所有模型的特征重要性
all_feature_importance = {}
feature_importance_dict = {}

if True:  # Logistic Regression
    feature_importance_dict['LogisticRegression'] = pd.Series(np.abs(lr_best.coef_[0]), index=selected_features)

if True:  # Random Forest
    feature_importance_dict['RandomForest'] = pd.Series(rf_best.feature_importances_, index=selected_features)

if True:  # Gradient Boosting
    feature_importance_dict['GradientBoosting'] = pd.Series(gb_best.feature_importances_, index=selected_features)

# Stacking - 使用各模型的平均
if True:
    feature_importance_dict['Stacking'] = (feature_importance_dict['LogisticRegression'] +
                                          feature_importance_dict['RandomForest'] +
                                          feature_importance_dict['GradientBoosting']) / 3

# 8.1 各模型特征重要性可视化
print("\n[8.1] 生成各模型特征重要性图...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
model_names_fi = ['LogisticRegression', 'RandomForest', 'GradientBoosting', 'Stacking']

for idx, name in enumerate(model_names_fi):
    fi = feature_importance_dict[name].sort_values(ascending=False)
    top_10 = fi.head(10)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_10)))[::-1]
    ax = axes[idx]
    top_10.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel('Importance')
    ax.set_title(f'{name} - Feature Importance (Top 10)')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('output3/09_all_models_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/09_all_models_feature_importance.png")

# 8.2 综合特征重要性对比图
print("\n[8.2] 生成综合特征重要性对比图...")
fig, ax = plt.subplots(figsize=(12, 8))
top_features_set = set()
for name in model_names_fi:
    top_features_set.update(feature_importance_dict[name].sort_values(ascending=False).head(10).index)
top_features_list = list(top_features_set)[:15]

comparison_data = pd.DataFrame({
    name: feature_importance_dict[name][top_features_list].sort_values(ascending=False)
    for name in model_names_fi
})
comparison_data.plot(kind='barh', ax=ax, width=0.8)
ax.set_xlabel('Importance')
ax.set_title('Feature Importance Comparison Across Models')
ax.legend(title='Model', loc='lower right')
plt.tight_layout()
plt.savefig('output3/10_feature_importance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/10_feature_importance_comparison.png")

# 8.3 最优模型特征重要性
best_model_idx = model_names_fi.index(best_model_name)
best_feature_importance = feature_importance_dict[best_model_name].sort_values(ascending=False)

print("\nTop 10 重要特征:")
for i, (feat, imp) in enumerate(best_feature_importance.head(10).items(), 1):
    print(f"  {i}. {feat}: {imp:.4f}")

# 保存特征重要性
fi_df = pd.DataFrame({
    '特征': best_feature_importance.index,
    '重要性': best_feature_importance.values
})
fi_df.to_csv('output3/05_feature_importance.csv', index=False)
print("\n  - 已保存: output3/05_feature_importance.csv")

# 绘制特征重要性图
fig, ax = plt.subplots(figsize=(10, 8))
top_features = best_feature_importance.head(15)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_features)))[::-1]
top_features.plot(kind='barh', ax=ax, color=colors)
ax.set_xlabel('Importance')
ax.set_title(f'{best_model_name} - Feature Importance (Top 15)')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('output3/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/06_feature_importance.png")

# ============================================
# 9. 测试集预测概率矩阵图
# ============================================
print("\n" + "=" * 60)
print("9. 测试集预测概率矩阵可视化")
print("=" * 60)

# 9.1 每个模型的预测概率热力图
print("\n[9.1] 生成各模型预测概率矩阵...")

for name in model_names:
    y_proba = all_probas[name]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：预测概率分布
    ax = axes[0]
    proba_df_viz = pd.DataFrame(y_proba, columns=['P(No Diab)', 'P(Pre-Diab)', 'P(Diab)'])
    for i, col in enumerate(proba_df_viz.columns):
        ax.hist(proba_df_viz[col], bins=50, alpha=0.6, label=col)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} - Prediction Probability Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：样本级别的预测概率热力图（前100个样本）
    ax = axes[1]
    sample_proba = y_proba[:100]
    im = ax.imshow(sample_proba, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Class')
    ax.set_ylabel('Sample Index')
    ax.set_title(f'{name} - Probability Matrix (First 100 Samples)')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['No Diab', 'Pre-Diab', 'Diab'])
    plt.colorbar(im, ax=ax, label='Probability')

    plt.tight_layout()
    plt.savefig(f'output3/11_{name}_probability_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - 已保存: output3/11_{name}_probability_matrix.png")

# 9.2 综合概率矩阵对比图
print("\n[9.2] 生成综合概率矩阵对比...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, name in enumerate(model_names):
    y_proba = all_probas[name]
    ax = axes[idx]

    # 显示前150个样本的概率矩阵
    sample_proba = y_proba[:150]
    im = ax.imshow(sample_proba, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Class')
    ax.set_ylabel('Sample Index')
    ax.set_title(f'{name} - Risk Probability Matrix')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['No Diab', 'Pre-Diab', 'Diab'])
    plt.colorbar(im, ax=ax, label='Probability')

    # 添加真实标签
    true_labels = y_test.values[:150]
    for i, label in enumerate(true_labels):
        ax.plot([2.5, 3.5], [i, i], 'k-', linewidth=2)

plt.tight_layout()
plt.savefig('output3/12_all_models_probability_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/12_all_models_probability_matrix.png")

# 9.3 风险等级分布
print("\n[9.3] 生成风险等级分布图...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, name in enumerate(model_names):
    y_proba = all_probas[name]
    ax = axes[idx]

    # 计算每个样本的最大预测概率对应的风险等级
    max_proba = np.max(y_proba, axis=1)
    pred_class = np.argmax(y_proba, axis=1)

    # 低风险：最大概率<0.4, 中风险：0.4-0.7, 高风险：>0.7
    risk_levels = np.where(max_proba < 0.4, 'Low', np.where(max_proba < 0.7, 'Medium', 'High'))

    risk_df = pd.DataFrame({
        'True Class': y_test.values,
        'Pred Class': pred_class,
        'Max Probability': max_proba,
        'Risk Level': risk_levels
    })

    # 按风险等级和真实标签统计
    risk_counts = risk_df.groupby(['Risk Level', 'True Class']).size().unstack(fill_value=0)
    risk_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#f1c40f', '#e74c3c'])
    ax.set_xlabel('Risk Level')
    ax.set_ylabel('Count')
    ax.set_title(f'{name} - Risk Level Distribution')
    ax.legend(title='True Class', labels=['No Diab', 'Pre-Diab', 'Diab'])
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('output3/13_risk_level_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/13_risk_level_distribution.png")

# 9.4 各模型预测概率对比（箱线图）
print("\n[9.4] 生成预测概率箱线图对比...")

fig, ax = plt.subplots(figsize=(14, 6))
box_data = []
labels = []

for name in model_names:
    y_proba = all_probas[name]
    for i in range(3):
        box_data.append(y_proba[:, i])
        labels.append(f'{name}\nClass {i}')

bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
colors_box = plt.cm.tab10(np.linspace(0, 1, len(box_data)))
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Predicted Probability')
ax.set_title('Prediction Probability Distribution by Model and Class')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('output3/14_probability_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output3/14_probability_boxplot.png")

# ============================================
# 10. 生成最终报告
# ============================================
print("\n[10.1] 生成最终报告...")

with open('output3/07_final_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("糖尿病预测模型 - 最终评估报告\n")
    f.write("=" * 60 + "\n\n")

    f.write("一、模型性能评估\n")
    f.write("-" * 40 + "\n")
    f.write(metrics_df.round(4).to_string())
    f.write("\n\n")

    f.write("二、最优模型选择\n")
    f.write("-" * 40 + "\n")
    f.write(f"最优模型: {best_model_name}\n")
    f.write(f"准确率: {best_score:.4f}\n")
    f.write(f"精确率: {metrics_df.loc[best_model_name, 'Precision']:.4f}\n")
    f.write(f"召回率: {metrics_df.loc[best_model_name, 'Recall']:.4f}\n")
    f.write(f"F1分数: {metrics_df.loc[best_model_name, 'F1']:.4f}\n")
    f.write(f"ROC-AUC: {metrics_df.loc[best_model_name, 'ROC-AUC']:.4f}\n")
    f.write("\n")

    f.write("三、特征重要性分析 (Top 10)\n")
    f.write("-" * 40 + "\n")
    for i, (feat, imp) in enumerate(best_feature_importance.head(10).items(), 1):
        f.write(f"{i}. {feat}: {imp:.4f}\n")
    f.write("\n")

    f.write("四、结论与建议\n")
    f.write("-" * 40 + "\n")
    f.write(f"1. 在所有模型中，{best_model_name} 表现最优\n")
    f.write(f"2. 准确率达到 {best_score:.2%}\n")
    top5 = ', '.join(best_feature_importance.head(5).index.tolist())
    f.write(f"3. 最重要的特征包括: {top5}\n")
    f.write("4. 建议使用该模型进行糖尿病风险预测\n")

print("  - 已保存: output3/07_final_report.txt")

# 保存完整分类报告
with open('output3/08_classification_reports.txt', 'w', encoding='utf-8') as f:
    for name in model_names:
        y_true = y_test.values
        y_pred = all_predictions[name]
        report = classification_report(y_true, y_pred, target_names=['No Diab', 'Pre-Diab', 'Diab'])
        f.write(f"\n{'='*40}\n{name}\n{'='*40}\n")
        f.write(report)

print("  - 已保存: output3/08_classification_reports.txt")

print("\n" + "=" * 60)
print("Part 3 完成!")
print("=" * 60)
