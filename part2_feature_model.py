"""
Part 2: 特征选择与模型构建
- 皮尔逊相关系数分析
- 递归特征消除(RFE)
- 逻辑回归、随机森林、梯度提升树模型
- 网格搜索+交叉验证参数优化
- Stacking集成学习模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Part 2: 特征选择与模型构建")
print("=" * 60)

# ============================================
# 1. 加载预处理后的数据
# ============================================
print("\n[1.1] 加载数据...")
df = pd.read_csv('output1/03_data_cleaned.csv')
print(f"数据形状: {df.shape}")

target = 'Diabetes_012'
print(f"\n目标变量分布:")
print(df[target].value_counts().sort_index())

X = df.drop(columns=[target])
y = df[target]

# ============================================
# 2. 皮尔逊相关系数分析
# ============================================
print("\n" + "=" * 60)
print("2. 皮尔逊相关系数分析")
print("=" * 60)

corr_with_target = X.corrwith(y).sort_values(ascending=False)
print("\n与目标变量的相关系数:")
print(corr_with_target)

# 保存相关性报告
corr_df = pd.DataFrame({
    '特征': corr_with_target.index,
    '相关系数': corr_with_target.values
})
corr_df.to_csv('output2/01_correlation_analysis.csv', index=False)
print("\n  - 已保存: output2/01_correlation_analysis.csv")

# 选择相关性强的特征 (|r| > 0.05)
strong_corr_features = corr_with_target[abs(corr_with_target) > 0.05].index.tolist()
print(f"\n强相关特征 (|r| > 0.05): {len(strong_corr_features)} 个")

# ============================================
# 3. 递归特征消除 (RFE)
# ============================================
print("\n" + "=" * 60)
print("3. 递归特征消除 (RFE)")
print("=" * 60)

# 使用随机森林进行RFE
rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, class_weight='balanced')
rfe = RFE(estimator=rf_estimator, n_features_to_select=10, step=1)
rfe.fit(X, y)

rfe_selected = X.columns[rfe.support_].tolist()
print(f"\nRFE选出的Top 10特征:")
for i, feat in enumerate(rfe_selected, 1):
    print(f"  {i}. {feat}")

# 保存RFE结果
rfe_df = pd.DataFrame({
    '特征': X.columns,
    '是否选中': rfe.support_
}).sort_values('是否选中', ascending=False)
rfe_df.to_csv('output2/02_rfe_results.csv', index=False)
print("\n  - 已保存: output2/02_rfe_results.csv")

# ============================================
# 4. 综合特征选择
# ============================================
print("\n" + "=" * 60)
print("4. 综合特征选择")
print("=" * 60)

# 使用SelectKBest验证
selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
selector.fit(X, y)
selected_indices = selector.get_support(indices=True)
selectk_features = X.columns[selected_indices].tolist()
print(f"\nSelectKBest Top 15特征:")
print(selectk_features)

# 最终特征集 - 合并多种方法的结果
final_features = list(set(strong_corr_features) | set(selectk_features) | set(rfe_selected))
print(f"\n最终特征数量: {len(final_features)}")
print(f"特征列表: {final_features}")

X_selected = X[final_features]

# 保存特征选择报告
with open('output2/03_feature_selection_report.txt', 'w', encoding='utf-8') as f:
    f.write("特征选择报告\n")
    f.write("=" * 40 + "\n\n")
    f.write("1. 皮尔逊相关系数分析\n")
    f.write(f"   强相关特征: {strong_corr_features}\n\n")
    f.write("2. RFE递归特征消除\n")
    f.write(f"   选中特征: {rfe_selected}\n\n")
    f.write("3. SelectKBest\n")
    f.write(f"   选中特征: {selectk_features}\n\n")
    f.write("4. 最终特征子集\n")
    f.write(f"   {final_features}\n")
print("  - 已保存: output2/03_feature_selection_report.txt")

# ============================================
# 5. 数据分割与标准化
# ============================================
print("\n" + "=" * 60)
print("5. 数据分割与标准化")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape[0]} 样本")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"训练集类别分布:\n{y_train.value_counts().sort_index()}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("数据标准化完成")

# ============================================
# 6. 模型训练与参数优化 + 可视化
# ============================================
print("\n" + "=" * 60)
print("6. 模型训练与参数优化")
print("=" * 60)

results = {}
cv_results_store = {}  # 存储CV结果用于可视化

# 使用分层K折
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 6.1 逻辑回归
print("\n[6.1] 逻辑回归 (Logistic Regression)")
print("-" * 40)

lr_params = {
    'C': [0.01, 0.1, 1],
    'max_iter': [500],
    'class_weight': ['balanced']
}

lr = LogisticRegression(random_state=42, solver='lbfgs')
lr_grid = GridSearchCV(lr, lr_params, cv=cv, scoring='accuracy', n_jobs=-1, return_train_score=True)
lr_grid.fit(X_train_scaled, y_train)

# 存储CV结果
cv_results_store['LogisticRegression'] = pd.DataFrame(lr_grid.cv_results_)

print(f"最佳参数: {lr_grid.best_params_}")
print(f"最佳CV分数: {lr_grid.best_score_:.4f}")

lr_best = lr_grid.best_estimator_

# 使用accuracy作为评分标准
lr_cv_scores = cross_val_score(lr_best, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"交叉验证分数: {lr_cv_scores.mean():.4f} (+/- {lr_cv_scores.std()*2:.4f})")

lr_test_score = lr_best.score(X_test_scaled, y_test)
print(f"测试集分数: {lr_test_score:.4f}")

results['LogisticRegression'] = {
    'model': lr_best,
    'cv_score': lr_cv_scores.mean(),
    'test_score': lr_test_score,
    'best_params': lr_grid.best_params_,
    'cv_results': cv_results_store['LogisticRegression']
}

# 6.2 随机森林
print("\n[6.2] 随机森林 (Random Forest)")
print("-" * 40)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5],
    'class_weight': ['balanced']
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_params, cv=cv, scoring='accuracy', n_jobs=-1, return_train_score=True)
rf_grid.fit(X_train, y_train)

# 存储CV结果
cv_results_store['RandomForest'] = pd.DataFrame(rf_grid.cv_results_)

print(f"最佳参数: {rf_grid.best_params_}")
print(f"最佳CV分数: {rf_grid.best_score_:.4f}")

rf_best = rf_grid.best_estimator_

rf_cv_scores = cross_val_score(rf_best, X_train, y_train, cv=cv, scoring='accuracy')
print(f"交叉验证分数: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std()*2:.4f})")

rf_test_score = rf_best.score(X_test, y_test)
print(f"测试集分数: {rf_test_score:.4f}")

results['RandomForest'] = {
    'model': rf_best,
    'cv_score': rf_cv_scores.mean(),
    'test_score': rf_test_score,
    'best_params': rf_grid.best_params_,
    'cv_results': cv_results_store['RandomForest']
}

# 6.3 梯度提升树
print("\n[6.3] 梯度提升树 (Gradient Boosting)")
print("-" * 40)

gb_params = {
    'n_estimators': [100],
    'max_depth': [5],
    'learning_rate': [0.1],
    'min_samples_split': [5]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=cv, scoring='accuracy', n_jobs=-1, return_train_score=True)
gb_grid.fit(X_train, y_train)

# 存储CV结果
cv_results_store['GradientBoosting'] = pd.DataFrame(gb_grid.cv_results_)

print(f"最佳参数: {gb_grid.best_params_}")
print(f"最佳CV分数: {gb_grid.best_score_:.4f}")

gb_best = gb_grid.best_estimator_

gb_cv_scores = cross_val_score(gb_best, X_train, y_train, cv=cv, scoring='accuracy')
print(f"交叉验证分数: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std()*2:.4f})")

gb_test_score = gb_best.score(X_test, y_test)
print(f"测试集分数: {gb_test_score:.4f}")

results['GradientBoosting'] = {
    'model': gb_best,
    'cv_score': gb_cv_scores.mean(),
    'test_score': gb_test_score,
    'best_params': gb_grid.best_params_,
    'cv_results': cv_results_store['GradientBoosting']
}

# ============================================
# 6.4 参数调优可视化
# ============================================
print("\n" + "=" * 60)
print("6.4 参数调优可视化")
print("=" * 60)

# 逻辑回归参数可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# LR: C参数对性能的影响
lr_cv = cv_results_store['LogisticRegression']
ax = axes[0]
means = lr_cv[lr_cv['param_C'].apply(lambda x: x == 'LR')]['mean_test_score'] if 'param_C' not in lr_cv.columns else None
if 'param_C' in lr_cv.columns:
    c_values = sorted(lr_cv['param_C'].unique(), key=lambda x: float(x))
    mean_scores = [lr_cv[lr_cv['param_C'] == c]['mean_test_score'].mean() for c in c_values]
    ax.plot(c_values, mean_scores, 'bo-', linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('C (Regularization Strength)')
    ax.set_ylabel('CV Accuracy')
    ax.set_title('Logistic Regression: C Parameter Effect')
    ax.grid(True, alpha=0.3)

# LR: 不同C值的boxplot
ax = axes[1]
if 'param_C' in lr_cv.columns:
    data_to_plot = [lr_cv[lr_cv['param_C'] == c]['mean_test_score'].values for c in sorted(lr_cv['param_C'].unique(), key=lambda x: float(x))]
    ax.boxplot(data_to_plot, labels=[str(c) for c in sorted(lr_cv['param_C'].unique(), key=lambda x: float(x))])
    ax.set_xlabel('C Value')
    ax.set_ylabel('CV Accuracy')
    ax.set_title('Logistic Regression: C Parameter Boxplot')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output2/10_lr_param_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output2/10_lr_param_tuning.png")

# 随机森林参数热力图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

rf_cv = cv_results_store['RandomForest']

# n_estimators vs max_depth 热力图
ax = axes[0]
if 'param_n_estimators' in rf_cv.columns and 'param_max_depth' in rf_cv.columns:
    pivot_data = rf_cv.pivot_table(values='mean_test_score', index='param_n_estimators', columns='param_max_depth')
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'CV Accuracy'})
    ax.set_title('Random Forest: n_estimators vs max_depth')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('n_estimators')

# min_samples_split vs max_depth 热力图
ax = axes[1]
if 'param_min_samples_split' in rf_cv.columns and 'param_max_depth' in rf_cv.columns:
    pivot_data = rf_cv.pivot_table(values='mean_test_score', index='param_min_samples_split', columns='param_max_depth')
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'CV Accuracy'})
    ax.set_title('Random Forest: min_samples_split vs max_depth')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('min_samples_split')

plt.tight_layout()
plt.savefig('output2/11_rf_param_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output2/11_rf_param_tuning.png")

# 梯度提升树参数热力图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

gb_cv = cv_results_store['GradientBoosting']

# n_estimators vs learning_rate 热力图
ax = axes[0]
if 'param_n_estimators' in gb_cv.columns and 'param_learning_rate' in gb_cv.columns:
    pivot_data = gb_cv.pivot_table(values='mean_test_score', index='param_n_estimators', columns='param_learning_rate')
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'CV Accuracy'})
    ax.set_title('Gradient Boosting: n_estimators vs learning_rate')
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('n_estimators')

# n_estimators vs max_depth 热力图
ax = axes[1]
if 'param_n_estimators' in gb_cv.columns and 'param_max_depth' in gb_cv.columns:
    pivot_data = gb_cv.pivot_table(values='mean_test_score', index='param_n_estimators', columns='param_max_depth')
    sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'CV Accuracy'})
    ax.set_title('Gradient Boosting: n_estimators vs max_depth')
    ax.set_xlabel('max_depth')
    ax.set_ylabel('n_estimators')

plt.tight_layout()
plt.savefig('output2/12_gb_param_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output2/12_gb_param_tuning.png")

# 综合参数调优对比图
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
model_names_list = ['LogisticRegression', 'RandomForest', 'GradientBoosting']

for idx, model_name in enumerate(model_names_list):
    cv_df = cv_results_store[model_name]
    ax = axes[idx]
    # 按mean_test_score排序并显示top10
    top_results = cv_df.nlargest(10, 'mean_test_score')
    ax.barh(range(len(top_results)), top_results['mean_test_score'].values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(top_results))))
    ax.set_yticks(range(len(top_results)))
    param_strs = []
    for _, row in top_results.iterrows():
        params = {k.replace('param_', ''): v for k, v in row.items() if k.startswith('param_')}
        param_strs.append(str(params)[:30])
    ax.set_yticklabels(param_strs, fontsize=8)
    ax.set_xlabel('CV Accuracy')
    ax.set_title(f'{model_name}: Top 10 Param Combinations')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('output2/13_all_param_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output2/13_all_param_comparison.png")

print("\n参数调优可视化完成!")

# ============================================
# 7. Stacking集成模型
# ============================================
print("\n" + "=" * 60)
print("7. Stacking集成模型")
print("=" * 60)

from sklearn.ensemble import StackingClassifier

# 定义基模型
base_estimators = [
    ('lr', lr_best),
    ('rf', rf_best),
    ('gb', gb_best)
]

# 元学习器 - 使用较简单的模型
meta_learner = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')

stacking = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_learner,
    cv=cv,
    n_jobs=-1,
    passthrough=False
)

print("训练Stacking集成模型...")
stacking.fit(X_train_scaled, y_train)

stacking_cv_scores = cross_val_score(stacking, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"Stacking CV分数: {stacking_cv_scores.mean():.4f} (+/- {stacking_cv_scores.std()*2:.4f})")

stacking_score = stacking.score(X_test_scaled, y_test)
print(f"Stacking测试集分数: {stacking_score:.4f}")

results['Stacking'] = {
    'model': stacking,
    'cv_score': stacking_cv_scores.mean(),
    'test_score': stacking_score
}

# ============================================
# 8. 保存模型结果
# ============================================
print("\n" + "=" * 60)
print("8. 保存模型结果")
print("=" * 60)

# 保存模型比较
comparison = pd.DataFrame({
    '模型': list(results.keys()),
    'CV分数': [r['cv_score'] for r in results.values()],
    '测试集分数': [r['test_score'] for r in results.values()]
}).sort_values('测试集分数', ascending=False)
comparison.to_csv('output2/04_model_comparison.csv', index=False)
print("\n模型比较:")
print(comparison)
print("\n  - 已保存: output2/04_model_comparison.csv")

# 保存最佳模型参数
with open('output2/05_best_params.txt', 'w', encoding='utf-8') as f:
    f.write("模型最佳参数\n")
    f.write("=" * 40 + "\n\n")
    f.write("逻辑回归:\n")
    f.write(str(lr_grid.best_params_) + "\n\n")
    f.write("随机森林:\n")
    f.write(str(rf_grid.best_params_) + "\n\n")
    f.write("梯度提升树:\n")
    f.write(str(gb_grid.best_params_) + "\n\n")
print("  - 已保存: output2/05_best_params.txt")

# 保存预测结果
predictions = {
    'LogisticRegression': lr_best.predict(X_test_scaled),
    'RandomForest': rf_best.predict(X_test),
    'GradientBoosting': gb_best.predict(X_test),
    'Stacking': stacking.predict(X_test_scaled)
}

for name, pred in predictions.items():
    pred_df = pd.DataFrame({'y_true': y_test, 'y_pred': pred})
    pred_df.to_csv(f'output2/06_{name}_predictions.csv', index=False)
print("  - 已保存: output2/06_*_predictions.csv")

# 保存预测概率
probas = {
    'LogisticRegression': lr_best.predict_proba(X_test_scaled),
    'RandomForest': rf_best.predict_proba(X_test),
    'GradientBoosting': gb_best.predict_proba(X_test),
    'Stacking': stacking.predict_proba(X_test_scaled)
}

for name, proba in probas.items():
    proba_df = pd.DataFrame(proba, columns=[f'class_{i}' for i in range(proba.shape[1])])
    proba_df.to_csv(f'output2/06_{name}_probas.csv', index=False)
print("  - 已保存: output2/06_*_probas.csv")

# 保存训练好的特征和模型数据
X_selected.to_csv('output2/07_selected_features.csv', index=False)
print("  - 已保存: output2/07_selected_features.csv")

# 保存scaler
import pickle
with open('output2/08_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("  - 已保存: output2/08_scaler.pkl")

# 保存训练集和测试集索引
train_indices = X_train.index.tolist()
test_indices = X_test.index.tolist()
np.save('output2/09_train_indices.npy', train_indices)
np.save('output2/09_test_indices.npy', test_indices)
print("  - 已保存: output2/09_train/test_indices.npy")

print("\n" + "=" * 60)
print("Part 2 完成!")
print("=" * 60)
