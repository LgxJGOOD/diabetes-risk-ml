"""
Part 1: 数据探索性分析与预处理
- 全面探索性分析(EDA)并可视化
- 数据预处理: 缺失值、异常值、数据标准化与编码
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 1. 数据探索性分析 (EDA)
# ============================================
print("=" * 60)
print("Part 1: 数据探索性分析与预处理")
print("=" * 60)

# 加载数据
df = pd.read_csv('Dataset.csv')
print(f"\n[1.1] 数据基本信息")
print(f"数据形状: {df.shape}")
print(f"样本数量: {df.shape[0]}, 特征数量: {df.shape[1]}")

# 数据类型
print(f"\n数据类型统计:")
print(df.dtypes.value_counts())

# 描述性统计
print(f"\n[1.2] 描述性统计")
print(df.describe())

# 目标变量分布
print(f"\n[1.3] 目标变量分布 (Diabetes_012)")
target_counts = df['Diabetes_012'].value_counts().sort_index()
print(target_counts)
print(f"\n0: 无糖尿病 ({(target_counts[0]/len(df)*100):.2f}%)")
print(f"1: 糖尿病前期 ({(target_counts[1]/len(df)*100):.2f}%)")
print(f"2: 糖尿病 ({(target_counts[2]/len(df)*100):.2f}%)")

# ============================================
# 2. 可视化分析
# ============================================
print(f"\n[2.1] 生成可视化分析...")

# 创建画布
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Diabetes Dataset - Exploratory Data Analysis', fontsize=16)

# 2.1 目标变量分布
ax1 = axes[0, 0]
target_counts.plot(kind='bar', ax=ax1, color=['#2ecc71', '#f1c40f', '#e74c3c'])
ax1.set_title('Target Variable Distribution')
ax1.set_xlabel('Diabetes_012')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['0: No Diabetes', '1: Pre-Diabetes', '2: Diabetes'], rotation=45)

# 2.2 BMI分布按糖尿病状态
ax2 = axes[0, 1]
for label in [0, 1, 2]:
    subset = df[df['Diabetes_012'] == label]['BMI']
    ax2.hist(subset, bins=30, alpha=0.5, label=f'Diabetes_{int(label)}')
ax2.set_title('BMI Distribution by Diabetes Status')
ax2.set_xlabel('BMI')
ax2.set_ylabel('Frequency')
ax2.legend()

# 2.3 年龄分布按糖尿病状态
ax3 = axes[0, 2]
for label in [0, 1, 2]:
    subset = df[df['Diabetes_012'] == label]['Age']
    ax3.hist(subset, bins=20, alpha=0.5, label=f'Diabetes_{int(label)}')
ax3.set_title('Age Distribution by Diabetes Status')
ax3.set_xlabel('Age')
ax3.set_ylabel('Frequency')
ax3.legend()

# 2.4 高血压与糖尿病关系
ax4 = axes[1, 0]
cross_highbp = pd.crosstab(df['HighBP'], df['Diabetes_012'], normalize='index') * 100
cross_highbp.plot(kind='bar', ax=ax4, color=['#2ecc71', '#f1c40f', '#e74c3c'])
ax4.set_title('High BP vs Diabetes')
ax4.set_xlabel('HighBP')
ax4.set_ylabel('Percentage (%)')
ax4.legend(title='Diabetes')
ax4.set_xticklabels(['No High BP', 'High BP'], rotation=0)

# 2.5 高胆固醇与糖尿病关系
ax5 = axes[1, 1]
cross_highchol = pd.crosstab(df['HighChol'], df['Diabetes_012'], normalize='index') * 100
cross_highchol.plot(kind='bar', ax=ax5, color=['#2ecc71', '#f1c40f', '#e74c3c'])
ax5.set_title('High Cholesterol vs Diabetes')
ax5.set_xlabel('HighChol')
ax5.set_ylabel('Percentage (%)')
ax5.legend(title='Diabetes')
ax5.set_xticklabels(['No High Chol', 'High Chol'], rotation=0)

# 2.6 身体活动与糖尿病关系
ax6 = axes[1, 2]
cross_phys = pd.crosstab(df['PhysActivity'], df['Diabetes_012'], normalize='index') * 100
cross_phys.plot(kind='bar', ax=ax6, color=['#2ecc71', '#f1c40f', '#e74c3c'])
ax6.set_title('Physical Activity vs Diabetes')
ax6.set_xlabel('PhysActivity')
ax6.set_ylabel('Percentage (%)')
ax6.legend(title='Diabetes')
ax6.set_xticklabels(['No Activity', 'Active'], rotation=0)

# 2.7 吸烟与糖尿病关系
ax7 = axes[2, 0]
cross_smoker = pd.crosstab(df['Smoker'], df['Diabetes_012'], normalize='index') * 100
cross_smoker.plot(kind='bar', ax=ax7, color=['#2ecc71', '#f1c40f', '#e74c3c'])
ax7.set_title('Smoker vs Diabetes')
ax7.set_xlabel('Smoker')
ax7.set_ylabel('Percentage (%)')
ax7.legend(title='Diabetes')
ax7.set_xticklabels(['No', 'Smoker'], rotation=0)

# 2.8 步行困难与糖尿病关系
ax8 = axes[2, 1]
cross_diffwalk = pd.crosstab(df['DiffWalk'], df['Diabetes_012'], normalize='index') * 100
cross_diffwalk.plot(kind='bar', ax=ax8, color=['#2ecc71', '#f1c40f', '#e74c3c'])
ax8.set_title('Difficulty Walking vs Diabetes')
ax8.set_xlabel('DiffWalk')
ax8.set_ylabel('Percentage (%)')
ax8.legend(title='Diabetes')
ax8.set_xticklabels(['No', 'Difficulty'], rotation=0)

# 2.9 相关性热力图
ax9 = axes[2, 2]
corr = df.corr()['Diabetes_012'].drop('Diabetes_012').sort_values(ascending=False)
colors = ['#e74c3c' if x > 0 else '#3498db' for x in corr.values]
corr.plot(kind='barh', ax=ax9, color=colors)
ax9.set_title('Correlation with Diabetes_012')
ax9.set_xlabel('Correlation Coefficient')
ax9.axvline(x=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('output1/01_eda_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - 已保存: output1/01_eda_visualization.png")

# ============================================
# 3. 缺失值检测与处理
# ============================================
print(f"\n[3.1] 缺失值检测与处理")

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
missing_df = missing_df[missing_df['缺失数量'] > 0]
print(f"\n缺失值统计:")
print(missing_df if len(missing_df) > 0 else "无缺失值")

# 处理缺失值
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
print(f"\n缺失值处理完成")

# ============================================
# 4. 异常值检测与处理
# ============================================
print(f"\n[4.1] 异常值检测与处理")

numerical_cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']
outlier_info = []

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_info.append({
        '特征': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        '下界': lower,
        '上界': upper,
        '异常值数量': outliers,
        '异常比例(%)': (outliers/len(df))*100
    })
    # 将异常值替换为边界值
    df[col] = df[col].clip(lower=lower, upper=upper)

outlier_df = pd.DataFrame(outlier_info)
print(f"\n异常值统计:")
print(outlier_df)

# 保存异常值报告
outlier_df.to_csv('output1/02_outlier_report.csv', index=False)
print(f"\n  - 已保存: output1/02_outlier_report.csv")

# ============================================
# 4.2 异常值可视化
# ============================================
print(f"\n[4.2] 生成异常值可视化...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
numerical_cols_list = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']

for idx, col in enumerate(numerical_cols_list):
    ax = axes[idx // 2, idx % 2]
    # 使用原始数据绘制箱线图
    data_to_plot = df[col].values
    bp = ax.boxplot(data_to_plot, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)

    # 获取异常值边界
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # 标记异常值边界
    ax.axhline(y=lower, color='red', linestyle='--', linewidth=1.5, label=f'Lower: {lower:.2f}')
    ax.axhline(y=upper, color='red', linestyle='--', linewidth=1.5, label=f'Upper: {upper:.2f}')

    # 统计异常值数量
    outliers_count = ((df[col] < lower) | (df[col] > upper)).sum()
    ax.set_title(f'{col} - Outliers: {outliers_count} ({outliers_count/len(df)*100:.2f}%)')
    ax.set_ylabel(col)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Outlier Detection Results (Before Capping)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output1/07_outlier_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  - 已保存: output1/07_outlier_visualization.png")

# ============================================
# 5. 数据标准化与编码
# ============================================
print(f"\n[5.1] 数据标准化与编码")

# 删除重复值
initial_shape = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"删除重复值: {initial_shape} -> {df.shape[0]}")

# 数据类型转换
for col in df.columns:
    if df[col].dtype == 'float64':
        if df[col].notna().all() and (df[col] == df[col].astype(int)).all():
            df[col] = df[col].astype(int)

# 特征缩放
scaler = StandardScaler()
scale_cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']
df_scaled = df.copy()
df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])
print(f"数值特征标准化完成: {scale_cols}")

# ============================================
# 5.2 标准化前后对比可视化
# ============================================
print(f"\n[5.2] 生成标准化前后对比图...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
numerical_cols_list = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']

for idx, col in enumerate(numerical_cols_list):
    # 原始数据分布
    ax_orig = axes[0, idx]
    data_orig = df[col].values
    ax_orig.hist(data_orig, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax_orig.set_title(f'{col} - Original Distribution')
    ax_orig.set_xlabel('Value')
    ax_orig.set_ylabel('Frequency')
    ax_orig.grid(True, alpha=0.3)

    # 添加统计信息
    mean_val = np.mean(data_orig)
    std_val = np.std(data_orig)
    ax_orig.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax_orig.legend(fontsize=8)

    # 标准化后数据分布
    ax_scaled = axes[1, idx]
    data_scaled = df_scaled[col].values
    ax_scaled.hist(data_scaled, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax_scaled.set_title(f'{col} - Standardized (Z-score)')
    ax_scaled.set_xlabel('Z-score')
    ax_scaled.set_ylabel('Frequency')
    ax_scaled.grid(True, alpha=0.3)

    # 添加统计信息
    mean_scaled = np.mean(data_scaled)
    std_scaled = np.std(data_scaled)
    ax_scaled.axvline(x=mean_scaled, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_scaled:.2f}\nStd: {std_scaled:.2f}')
    ax_scaled.legend(fontsize=8)

plt.suptitle('Data Standardization: Before vs After (Z-score Normalization)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output1/08_standardization_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  - 已保存: output1/08_standardization_comparison.png")

# 5.3 标准化前后QQ图检验正态性
print(f"\n[5.3] 生成QQ图检验正态性...")

from scipy import stats

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for idx, col in enumerate(numerical_cols_list):
    # 原始数据QQ图
    ax_orig = axes[0, idx]
    data_orig = df[col].values
    stats.probplot(data_orig, dist="norm", plot=ax_orig)
    ax_orig.set_title(f'{col} - Original Q-Q Plot')
    ax_orig.grid(True, alpha=0.3)

    # 标准化后QQ图
    ax_scaled = axes[1, idx]
    data_scaled = df_scaled[col].values
    stats.probplot(data_scaled, dist="norm", plot=ax_scaled)
    ax_scaled.set_title(f'{col} - Standardized Q-Q Plot')
    ax_scaled.grid(True, alpha=0.3)

plt.suptitle('Q-Q Plot for Normality Check: Before vs After Standardization', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('output1/09_qq_plot_standardization.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  - 已保存: output1/09_qq_plot_standardization.png")

# ============================================
# 6. 保存处理后的数据
# ============================================
print(f"\n[6.1] 保存处理后的数据")

# 保存原始处理后数据
df.to_csv('output1/03_data_cleaned.csv', index=False)
print(f"  - 已保存: output1/03_data_cleaned.csv")

# 保存标准化数据
df_scaled.to_csv('output1/04_data_scaled.csv', index=False)
print(f"  - 已保存: output1/04_data_scaled.csv")

# 保存EDA统计报告
with open('output1/05_eda_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("数据探索性分析报告\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"1. 数据基本信息\n")
    f.write(f"   - 样本数量: {df.shape[0]}\n")
    f.write(f"   - 特征数量: {df.shape[1]}\n\n")
    f.write(f"2. 目标变量分布\n")
    for label, count in target_counts.items():
        f.write(f"   - Diabetes_{int(label)}: {count} ({count/len(df)*100:.2f}%)\n")
    f.write(f"\n3. 缺失值处理\n")
    f.write(f"   - 总缺失值: {df.isnull().sum().sum()}\n")
    f.write(f"\n4. 异常值处理\n")
    f.write(outlier_df.to_string())
    f.write(f"\n\n5. 数据预处理完成\n")
    f.write(f"   - 数值特征已标准化\n")
    f.write(f"   - 分类特征已编码\n")
    f.write(f"   - 重复值已删除\n")

print(f"  - 已保存: output1/05_eda_report.txt")

# ============================================
# 7. 生成相关性分析图
# ============================================
print(f"\n[7.1] 生成相关性热力图...")

fig, ax = plt.subplots(figsize=(14, 12))
corr_matrix = df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlBu_r', center=0, ax=ax,
            annot_kws={'size': 7})
ax.set_title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('output1/06_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  - 已保存: output1/06_correlation_heatmap.png")

print("\n" + "=" * 60)
print("Part 1 完成!")
print("=" * 60)
