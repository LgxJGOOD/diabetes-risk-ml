# 糖尿病风险识别与防范系统

基于机器学习的糖尿病患病风险预测：25 万级样本、22 维健康指标，从 EDA 到模型评估的完整建模流程，并输出可解释的风险因素分析。

## 数据

- 数据源：Kaggle **Diabetes Health Indicators**（CDC BRFSS 2015 调查数据，公开数据集，未随仓库分发）
  下载地址：`https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset`
- 把下载得到的 CSV 重命名为 `Dataset.csv` 放在仓库根目录即可运行
- 样本量：约 **25.3 万**条，**22** 个健康与生活方式特征（BMI、血压、胆固醇、吸烟饮酒、年龄、既往病史等）
- 目标变量：是否患病（二分类）
- 类别不平衡：通过分层采样与类别权重处理

> 仓库内 `output1/ ~ output3/` 为本地完整跑通后落盘的结果，可先看产物再决定是否复跑。

## 运行

```bash
pip install -r requirements.txt
python part1_eda_preprocessing.py   # 需先准备 Dataset.csv
python part2_feature_model.py
python part3_evaluation.py
```

## 流程

三个阶段脚本独立可跑，中间产物落盘便于复现：

```
part1_eda_preprocessing.py   缺失值处理 → 异常值检查 → 分布可视化 → 编码与标准化
part2_feature_model.py       RFE 递归特征消除 → LR / 随机森林 / GBDT 训练 → 网格搜索调参 → Stacking 集成
part3_evaluation.py          混淆矩阵、ROC/AUC、特征重要性、分类报告
```

## 技术栈

Python、pandas、NumPy、scikit-learn、Matplotlib / Seaborn

## 输出

| 目录 | 内容 |
|---|---|
| `output1/` | EDA 阶段的分布图与相关性分析 |
| `output2/` | 各模型预测概率与中间特征 |
| `output3/` | 评估结果：`01_model_evaluation_metrics.csv`、混淆矩阵、ROC 曲线、模型对比图、特征重要性、最终报告 |

## 运行

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

python part1_eda_preprocessing.py
python part2_feature_model.py
python part3_evaluation.py
```

数据集请放入项目根目录（`Dataset.csv`），因体积原因未随仓库提供。

## 结论要点

- 单一模型中 **GBDT 类方法**表现最好，随机森林次之，逻辑回归作为可解释基线；
- **Stacking 集成**在 AUC 上较单模型有小幅提升，但收益小于特征工程带来的增益；
- 特征重要性高度集中：血糖相关指标、BMI、年龄贡献最大，与医学常识一致。

## 联系

刘冠湘 · 湖南第一师范学院 · 人工智能专业 · 2027 届 · sxangus@qq.com
