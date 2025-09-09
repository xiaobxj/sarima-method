# SARIMA Time Series Modeling for Treasury Cash Flow Data

## 项目简介

本项目实现了对美国财政部现金流数据的SARIMA时间序列建模。项目包含数据收集、预处理、个人时间序列生成以及批量SARIMA建模功能。

## 项目结构

```
pythonProject/
├── data/
│   ├── raw/                              # 原始数据文件
│   │   ├── deposits_withdrawals_operating_cash.csv
│   │   ├── daily_cash_flows_*.csv
│   │   └── ...
│   └── processed/
│       ├── individual_time_series/       # 197个独立时间序列文件
│       └── treasury_modeling_data_*.csv
├── src/
│   ├── data/
│   │   ├── data_collector.py             # 数据收集模块
│   │   ├── enhanced_field_mapper_complete.py
│   │   └── run_data_collection.py
│   └── model/
│       ├── sarima_model.py               # SARIMA建模核心类
│       ├── batch_modeling.py             # 批量建模处理
│       ├── run_batch_sarima.py           # 批量建模执行脚本
│       └── fitted_models/                # 186个训练好的SARIMA模型
├── config/
│   ├── config.py                         # 配置文件
│   └── env_template.txt
├── logs/                                 # 日志文件
├── requirements.txt                      # 依赖包列表
├── main.py                              # 主程序入口
└── create_sarima_models.py              # 快速建模脚本
```

## 功能特性

### 1. 数据处理
- **个人时间序列生成**: 从原始数据中提取197个独特的财政类别，生成独立的时间序列数据集
- **数据预处理**: 处理缺失值、异常值检测和数据清洗
- **非建模项目过滤**: 自动排除Cash FTD's、Public Debt等非建模类别

### 2. SARIMA建模
- **自动参数优化**: 使用网格搜索找到最优的SARIMA参数
- **批量建模**: 并行处理多个时间序列，提高建模效率
- **模型验证**: 包含残差分析、Ljung-Box检验等模型诊断
- **预测功能**: 支持多步预测和置信区间计算

### 3. 主要类别覆盖

**存款类 (Deposits)**:
- Withheld Income and Employment Taxes
- Individual Income Taxes  
- Corporation Income Taxes
- Federal Reserve Earnings
- Other Deposits

**支出类 (Withdrawals)**:
- Social Security Benefits (EFT)
- Medicare & Medicaid
- Defense Vendor Payments (EFT)
- Education Department Programs
- Agriculture Department Programs
- Unemployment Insurance Benefits
- Other Withdrawals

## 安装和使用

### 环境要求
```bash
pip install -r requirements.txt
```

### 主要依赖
- pandas >= 2.0.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- scikit-learn
- matplotlib
- seaborn

### 快速开始

1. **生成个人时间序列**:
```python
# 已完成：197个独立时间序列文件保存在 data/processed/individual_time_series/
```

2. **批量SARIMA建模**:
```bash
python create_sarima_models.py
```

3. **使用单个模型**:
```python
from src.model.sarima_model import SARIMAModel

# 加载特定时间序列
model = SARIMAModel('Defense_Vendor_Payments_EFT', 
                   'data/processed/individual_time_series/Defense_Vendor_Payments_EFT.csv')
model.load_data()
model.fit_model()
forecast = model.forecast(steps=30)
```

## 建模结果

- **成功建模**: 186/195 个时间序列 (95.4% 成功率)
- **跳过文件**: 9个文件因数据不足被跳过
- **模型类型**: SARIMA(p,d,q)×(P,D,Q,s) 其中 s=7 (周季节性)
- **评估指标**: 使用AIC进行模型选择和比较

### 顶级模型示例 (按AIC排序)
- Change_in_Balance_of_Uncollected_Funds: AIC = -40742.81
- Transfers_to_Depositaries: AIC = -40742.81  
- Transfers_to_Federal_Reserve_Account: AIC = -37209.87
- Transfers_from_Federal_Reserve_Account: AIC = -37235.66
- Interest_recd_from_cash_investments: AIC = -35868.90

## 数据来源

本项目使用美国财政部Daily Treasury Statement数据，包含：
- 联邦储备账户存取款记录
- 公共债务交易数据
- 税收存款和退税数据
- 各政府部门支出数据
- 时间跨度：2016年至2025年

## 技术特点

- **并行处理**: 支持多核并行建模
- **自动化流程**: 从数据加载到模型保存的全自动化流程
- **错误处理**: 完善的异常处理和日志记录
- **模型持久化**: 所有训练好的模型保存为pickle格式
- **诊断工具**: 内置模型诊断和可视化功能

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License

---

**作者**: xiaobxj  
**项目链接**: https://github.com/xiaobxj/sarima-method  
**创建时间**: 2025年1月
