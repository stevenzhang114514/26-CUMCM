# 最小复现材料与运行记录

生成时间：2026-09-11T14:05:58

## 1. 环境信息

- Python：3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
- 平台：Windows-11-10.0.26200-SP0
- numpy：2.5.3
- pandas：3.0.5
- scipy：1.18.1
- matplotlib：3.11.1

## 2. 配置信息

### p1_v1

- 问题：None
- 时间范围：None
- 空间范围：None
- 时间步长：None
- 空间步长：None
- 求解方法：None
- 时间格式：None
- 最大残差：None
- 验证状态：None
- 备注：None

### p2_v1

- 问题：None
- 时间范围：None
- 空间范围：None
- 时间步长：None
- 空间步长：None
- 求解方法：None
- 时间格式：None
- 最大残差：None
- 验证状态：None
- 备注：None

### p3_v1

- 问题：None
- 时间范围：None
- 空间范围：None
- 时间步长：None
- 空间步长：None
- 求解方法：None
- 时间格式：None
- 最大残差：None
- 验证状态：None
- 备注：None

### p4_v1

- 问题：None
- 时间范围：None
- 空间范围：None
- 时间步长：None
- 空间步长：None
- 求解方法：None
- 时间格式：None
- 最大残差：None
- 验证状态：None
- 备注：None

## 3. 结果文件

- data\result1.xlsx
- data\result1_template.xlsx
- data\result2.xlsx
- data\result2_template.xlsx
- data\result3.xlsx
- data\result3_template.xlsx
- data\result4.xlsx
- data\result4_template.xlsx

## 4. 运行入口

```powershell
cd D:\math_modeling\project
python io\code31_template_parser.py
python io\code31b_build_templates.py
python data_prep\code02_env_interp.py
python data_prep\code03_stage_identify.py
python io\code16_resample.py
python io\code17_excel_export.py
python io\code30_reproduce.py
```

## 5. 待确认项

- [ ] 阶段边界时间是否与甲、乙一致
- [ ] 14400 s 后环境边界外推规则
- [ ] 问题2 完整时间范围
- [ ] 问题3 烘干结束时间
- [ ] 问题4 烘干结束时间
- [ ] result4 中间列是物理坐标还是变换坐标
- [ ] result3 是否包含 0 s
- [ ] result1–4 A 列是否从 0 开始
- [ ] 论文表3/4 是 3 h 还是完整过程