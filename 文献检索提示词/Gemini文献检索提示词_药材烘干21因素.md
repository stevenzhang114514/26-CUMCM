# Gemini 文献检索提示词 · 药材烘干「21 条被省略因素」

> **用途**：为「2026 CUMCM A 题 · 药材的烘干问题」论文的**模型评价与推广**章节，
> 检索支撑文献。
>
> **使用对象**：Gemini（建议开启 **Deep Research / 联网搜索**模式）
>
> **核心难点**：不是"让它找文献"，而是**防它编造文献** —— 主提示词含一整节反幻觉约束，
> 且第五部分有必做的核查追问。**DOI 必须逐个点开验证，这一步不能省。**

---

## 目录

- [一、使用建议](#一使用建议)
- [二、主提示词（复制即用）](#二主提示词复制即用)
- [三、精简版（快速摸底）](#三精简版快速摸底)
- [四、单因素深挖模板](#四单因素深挖模板)
- [五、追问提示词（拿到结果后必做）](#五追问提示词拿到结果后必做)
- [六、三个提醒](#六三个提醒)

---

## 一、使用建议

- **分批投喂**：21 条一次问完，Gemini 大概率会开始编。建议按类别分 **4 批**：

  | 批次 | 因素编号 | 类别 |
  |---|---|---|
  | 批次 1 | 1–7 | 传热传质类 |
  | 批次 2 | 8–13 | 烘房系统类 |
  | 批次 3 | 14–17 | 生物化学类 |
  | 批次 4 | 18–21 | 力学与统计类 |

- **开启 Deep Research 模式**（如果有）。普通模式建议一次不超过 5 条。
- **每批跑完后立刻做一次交叉验证**（见第五部分）。
- **试点建议**：先用第 15、14、18 条（挥发油损失、微生物霉变、干裂）跑通流程 ——
  这三条中英文文献都丰富，且与中药材关联度最高，最容易找到可直接引用的好文献。

---

## 二、主提示词（复制即用）

```
【角色】
你是一位干燥技术与传热传质领域的文献检索专家，熟悉中药材/农产品热风干燥、
多孔介质传热传质、食品工程与干燥设备设计的中英文文献。

【研究背景】
我正在为一道数学建模竞赛题建立「圆柱形中药材热风烘干」的数学模型。现有模型：
- 几何：一维轴对称无限长圆柱（半径 2 cm，长 25 cm）
- 方程：耦合的径向导热方程 + 有效扩散型水分迁移方程
- 边界：表面对流换热（h=25 W/(m²·K)）+ 表面对流传质（h_m=8×10⁻⁷ m/s）
- 物性：经验公式给出的 ρ(C)、c_p(C)、k(C)、D(C,T)
- 工况：28 °C 升到 50 °C，烘干时长 2–3 天，目标含水率从 2.55 降到 0.15 kg/kg（干基）

题目只给了上述理想化条件，明确省略了大量真实物理因素。
我需要为这些「被省略的因素」找到可靠的学术文献支撑，用于论文的
「模型评价与推广」章节。

【核心任务】
针对下面列出的每一个因素，检索并给出 2–5 篇**高质量、真实存在、可溯源**的文献。
每个因素我需要的不是泛泛的综述，而是能直接用于建模的东西：
  (a) 该因素的标准数学描述（控制方程 / 本构关系 / 判据）
  (b) 典型参数取值范围（例如某类物料的潜热、发射率、玻璃化转变温度、
      临界含水率、断裂应力等的典型数值）
  (c) 该因素对干燥时间或品质的影响量级（有定量结论最好）
  (d) 是否可忽略的判据（例如某个无量纲数小于某值时可以忽略）

【检索范围与质量要求】
1. 中英文文献都要，英文文献优先（该领域英文文献更系统）
2. 优先顺序：权威期刊综述 > 高被引经典文献 > 近 5 年新文献 > 学位论文
3. 推荐期刊方向：Journal of Food Engineering、Drying Technology、
   Food and Bioproducts Processing、International Journal of Heat and Mass Transfer、
   Journal of Food Science、Bioresource Technology、
   中文的《农业工程学报》《食品科学》《中药材》《中草药》《干燥技术与设备》等
4. 时间范围以近 20 年为主，但奠基性的经典文献不受年份限制
5. 明确区分两类文献：
   【方法类】提供建模方法与方程来源的
   【数据类】提供参数取值、实验数据、经验关联式的

【21 个待检索因素】
（编号、名称、物理含义、中英文检索关键词）

1. 蒸发潜热（表面蒸发吸热导致的降温效应）
   中：蒸发潜热 表面能量平衡 干燥 物料温度 湿球温度
   英：latent heat of vaporization drying, evaporative cooling,
       wet bulb temperature, surface energy balance drying

2. 内部蒸发与水蒸气渗流（水在物料内部汽化后以气体形式迁移）
   中：多孔介质 内部蒸发 水蒸气扩散 达西流 干燥
   英：internal evaporation porous media, vapor diffusion drying,
       Darcy flow porous media, multiphase drying model

3. Soret/Dufour 交叉效应（温度梯度驱动水分、浓度梯度驱动热流）
   中：热湿耦合 热扩散效应 Luikov 模型 不可逆热力学
   英：Soret effect drying, Dufour effect, thermodiffusion moisture,
       Luikov model coupled heat mass transfer, irreversible thermodynamics

4. 辐射换热（烘房壁面与物料之间的热辐射）
   中：干燥 辐射换热 复合换热系数 对流辐射耦合
   英：thermal radiation drying, combined convection radiation drying,
       radiative heat transfer dryer, effective heat transfer coefficient

5. 表面结壳 / 硬化（表面过干形成致密层阻碍内部水分迁出）
   中：表面结壳 硬化 干燥 表皮效应
   英：case hardening drying, crust formation, surface crust,
       skin formation food drying

6. 结合水与自由水（水分存在状态对脱除难易的影响）
   中：结合水 自由水 水分状态 核磁共振 干燥
   英：bound water free water drying, water state,
       NMR moisture distribution, water activity drying

7. 玻璃化转变（干燥后期物料由橡胶态转为玻璃态，扩散系数骤降）
   中：玻璃化转变 干燥 橡胶态 玻璃态 状态图
   英：glass transition drying, rubbery glassy state,
       state diagram food, stickiness drying, molecular mobility

8. 烘房与物料的双向耦合（物料蒸发的水分反过来改变烘房湿度，蒸发吸热改变烘房温度）
   中：烘房 物料 耦合 湿度 批次干燥 建模
   英：dryer material coupling, two-way coupling drying,
       batch dryer modeling, humidity buildup drying,
       coupled dryer model

9. 排湿换气与新风量（排湿风门、循环风比、新风率）
   中：排湿 换气率 循环风 热风干燥 新风比
   英：ventilation rate drying, air exchange rate dryer,
       recirculation ratio drying, dehumidification drying,
       exhaust air ratio

10. 烘房内温湿度空间不均匀（上下层温差、边角风速差）
    中：烘房 气流组织 温湿度分布 不均匀性 干燥
    英：non-uniform temperature humidity dryer, airflow distribution drying,
        CFD dryer simulation, air velocity uniformity

11. 装料方式与料层遮挡（多层堆叠导致有效换热面积减小）
    中：料层 堆积 装料量 干燥 有效面积
    英：loading pattern drying, tray stacking drying,
        packed bed drying, bed depth effect drying

12. 烘房壁面蓄热与热惰性
    中：烘房 蓄热 热惰性 升温 滞后
    英：thermal inertia dryer, wall heat storage drying,
        dryer transient response, heat-up period dryer

13. 控温波动与控制器超调
    中：干燥 温度控制 波动 超调 PID
    英：temperature fluctuation drying, control loop oscillation dryer,
        PID overshoot drying, control stability drying

14. 微生物生长与霉变（烘干前期低温高湿是霉菌繁殖窗口）
    中：中药材 霉变 霉菌 干燥 微生物 预测微生物学
    英：mold growth during drying, fungal spoilage herbs,
        predictive microbiology drying, Gompertz model microbial growth,
        microbial inactivation drying temperature

15. 挥发油损失（水蒸气蒸馏效应带走挥发性有效成分）
    中：中药材 挥发油 损失 干燥 有效成分 水蒸气蒸馏
    英：essential oil loss drying, volatile retention drying,
        aroma retention drying, steam distillation effect,
        volatile compound loss food drying

16. 酶促褐变 / 美拉德反应 / 有效成分热降解
    中：酶促褐变 美拉德 干燥 多酚氧化酶 热降解 中药材
    英：enzymatic browning drying, Maillard reaction drying,
        polyphenol oxidase inactivation, thermal degradation bioactive compounds,
        color change drying kinetics

17. 虫卵与虫害（50 °C 不足以杀灭虫卵）
    中：中药材 虫卵 杀虫 干燥 仓储害虫
    英：insect egg disinfestation drying, pest control dried herbs,
        thermal disinfestation temperature, stored product insects

18. 干裂、龟裂与碎粉（收缩内应力导致开裂破碎）
    中：干燥 开裂 龟裂 应力 破碎 断裂
    英：cracking during drying, stress crack drying,
        fissuring, viscoelastic stress drying, breakage dried product,
        tensile strength failure drying

19. 各向异性收缩与长度变化（径向与轴向收缩率不同）
    中：干燥 收缩 各向异性 体积变化 纤维方向
    英：anisotropic shrinkage drying, shrinkage drying model,
        volume change drying, fiber orientation shrinkage,
        ideal shrinkage assumption

20. 端面效应（有限长圆柱的两端额外传热传质）
    中：有限长圆柱 端面效应 二维轴对称 干燥
    英：finite cylinder drying, end effect drying,
        2D axisymmetric drying model, edge effect moisture transfer

21. 参数不确定性、测量误差与批次差异
    中：干燥 参数不确定性 灵敏度分析 蒙特卡洛 批次差异
    英：parameter uncertainty drying, uncertainty quantification food process,
        sensitivity analysis drying model, Monte Carlo drying,
        batch to batch variation drying

【输出格式】
按因素编号逐条输出，每条严格使用以下模板：

---
### 因素 N：<中文名>（<英文名>）
**对该因素的标准处理方式**
<2–4 句话说明学术界通常怎么建模，给出关键方程或判据（可用 LaTeX）>

**可忽略性判据**
<如果有，给出无量纲数或临界条件；如果没有统一判据，写"无统一判据">

**文献列表**

| # | 标题 | 作者 | 期刊/会议 | 年份 | 类型 | 可提供的建模要素 | 链接/DOI |
|---|------|------|-----------|------|------|------------------|----------|
| 1 |      |      |           |      | 方法类/数据类 | 方程 / 参数值 / 影响量级 |    |

**关键结论摘录**
- <定量结论 1，例如"某物料在 X 条件下潜热使升温速率降低 Y%">
- <定量结论 2>

**可信度标注**
<高 / 中 / 低 —— 说明你对该条文献真实性的把握程度>
---

【反幻觉与溯源要求（最重要，必须严格执行）】
1. 绝对不要编造文献。不确定是否真实存在的文献，宁可不给。
2. 每一篇文献必须给出可核验的信息：标题、第一作者、期刊、年份、DOI 或稳定链接。
   缺少 DOI 时给出可访问的 URL（如 PubMed、ScienceDirect、CNKI、万方链接）。
3. 如果你不确定某篇文献是否真实存在，或不确定某个字段（年份、期刊名等），
   必须显式标注「待核实」，绝不允许猜测填充。
4. 如果某个因素你确实找不到合格文献，直接写「**未找到合格文献**」，
   并说明你尝试过的检索方向。不要用主题相近的其他文献充数。
5. 禁止把综述文章里提到的次级引用当作你亲自检索到的文献列出来。
6. 所有数值型结论（参数范围、影响百分比等）必须注明出处文献编号，
   无法注明出处的数值一律不写。
7. 如果检索结果不足以支撑某个结论，明确说明「证据不足」。

【附加要求】
1. 最后给出一张汇总表：21 个因素 × 文献数量 × 可信度 × 是否建议纳入论文推广章节。
2. 标出其中「文献丰富、结论明确、最值得深入建模」的前 5 个因素。
3. 标出其中「文献稀少、争议大、建议在论文中仅作定性提及」的因素。
4. 如某因素存在相互矛盾的研究结论，请明确指出矛盾点。
```

---

## 三、精简版（快速摸底）

```
请为「圆柱形中药材热风烘干数学模型」的以下被省略因素检索真实存在的学术文献，
每个因素 2–3 篇，中英文皆可，英文优先。

因素：<粘贴 1–5 条>

对每篇文献给出：标题 / 作者 / 期刊 / 年份 / DOI或链接 /
属于「建模方法类」还是「参数数据类」/ 一句话说明它能提供什么。

严禁编造。不确定真实性的文献请直接省略，并说明"未找到合格文献"。
所有不确定的字段标注「待核实」。
```

---

## 四、单因素深挖模板

找到方向后，用这个挖细节：

```
围绕以下因素做一次深入检索：

因素：<名称>
上下文：圆柱形中药材热风干燥，一维轴对称模型，28→50 °C，2–3 天

请重点检索并回答：
1. 该因素的标准控制方程或本构关系是什么？请给出完整数学形式。
2. 该因素在建模中引入后，会新增哪些参数？这些参数的典型取值范围是多少？
   请按物料类别给出（中药材 / 果蔬 / 谷物 / 通用多孔介质）。
3. 有哪些现成的无量纲判据可以判断该因素能否忽略？给出判据表达式和临界值。
4. 有没有实验数据可以直接引用？特别是干燥曲线、温度曲线、品质指标的变化。
5. 该因素与主控方程（导热 + 有效扩散）的耦合强度如何？
   是单向影响还是双向耦合？
6. 在类似规模的建模工作中，学界的主流做法是纳入还是忽略？理由是什么？

每条结论都必须注明来源文献。
没有文献支撑的部分请明确标注「无文献支撑，属推论」。
```

---

## 五、追问提示词（拿到结果后必做）

### ① 核查真实性

```
请对上面列出的全部文献做一次真实性自检：
- 逐条说明你对每篇文献确定性有多高（确定存在 / 较确定 / 不确定）
- 把标注为「不确定」的全部单独列出
- 对于不确定的条目，说明你是基于什么信息认为它可能存在的
- 明确告诉我哪些条目我应当自己去数据库核实
```

### ② 提炼可用参数

```
请把上面的检索结果整理成一张「建模可用参数表」，
只保留有明确文献出处、可直接代入模型的数值：

| 参数 | 符号 | 含义 | 典型值/范围 | 适用物料 | 来源文献编号 |

无法给出明确出处的参数一律不列入。
```

### ③ 生成论文段落

```
基于以上检索结果，帮我撰写数学建模论文「模型评价与推广」章节的草稿。
要求：
1. 按「未纳入的因素 + 忽略依据 + 影响量级 + 推广方向」四段式组织
2. 主动交代模型假设的边界，不要写成"我们忽略了这些"的自辩语气
3. 凡是引用文献结论的地方标注 [编号]
4. 凡是没有文献支撑、仅为量级估算的内容，明确标注"本文估算"
5. 中文写作，学术论文语体
```

### ④ 找反例

```
上面这些因素，学界有没有研究得出「该因素其实可以忽略」的结论？
请找出支持"可以忽略"的证据，并给出适用的条件范围。
这对我判断模型中哪些简化是安全的很重要。
```

---

## 六、三个提醒

1. **Gemini 编造文献的概率不低**，尤其是作者名和年份。拿到结果后，把 DOI 逐个粘到浏览器里点一遍 —— **这一步不能省**。反过来，如果它给出了一个看起来过于"完美"的引用（标题正好切题、年份正好近年），更要警惕。

2. **要求它区分「方法类」和「数据类」文献**是这份提示词里最实用的一条。建模需要的是前者（方程从哪来），敏感性分析和参数代入需要的是后者（数值是多少）。两类需求混在一起，检索质量会下降。

3. **优先用第 15、14、18 条做试点**（挥发油损失、微生物霉变、干裂）。这三条是中英文文献都很丰富的方向，且与中药材的关联度最高，最容易找到能直接引用的好文献 —— 先跑通流程，再套用到其他条目。

---

## 附：21 条因素的物理内涵速查

| # | 因素 | 一句话说明 | 论文价值 |
|---|---|---|---|
| 1 | 蒸发潜热 | 蒸发耗热约为物料升温显热的 23 倍 | ★★★ |
| 2 | 内部蒸发与水蒸气渗流 | 水在内部汽化后以气体形式迁移，引入内部气压 | ★★ |
| 3 | Soret/Dufour 效应 | 温度梯度把水往冷处推，与干燥方向相反 | ★ |
| 4 | 辐射换热 | 净辐射约为对流的 25%，是否已含在 h 中不明 | ★★★ |
| 5 | 表面结壳 | 表面过干形成致密层，把内部水分锁死 | ★★ |
| 6 | 结合水与自由水 | 末端剩余的是结合水，需更高能量脱除 | ★★ |
| 7 | 玻璃化转变 | 后期转为玻璃态，扩散系数出现断崖式下降 | ★★ |
| 8 | 烘房双向耦合 | 一根药材的水分足以加湿 6.25 m³ 空气 | ★★★ |
| 9 | 排湿换气 | 工程上最重要的操作变量之一 | ★★ |
| 10 | 空间不均匀 | 同批不同位置干燥速率可差 20–30% | ★★ |
| 11 | 装料遮挡 | 多层堆叠使有效换热面积远小于几何表面积 | ★ |
| 12 | 壁面蓄热 | 开机后需先加热墙体料架，造成升温滞后 | ★ |
| 13 | 控温波动 | 附件1 尾部有 ±0.2 °C 周期性抖动 | ★ |
| 14 | 微生物霉变 | 霉菌最适 25–30 °C，而物料起步正是 28 °C | ★★★ |
| 15 | 挥发油损失 | 烘干本质是一场不受控的水蒸气蒸馏 | ★★★ |
| 16 | 褐变与热降解 | 2–3 天 × 50 °C 对热敏性成分是严峻考验 | ★★ |
| 17 | 虫卵 | 50 °C 不足以杀灭虫卵（需 55–60 °C 以上） | ★ |
| 18 | 干裂与碎粉 | 非均匀收缩的内应力导致开裂、物料损失 | ★★★ |
| 19 | 各向异性收缩 | 按干物质守恒反推，实测比理论多缩了 7% | ★★★ |
| 20 | 端面效应 | 端面面积是侧面的 8%（R₀/L = 2/25） | ★★ |
| 21 | 参数与批次不确定 | 题目要求 4 位小数，但物理精度远达不到 | ★★★ |
