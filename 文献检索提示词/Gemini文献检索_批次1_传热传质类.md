<!-- 批次 1/4 · 传热传质类（因素 1–7）· 全文即为提示词，Ctrl+A 复制后直接粘贴给 Gemini / GPT 即可 -->

【角色】
你是一位干燥技术与传热传质领域的文献检索专家，熟悉中药材/农产品热风干燥、
多孔介质传热传质、食品工程与干燥设备设计的中英文文献。

【研究背景】
我正在为一道数学建模竞赛题建立「圆柱形中药材热风烘干」的数学模型。现有模型如下：
- 几何：一维轴对称无限长圆柱（半径 2 cm，长 25 cm）
- 方程：耦合的径向导热方程 + 有效扩散型水分迁移方程
- 边界：表面对流换热（h = 25 W/(m²·K)）+ 表面对流传质（h_m = 8×10⁻⁷ m/s）
- 物性：经验公式给出的 ρ(C)、c_p(C)、k(C)、D(C,T)
- 工况：烘房温度 28 °C 升至 50 °C，烘干时长 2–3 天，
  含水率（干基）从 2.55 kg/kg 降到 0.15 kg/kg
- 已知特征：水分扩散系数随含水率下降而骤降（C 从 2.55 降到 0.15 时相差约 265 倍），
  干燥呈"前快后慢"、干燥前沿从表面向中心推进的形态

题目只给了上述理想化条件，明确省略了大量真实物理因素。
我现在需要为**本批次的 7 个被省略因素**找到可靠的学术文献支撑，
用于论文的「模型评价与推广」章节。

【核心任务】
针对下面列出的每一个因素，检索并给出 3–5 篇**高质量、真实存在、可溯源**的文献。
每个因素我需要的不是泛泛的综述，而是能直接用于建模的东西：
  (a) 该因素的标准数学描述（控制方程 / 本构关系 / 判据）
  (b) 典型参数取值范围（按物料类别给出：中药材 / 果蔬 / 谷物 / 通用多孔介质）
  (c) 该因素对干燥时间或物料温度的影响量级（有定量结论最好）
  (d) 是否可忽略的判据（例如某个无量纲数小于某值时可以忽略）

【检索范围与质量要求】
1. 中英文文献都要，英文文献优先（该领域英文文献更系统）
2. 优先顺序：权威期刊综述 > 高被引经典文献 > 近 5 年新文献 > 学位论文
3. 本批次重点期刊方向：
   Journal of Food Engineering、Drying Technology、
   International Journal of Heat and Mass Transfer、
   Food and Bioproducts Processing、Journal of Food Science、
   International Journal of Thermal Sciences、
   中文的《农业工程学报》《工程热物理学报》《食品科学》《干燥技术与设备》《化工学报》
4. 时间范围以近 20 年为主；奠基性经典文献（如 Luikov、Crank 等）不受年份限制
5. 明确区分两类文献，并在表中标注：
   【方法类】提供建模方法与方程来源的
   【数据类】提供参数取值、实验数据、经验关联式的

【本批次待检索因素：7 个】
（编号、名称、物理含义、中英文检索关键词）

1. 蒸发潜热（表面蒸发吸热导致的降温效应；本模型未引入汽化潜热项）
   物理内涵：水在表面蒸发时吸收大量潜热，使物料表面温度低于热风温度，
             恒速干燥期表面应稳定在湿球温度附近
   中：蒸发潜热 表面能量平衡 干燥 物料温度 湿球温度 蒸发冷却
   英：latent heat of vaporization drying, evaporative cooling,
       wet bulb temperature, surface energy balance drying,
       drying rate constant rate period

2. 内部蒸发与水蒸气渗流（水在物料内部汽化后以气体形式迁移）
   物理内涵：多孔物料内部的水可能在内部就汽化，以水蒸气形式通过孔隙向外渗流
             （压力梯度驱动，符合达西定律），而非仅以液相扩散迁移；
             这引入"内部气压"这一本模型没有的变量
   中：多孔介质 内部蒸发 水蒸气扩散 达西流 干燥 多相迁移 孔隙压力
   英：internal evaporation porous media, vapor diffusion drying,
       Darcy flow porous media, multiphase drying model,
       pore pressure drying, vapor pressure driven moisture

3. Soret / Dufour 交叉效应（温度梯度驱动水分、浓度梯度驱动热流）
   物理内涵：完整的热湿耦合模型（Luikov 方程组）中，
             温度梯度本身能驱动水分迁移（Soret，水往冷处跑，
             干燥初期与干燥方向相反），浓度梯度能驱动热流（Dufour）
   中：热湿耦合 热扩散效应 湿扩散 Luikov 模型 不可逆热力学 交叉效应
   英：Soret effect drying, Dufour effect, thermodiffusion moisture,
       Luikov model coupled heat mass transfer, irreversible thermodynamics,
       Luikov number, coupled heat and moisture transfer

4. 辐射换热（烘房壁面与物料之间的热辐射）
   物理内涵：题设只给了"对流换热系数"，但热风烘房中壁面与物料间的辐射不可忽略；
             工程上的复合换热系数往往已包含辐射，需要确认这一点
   中：干燥 辐射换热 复合换热系数 对流辐射耦合 发射率
   英：thermal radiation drying, combined convection radiation drying,
       radiative heat transfer dryer, effective heat transfer coefficient,
       emissivity food material

5. 表面结壳 / 硬化（表面过干形成致密层阻碍内部水分迁出）
   物理内涵：表面干燥过快会形成致密低渗透性硬壳，把内部水分锁死，
             是中药材烘干最常见的失败模式（外干内湿、切开后芯还是软的）
   中：表面结壳 硬化 干燥 表皮效应 外干内湿
   英：case hardening drying, crust formation, surface crust,
       skin formation food drying, surface hardening drying

6. 结合水与自由水（水分存在状态对脱除难易的影响）
   物理内涵：物料中的水分为自由水、物理结合水、化学结合水三类，
             干燥后期主要剩余结合水，需要更高能量才能脱除；
             本模型仅用一个"干基含水率"描述所有水分
   中：结合水 自由水 水分状态 核磁共振 低场核磁 水分活度 干燥
   英：bound water free water drying, water state,
       NMR moisture distribution, water activity drying,
       moisture binding state, LF-NMR drying

7. 玻璃化转变（干燥后期物料由橡胶态转为玻璃态，扩散系数骤降）
   物理内涵：含水率降到临界值以下时物料发生玻璃化转变，
             分子链段被冻结，水分扩散系数出现数量级突降；
             本模型给出的 D(C,T) 是光滑指数函数，没有这个"断崖"
   中：玻璃化转变 干燥 橡胶态 玻璃态 状态图 分子流动性 结块
   英：glass transition drying, rubbery glassy state,
       state diagram food, stickiness drying, molecular mobility,
       glass transition temperature food

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
- <定量结论，例如"某物料在 X 条件下潜热使升温速率降低 Y%"，须注明文献编号>

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
1. 最后给出一张汇总表：7 个因素 × 文献数量 × 可信度 × 是否建议纳入论文推广章节。
2. 标出其中「文献丰富、结论明确、最值得深入建模」的因素（本批次预计是第 1、7 条）。
3. 标出其中「文献稀少、争议大、建议仅作定性提及」的因素。
4. 如某因素存在相互矛盾的研究结论，请明确指出矛盾点及其可能的成因
   （物料种类差异？实验条件差异？测量方法差异？）。
5. 特别说明：第 1 条（蒸发潜热）与第 3 条（Soret/Dufour）都属于「耦合传热传质理论」
   体系，请说明二者在 Luikov 方程组中的相对地位与量级对比，
   并给出判断二者可否忽略的无量纲数（如 Luikov 数、Kossovitch 数等）。
