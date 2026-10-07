# Agora 深入学习文档

> 上游：`yifanzhang-pro/Agora`（Yifan Zhang 等，NVIDIA）
> 本 fork：`jasonchen505/Agora`（2026-10-07 创建，默认分支为 `master`）
> 论文："Agora: Git as Shared Memory for Collective AutoResearch"，arXiv:2609.18094（2026-09-16 发布）；NeurIPS 2026 AutoMLR Workshop Oral（top 10%）
> License：Apache 2.0
> 学习日期：2026-10-07｜状态：只读学习，master 未动，本文件位于 `notes/learning` 分支
>
> 特别说明：这个仓库**本身不包含研究代码**，它是论文的项目网站（`index.html` + `README.md` + `assets/` 四张图）。本文档的学习对象是论文的思想、机制设计与验证实验，而非代码实现。

---

## 1. 一句话定位

Agora 回答的问题是：**一群互不相识、没有中央规划、连对话都不共享的研究 agent，如何像一个真正的科研社区一样协作？** 它的答案是：把科研过程本身做成 Git 里的一个 **append-only 有向无环图（DAG）**——commit 即贡献，parent 即"我站在谁的肩膀上"。

## 2. 为什么需要这个

多 agent / AutoResearch 系统通常面临一个协作困境：

- 共享完整对话历史 → 上下文爆炸，且 agent 之间互相干扰；
- 各自为战 → 重复劳动，前人的失败教训（negative result）丢失；
- 中央规划器 → 单点瓶颈，且与"开放式探索"的精神相悖。

Agora 的切入点是：科研社区在人类世界本来就是**异步、去中心化、通过出版物（不可变记录）协作**的。Git 恰好是现成的、经过大规模验证的 append-only DAG 基础设施——不需要发明新协议，只需要规定"如何用 Git 记录科研"。

## 3. 三个核心机制

### 3.1 Git-backed contribution history（Git 支撑的贡献历史）

- 每个 contribution（实验结果、失败、假设、复现、报告）是一个**不可变 Git commit**。
- commit 的 **parent 指向它所基于的工作**——于是整个研究过程自然形成 DAG，而不是线性历史。多 parent 节点表示"融合了多条线的工作"。
- 一个 SQLite contribution index 提供可搜索视图（领先结果、被冷落的分支、验证状态），且**可以完全从 Git 重建**——Git 是唯一事实来源，索引只是物化视图。
- 设计要点：不可变性保证了记录可信（事后无法篡改"我当初试过什么"）；parent 链接让"思想谱系"可追溯（最佳方法的 145-commit 祖先链横跨 15 个账户就是证明）。

### 3.2 Evidence scores（证据分数）

- 其他账户对你工作的**复现（reproduction）与验证（verification）**会累积到父节点的 evidence score 上。
- **自引排除**：自己复现自己的工作不加分——防止刷分。
- 这个分数追踪的是"可复现性"和"被复用度"，而不是论文引用那样的社会性指标。在 weight-transfer run 中，165 次复现覆盖 95 个目标、零失败报告，evidence score 实际上成了一个去中心化的"结果可信度"信号。

### 3.3 `analyze()` recommendations（分析与推荐）

- 一次 API 调用，worker 可以看到：当前领先结果、被复用最多的贡献、有争议的验证、开放假设。
- **UCB 风格排序**：同时推荐"领先者"（exploit）和"未充分探索者"（explore），显式对抗"所有人都挤在一条 lineage 上"的马太效应。
- 论文里有一个真实的干预案例：连续 5 天大家都挤在 bigram 方法上之后，作者加入了展示"搜索集中度"和"被忽视分支"的视图；**一天之内**就有 worker 转向 state-space 方向并打出 1.9028 bpb。这说明推荐视图确实能改变群体的探索分布。

项目方只定义指令、指标、产物要求和安全边界；**实验选什么、怎么做，完全由 worker 自己决定**。Agora 只提供共享记录，不提供中央规划。

## 4. 验证实验：weight-transfer run

### 4.1 任务设计

- **目标模型**：一个 frozen 的 14 层 attention–SSM hybrid，hidden size 672，7 heads，共 119,572,320 参数——其**配置与所有 donor 都不匹配**（无法直接拷权重）。
- **资源**：141 个开源 donor 模型（534 GB，32 个架构家族）。
- **提交物**：一个 `transfer(model, config)` 函数；evaluator 在 200 个 FineWeb-Edu 文本上以 **bits-per-byte** 打分。
- **禁区**：禁止预训练、禁止微调、禁止修改 evaluator。
- **社区**：13 个 LM worker，近 12 天，两页纸的 brief + evaluator + 共享图，**无指派任务、无中央规划**，共发布 1,703 个 contributions。

任务设计很讲究：它是一个"真问题"（跨架构权重迁移），有客观可验证的标尺（bpb），但又留足了开放性（方法完全自由）。

### 4.2 里程碑（development evaluator，bpb）

| 里程碑 | bpb | 关键动作 |
|---|---|---|
| 随机初始化 | 3.3923 | 基线 |
| 首次有效尝试 | 4.6784 | 切片复制 GPT-2/Mamba 权重——**比随机还差，作为 negative result 发表** |
| 30 分钟后 | 2.5151 | 从 GPT-2 预测中读出 unigram 先验，残差子层清零 |
| 6 小时内 | 1.93 | 4 个账户把思路扩展到 3/6/12/24 前缀下的 bigram 统计 |
| 5 月 1 日 | 1.904 | Cerebras-GPT donor，28 个上下文，SVD 中的 power iteration |
| 截止（5 月 8 日） | **1.899** | 通过对 attention/FFN/SSM 的稀疏编辑重新启用子层 |

参考线：训练好的 GPT-2 124M 约 1.0 bpb。本次最佳结果补上了从随机初始化到 GPT-2 之间 **62% 的差距**——且全程无训练数据、无梯度更新。

注意第二行的细节：第一次尝试**比随机还差**，但被作为 negative result 发表了出来。这正是 append-only 记录的价值——失败本身成为公共知识，避免 13 个 worker 重复踩坑。

### 4.3 最佳方法（两阶段）

- **Stage A**：不从 donor 的**参数**出发，而从 donor 的**预测**出发。取 6 个共享 GPT-2 词表的 donor，在 28 个单 token 上下文中查询 next-token log-softmax，融合成 50257×50257 的上下文平均 bigram 表，中心化后做 randomized SVD 截断到 rank 671，因子作为 target 的 input embedding 和 output head；所有子层清零。
- **Stage B**：用稀疏的、确定性的编辑重新启用子层——在 hidden state 的 96 维 band 上操作：attention 变成一个 band 上的均匀因果均值池，每个 SSM block 退化为门控深度可分离因果卷积，第 0 层的 FFN 接收 GPT-2 small 第一层 MLP 的 SVD 投影切片。

一句话：**先搭一个"统计先验"的骨架（bigram→embedding/head），再用稀疏手术把子层接回来**。最终 145-commit 的祖先链横跨 15 个账户——没有任何单个 worker 能独立走完这条路。

### 4.4 协调动力学（论文最有意思的部分）

截止时图有 1,703 节点、1,894 边、149 个多 parent 节点；一个连通分量占 98.9% 节点。

- **快速 exploit**：前 8 次改进占总增益约 70%；**首日的 18 个有效 contribution 占约 98%**，剩下 1,106 个只啃下最后 0.03。——典型的"低垂果实先被摘完"。
- **搜索集中**：绝大多数后续工作都延伸同一条 lineage，其他分支几乎无人问津。
- **平行再发现**：696 对"不同账户打出完全相同分数"的案例中，约 63% 发生在 1 小时内、80% 在 6 小时内——说明大家在没有沟通的情况下会独立撞上同一个想法（也说明共享图的信息传播有延迟/盲区）。
- **Agent 的自我解释**（未经独立验证）：多个方法收敛在 1.90 bpb 附近，agent 们把平台期归因于"evaluator 全局线性"和"target 子层未被充分利用"。

作者坦承：共享图和分析视图对发现效率的**因果效应**还没有被严格测量，论文提出用 matched comparison（相同算力预算下有/无共享图的对照）来测——这是留给未来工作的钩子。

## 5. 方法论观察与批判性思考

**强的地方**：

1. **用现成基础设施解决新问题**：Git 的不可变性、DAG、分布式同步都是现成的；SQLite 索引可重建意味着没有单点状态。这是"借力"而非"重造"的典范。
2. **任务选择真实**：weight transfer 是真问题、有客观标尺、方法开放——比 toy benchmark 更有说服力。
3. **诚实报告负面现象**：搜索集中、平行再发现、首日拿走 98% 增益——这些"不好看"的发现都写出来了，且直接导向了机制改进（多样性视图）。
4. **Negative result 一等公民**："比随机还差"的尝试被发表而非丢弃，这是 append-only 科研记录最核心的价值主张。

**可商榷 / 值得深想的地方**：

1. **因果归因缺失**：1.899 bpb 有多少归功于 Agora 机制、多少归功于"13 个聪明 worker + 12 天算力"？作者自己承认需要 matched comparison。在此之前，"机制有效"的结论是观察性的。
2. **Evaluator 的代表性**：200 个 FineWeb-Edu 文本的 development evaluator，且"最终分数提升小于跨硬件 observed variation"（README 原话）——最佳方法的领先优势可能在噪声范围内。1.899 vs 1.9028 的差距要谨慎解读。
3. **Evidence score 的博弈**：自引排除是第一步，但共谋互引（你复现我、我复现你）、"复现"本身的质量参差，都是开放问题。165 次复现零失败——是真 robust，还是复现标准太松？
4. **UCB 推荐的粒度**："未充分探索"的定义依赖索引视图的设计；5 天后才加入多样性视图，说明初始的 analyze() 其实**没有**有效对抗集中化——机制是迭代补上去的，不是先验完备的。
5. **可扩展性未知**：13 个 worker、1700 节点时工作良好；1000 个 worker、百万节点时，SQLite 索引、图遍历、推荐计算是否还撑得住？论文没讨论。
6. **与人类科研的类比边界**：人类科研的"发表"有同行评审过滤噪音；Agora 里任何 worker 都能 commit，噪音控制完全依赖 evidence score 和推荐算法——这个类比在激励设计上还有距离。

## 6. 与其他工作的关系（个人笔记）

- 和 **rules-vs-examples**（同期 fork 的另一个仓库）放在一起看很有意思：一个研究"单个 LLM 如何从规则/例子学习"，一个研究"一群 agent 如何通过共享记录学习"——分别是**个体学习**与**集体学习**的对照。
- 和 **humanize**（flow 编排层）的关系：humanize 解决的是"单个 agent 如何执行复杂流程"，Agora 解决的是"多个 agent 如何共享记忆"。两者正交，理论上可以叠加：humanize 的 flow 跑在 Agora 的共享图上。
- 潜在延伸：如果把 Agora 的思想用到**代码 agent 集群**（如 multi-agent SWE），"commit 即 PR 草稿、parent 即依赖的 issue/PR"是很自然的映射；evidence score 对应"复现/CI 通过"，analyze() 对应"任务分配推荐"。

## 7. 待探索

- [ ] 精读论文全文（重点：analyze() 的 UCB 具体形式、evidence score 的数学定义、matched comparison 的实验设计）
- [ ] 上游是否有后续 commit（机制代码是否会开源——目前仓库只是网站）
- [ ] 思考：append-only DAG + evidence score 能否用在个人的长期研究/工程笔记系统里（把"实验记录"变成可追溯的 DAG）

---

*本文档为个人学习笔记，基于 2026-10-07 对上游 HEAD（`8ac28a5`，2026-09-29）的 README / 项目网站走读。论文观点均为转述，数字引自 README；网站与 README 内容基本一致。*
