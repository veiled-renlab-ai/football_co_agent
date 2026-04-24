# Football Agent — 改进路线（v1，2026-04-24）

> 来源：4 个并行 subagent 独立调研 2024-2026 多智能体 LLM 协作 / training-free 学习 / 高级 prompt engineering / 现成开源项目后的综合结论。

---

## 一、问题定义

现在的 10 个 LLM agent 在 5v5 gfootball 比赛里表现出三类缺陷：

1. **协作差** —— 队友之间抢球、站位重叠、不补位、不传球
2. **对抗差** —— 防守站位乱、不盯人、不封角度、不补防
3. **个体踢球意识差** —— 明明是中锋却回防、禁区前犹豫不射、被贴身不护球

约束：
- **不能 fine-tune 模型权重**（用 API: Doubao / DeepSeek-V3.2 / Kimi via 火山方舟 Coding Plan）
- 单开发者、实时 demo 需求
- 必须在 **prompt / memory / tool / inference-time** 这层动手

---

## 二、三大共识（4 个 subagent 一致）

### 1. "LLM 每 tick 决策" 是反模式
三个独立工作（SwarmBrain / HEP-LLM / PLfB-URI）都收敛到**双层架构**：
- **慢层（Overmind / Coach / Captain）** — 几秒一次，战术决策，读全场
- **快层（Reflex / Player）** — 实时，规则+条件响应，不调 LLM

我们现在的 motor + LLM 是二层，但 fallback 只是"填空白"，**还缺 Captain 这一层**。这是最大的架构缺口。

### 2. 纯 prompt engineering 能到"像样"，但撞墙在亚秒级默契
共识路径：**Hierarchical Captain + ReAct 看评决 + Few-shot 专家片段** 叠加 ROI 最高，能拿到 7–9/10 的提升。
但"二过一时机 / 一脚直塞窗口" 这类 sub-second coordination 是 LLM 延迟+token 成本解不了的 —— 那需要 MARL 训练权重，已超出本项目范围。

### 3. Training-free learning 路线成熟
三个独立 subagent 都推 **"经验自我迭代"** 方向（不改权重，改 prompt + 外部 memory）：
- **ACE** (arxiv 2510.04618) — Generator→Reflector→Curator 进化 playbook
- **ExpeL** (arxiv 2308.10144) — 从成功/失败 trajectory 抽 insights 做 retrieval
- **ICL for LM Agents** (arxiv 2506.13109) — 专家轨迹片段做 few-shot

---

## 三、可以直接偷的 3 个现成工作

| 项目 | 链接 | 偷什么 |
|---|---|---|
| **LCDSP (AAAI 2026)** | [paper](https://arxiv.org/abs/2511.19885) | **直接对口我们的 5v5 gfootball Discrete(19)**。6 个战术风格词汇（Tiki-Taka / Park the Bus / Counter Attack / Positive Attack / Balanced / All-out Attack）和 10 个 style 参数，可以**今天就挪进 team profile 的 system prompt** |
| **ProAgent (AAAI 2024)** | [paper](https://arxiv.org/abs/2308.11339) | 五段式 pipeline：state-grounding → planning → **belief correction** → controller → memory。**"信念修正"正是治协作差的药** — 每 tick 让 agent 回顾"我上次预测队友会干什么、实际他干了什么" |
| **HEP-LLM-play-StarCraft (2025)** | [repo](https://github.com/NKAI-Decision-Team/HEP-LLM-play-StarCraftII) | Expert Tactic Prompt（条件性注入战术片段）+ Chain-of-Summarization（把数值观察压缩成自然语言解说） |

其它可参考：
- **PLfB / URI (NeurIPS 2024)** [paper](https://arxiv.org/abs/2504.19997) — gfootball 11v11，LLM 读战术书→生成伪代码 playbook。警示：**裸 GPT-as-agent 胜率仅 6%，URI scaffolding 才到 37%**
- **SwarmBrain** [repo](https://github.com/ramsayxiaoshao/SwarmBrain) — Overmind + Swarm ReflexNet，**双层架构最干净的参考**
- **AgentScope 狼人杀** [repo](https://github.com/modelscope/agentscope/tree/main/examples/game_werewolf) — 中文 multi-agent prompt 风格参考

---

## 四、三阶段实施路线（按 ROI 从高到低）

### Phase A · 把"纯 prompt 能做的"榨干（~1 周）

#### A1. Hierarchical Captain [🔥 首选]
- 给每队加一个 `captain_agent`，每 5 tick 跑一次，输出 1 句 `team_call`
- 例如："左路 overload" / "高位逼抢 5 秒" / "回收防守" / "换边直塞"
- 塞进每球员 prompt 的 `<current_team_call>` 字段
- **4 个 subagent 都点名的最大单点提升**
- 效果：把 "10 个单打独斗" 变成 "一个团队" 的最小充分条件
- 工作量：~半天（新增一个 LLM 调用 + broadcast + 新 prompt 字段）

#### A2. 看-评-决 Structured ReAct
- 强制 agent 输出 JSON `{observe, evaluate, intent, action}`
- `observe` — 队友/对手威胁 top 3
- `evaluate` — "我们正控球但左路被压" 一句
- `intent` — 宏观意图（"拉边换向"）
- `action` — 才选具体 skill
- 给球场决策 "语义地基"，治"踢球意识差"
- 参考：[ReAct (arxiv 2210.03629)](https://arxiv.org/abs/2210.03629)

#### A3. LCDSP 战术风格词汇
- 把现在的 `BLUE_TEAM_PROFILE` / `RED_TEAM_PROFILE` 替换成 LCDSP 的 6 风格之一
- 加 10 个可量化维度（Win / Goal / Hold-Ball / Pass / Spacing / Shot / Move / ...）
- 基本是免费的升级（直接改文本）

#### A4. Few-shot 专家片段
- 手写 20–30 条 `{situation, role, good_action, bad_action, reason}` JSON
- 按 `{role, zone, has_ball}` 三元组索引存 JSON
- 每 tick 用 embedding / 规则检索 **top-2 最相似片段** 塞进 prompt
- 需要足球经验投入写片段
- 参考：[ICL for LM Agents (arxiv 2506.13109)](https://arxiv.org/abs/2506.13109)

#### A5. Call skill 加 schema
- 自由文本喊话 → 结构化 `call("cover_left" | "press" | "drop_back" | "switch_wing" | ...)`
- Riedl 2025 ([arxiv 2510.05174](https://arxiv.org/abs/2510.05174)) 证明无 schema 的 emergent communication 会退化成噪声
- 同时降低 token 成本

---

### Phase B · Training-free Learning（~1–2 周）

#### B1. 赛后复盘 loop（Reflexion / Multi-Agent Reflexion）
- 每场结束跑一次 `reflector` prompt
- 输入：整场 match log（失球事件、成功反击、失误传球）
- 输出：2–3 条 `<tactic_lesson>`（≤80 字），追加到 role-specific `tactics_memory.json`
- 下一场按角色匹配 top-3 lesson 注入 system prompt
- 参考：[Reflexion (arxiv 2303.11366)](https://arxiv.org/abs/2303.11366)、[Multi-Agent Reflexion (arxiv 2512.20845)](https://arxiv.org/abs/2512.20845)

#### B2. gfootball academy 课程学习
- gfootball 内置 academy 场景（`academy_3_vs_1_with_keeper` / `academy_counterattack_easy` / `academy_corner` 等）
- 先在 `3_vs_1` 跑 20 场，学会传接 + 跑位
- 再切 `counterattack_easy`
- 最后上 5v5
- 每阶段累积的 tactics_memory 带到下一阶段
- 参考：[Curriculum RL (arxiv 2506.06632)](https://arxiv.org/abs/2506.06632)

#### B3. ACE 风格的 playbook 进化
- Generator（比赛 agent）→ Reflector（失败片段打 insight）→ Curator（合并去重进 playbook）
- 每周迭代一次 playbook，不改代码
- 参考：[ACE (arxiv 2510.04618)](https://arxiv.org/abs/2510.04618)

---

### Phase C · 双层架构重构（如果 A + B 还不够）

#### C1. 引入真正的 Reflex 层
```python
def reflex_policy(obs: Observation) -> Optional[Skill]:
    # 条件反射，不调 LLM
    if obs.self_state.has_ball and opponent_distance < 0.05:
        return DribbleToward(protect=True, ...)   # 护球本能
    if obs.self_state.has_ball and in_shooting_zone() and no_keeper_angle():
        return Shoot(...)                         # 面对空门直接射
    if loose_ball_in_my_zone() and i_am_closest():
        return MoveTo(ball.position, sprint)      # 抢球反射
    return None   # 交给 LLM
```
- LLM 只在 `reflex_policy() == None` 时调
- 预期减少 50%+ LLM 调用，同时提升实时反应质量
- 参考：SwarmBrain Overmind + Swarm ReflexNet

#### C2. Belief Correction（ProAgent 风格）
- 每 tick 让 agent 先回顾："我上次预测队友 #3 会 XXX，实际他 YYY"
- 修正过的信念进 memory
- 治"协作差"的根因：agent 不知道队友在想什么

---

## 五、不推荐先做的

| 方案 | 原因 |
|---|---|
| **Best-of-N 推理时采样** | 实时 5v5 每 tick × N 个 LLM 调用是 latency 炸弹。只适合离线蒸馏好的 play 进 memory |
| **Imitation from gfootball scripted bots 直接 BC** | API 层做不了 behavior cloning，只能变成 ExpeL 式的 trajectory→prompt |
| **Mental rollout / World model** | Doubao 的 spatial reasoning 还撑不起复杂推演，且每 tick 多次 API call 的 latency 吃不起 |
| **Fine-tune LLM 权重** | 超出项目约束（API-only） |

---

## 六、一句话结论

> **现在瓶颈不在"模型智商"，在"没有人当教练"。**

最小开工版 = **Phase A 的 A1**：加一个 captain LLM，每 5 tick 给全队 broadcast 一句战术指令。这一步能吃掉最大一块"协作差"的帽子，工作量 ~半天。

---

## 七、推荐的起步选项

- **(A) Captain + team_call broadcast** —— 最小改动见效，~半天
- **(B) 先写 20–30 条 few-shot 专家片段** —— 需要投入足球经验，~1-2 天
- **(C) 先实现赛后 reflector loop 让系统自己学** —— 慢迭代路径，~1 天搭架子
- **(D) 直接重构成双层（A + Phase C 合并）** —— 激进重构，~3-5 天

---

## 附：完整参考文献

### 现成可 fork 的项目
- [LCDSP (5v5 gfootball, AAAI 2026)](https://arxiv.org/abs/2511.19885) · [project page](https://lcdsp-webpage.github.io/LCDSP/)
- [PLfB / URI (gfootball 11v11, NeurIPS 2024)](https://arxiv.org/abs/2504.19997) · [project page](https://plfb-football.github.io/)
- [ProAgent (Overcooked, AAAI 2024)](https://arxiv.org/abs/2308.11339) · [project page](https://pku-proagent.github.io/)
- [HEP-LLM-play-StarCraftII (2025)](https://github.com/NKAI-Decision-Team/HEP-LLM-play-StarCraftII)
- [SwarmBrain (StarCraft II)](https://github.com/ramsayxiaoshao/SwarmBrain)
- [LLM-Coordination benchmark (NAACL 2025)](https://github.com/eric-ai-lab/llm_coordination)
- [SoccerAgent (ACM MM 2025, video QA)](https://github.com/jyrao/SoccerAgent)
- [TiKick (MARL baseline on gfootball)](https://arxiv.org/abs/2110.04507)
- [AgentScope 狼人杀 (中文 multi-agent 参考)](https://github.com/modelscope/agentscope/tree/main/examples/game_werewolf)

### Training-free 方法论
- [ACE — Agentic Context Engineering (arxiv 2510.04618)](https://arxiv.org/abs/2510.04618)
- [ExpeL — LLM Agents Are Experiential Learners (arxiv 2308.10144)](https://arxiv.org/abs/2308.10144)
- [ICL for LM Agents (arxiv 2506.13109, Google)](https://arxiv.org/abs/2506.13109)
- [Reflexion (arxiv 2303.11366)](https://arxiv.org/abs/2303.11366)
- [Multi-Agent Reflexion / MAR (arxiv 2512.20845)](https://arxiv.org/abs/2512.20845)
- [SPCT Inference-Time Reward (arxiv 2504.02495, DeepSeek)](https://arxiv.org/abs/2504.02495)
- [Best-of-Poisson (arxiv 2506.19248, reward hacking warning)](https://arxiv.org/abs/2506.19248)
- [AgentRM (arxiv 2502.18407, ACL 2025)](https://arxiv.org/html/2502.18407v1)
- [MultiAgentBench / MARBLE (arxiv 2503.01935)](https://arxiv.org/abs/2503.01935)

### Prompt engineering 核心技术
- [ReAct (arxiv 2210.03629)](https://arxiv.org/abs/2210.03629) · [project](https://react-lm.github.io/)
- [HMAW hierarchical multi-agent workflow](https://openreview.net/pdf?id=RVvXOrP2qm)
- [PARTNERMAS (arxiv 2509.24046)](https://arxiv.org/pdf/2509.24046)
- [Hierarchical Language Agent (AAMAS 2024, Overcooked)](https://www.ifaamas.org/Proceedings/aamas2024/pdfs/p1219.pdf)
- [LLM-Hanabi Theory of Mind (arxiv 2510.04980)](https://arxiv.org/abs/2510.04980)
- [Collab-Overcooked (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.249.pdf)
- [Guandan Theory of Mind (arxiv 2408.02559)](https://arxiv.org/abs/2408.02559)
- [Inner Monologue (arxiv 2207.05608)](https://arxiv.org/abs/2207.05608)
- [Voyager skill library (arxiv 2305.16291)](https://arxiv.org/abs/2305.16291)
- [Riedl Emergent Coordination (arxiv 2510.05174)](https://arxiv.org/abs/2510.05174)
- [LLM-PySC2 (NeurIPS 2025)](https://arxiv.org/abs/2411.05348)
- [Multi-Agent Collaboration Survey (arxiv 2501.06322)](https://arxiv.org/abs/2501.06322)
