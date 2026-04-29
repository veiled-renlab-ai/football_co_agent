# Football RL — 5v5 LLM Agent Football Simulation

5 对 5 足球仿真，每个球员由独立的 LLM agent 控制，运行在 Google Research Football 环境上。

---

## 架构

```
Brain (LLM)       每 25 ticks 决策一次，输出 intent-level 技能指令
   ↓
Motor (Python)    每 tick 执行，将技能转化为底层 gfootball action
   ↓
Body (gfootball)  物理仿真 + 渲染，每 tick 推进环境
```

**12 种技能**：MoveTo · PassTo · Shoot · Mark · Press · Tackle · Call · DribbleToward · HoldPosition · Intercept · Clear · GoalkeeperDive

**Stop-World 模式**：每轮所有 agent 收到同一帧观测 → 并行 LLM 决策 → 全部完成后 env 执行 25 ticks → 下一轮。彻底消除 stale-obs 问题，代价是比赛节奏取决于最慢 agent 的 LLM 延迟。

**Egocentric 感知**：FOV ±105°，视野外的球员不可见。

---

## 环境要求

- **WSL2** — Ubuntu 22.04（gfootball 不支持原生 Windows）
- **Python 3.10**（gfootball 2.10.2 在 3.11+ 有兼容问题）
- **GPU 可选**（渲染走 EGL 离屏，不强依赖 GPU）

---

## 安装

```bash
# 在 WSL 里
sudo apt-get install -y libgl1-mesa-dev libglib2.0-dev \
    python3-dev python3-venv cmake build-essential

python3.10 -m venv /home/$USER/football-env
source /home/$USER/football-env/bin/activate

pip install --upgrade pip
pip install --no-build-isolation -r requirements.txt
```

---

## 配置

```bash
cp .env.example .env
```

编辑 `.env`，填入 LLM API Key：

```env
# 选一个 provider
LLM_PROVIDER=volcengine          # 或 minimax / aliyun_token_plan

# 火山方舟（默认）
VOLCENGINE_API_KEY=your-key-here
VOLCENGINE_MODEL=doubao-seed-1.6-250615

# MiniMax（备选）
MINIMAX_API_KEY=your-key-here
```

---

## 运行

### 进入 WSL

**方式 A：直接进 WSL 终端**
```bash
wsl
source /home/dfgfd/football-env/bin/activate
cd /mnt/c/Users/dfgfd/Desktop/football_RL
```

**方式 B：从 PowerShell 一行运行**
```powershell
wsl -e bash -c "source /home/dfgfd/football-env/bin/activate && cd /mnt/c/Users/dfgfd/Desktop/football_RL && python3 -u scripts/run_eval_server.py"
```

---

### （可选）增强 gfootball 渲染效果

首次运行前打一次即可，自动备份原始文件：

```bash
python3 patch_gfootball_shaders.py
```

修改内容：亮度 +35%、饱和度提升、天空蓝色、环境光暖白色调、场地灯光更强。  
还原：将 `data/media/shaders/` 下的 `.orig` 文件重命名覆盖回去。

---

### 启动 Eval 平台

```bash
python3 -u scripts/run_eval_server.py
```

浏览器打开：`http://localhost:8000`

---

## Eval 平台使用

1. **New Run** — 填写参数，勾选 **"Render gfootball 3D 窗口"** 开启实时渲染
2. 点击 **Start**，UI 自动切到 3D 视图（MJPEG 流）
3. 左侧面板实时显示每个 agent 的决策、技能、推理过程
4. 比赛结束后可查看逐帧回放和统计数据

**主要参数：**

| 参数 | 说明 |
|---|---|
| Episodes | 跑几局 |
| Ticks per stream | 每几 ticks 推一次观测事件（建议 5） |
| Render | 开启 gfootball 3D 渲染（关闭时只有 2D 俯视图） |
| Stop-World | 暂停 env 等所有 agent 决策完毕（推荐开启） |

---

## 项目结构

```
football_agents/
├── llm_client.py          # LLM 调用，多 key 轮询，支持火山方舟/MiniMax/阿里云
├── player_agent.py        # 单个球员 agent，Brain + Motor 协调
├── motor.py               # 技能状态机，每 tick 执行
├── perception.py          # 观测解析，egocentric FOV 过滤
├── skills.py              # 12 种技能定义 + is_valid() 校验
├── prompts.py             # System prompt 生成，含锚点站位注入
├── multi_agent_runner.py  # 10 agent 并发 + stop-world 调度
├── fallbacks/             # 无 LLM 响应时的规则 fallback（每个位置独立）
│   ├── blue_gk/lb/cb/rm/cf.py
│   └── red_gk/lb/cb/rm/cf.py
├── players/               # LLM 人格 + 站位纪律（锚点坐标）
│   └── disciplines.py
└── eval_platform/
    ├── server.py          # FastAPI + WebSocket
    ├── harness.py         # Episode 调度
    └── static/index.html  # Web UI（Alpine.js + Chart.js）

scripts/
└── run_eval_server.py     # 启动入口

patch_gfootball_shaders.py # gfootball 渲染增强脚本
```

---

## LLM Provider

| Provider | 环境变量前缀 | 备注 |
|---|---|---|
| 火山方舟 | `VOLCENGINE_` | 默认，doubao-seed 系列，支持 tool use |
| MiniMax | `MINIMAX_` | 备选 |
| 阿里云百炼 Token Plan | `ALIYUN_TOKEN_PLAN_` | 多 key 轮询，提升 RPM |

切换 provider：修改 `.env` 中的 `LLM_PROVIDER=` 即可。
