# Football Agent Simulation — Development Plan

> Living document. Updated as decisions evolve. Last updated: 2026-04-22.

---

## 1. Vision (one sentence)

Build a multi-agent football match where **each player is an autonomous LLM agent** acting on **Google Research Football's 3D physics engine** through an **intent-level Skill API** that mirrors how real human players think (strategy in the cortex, motor skills in the cerebellum).

---

## 2. Core architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       MATCH RUNNER                                │
│  (manages clock, dispatches per-player decision cycles)           │
└──────────────────────────────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
   PLAYER #1                PLAYER #7               PLAYER #11
   (LLMPlayer)              (LLMPlayer)             (LLMPlayer)
  ┌──────────┐            ┌──────────┐            ┌──────────┐
  │  BRAIN   │            │  BRAIN   │            │  BRAIN   │
  │ (LLM)    │            │ (LLM)    │            │ (LLM)    │
  │ tool use │            │ tool use │            │ tool use │
  └────┬─────┘            └────┬─────┘            └────┬─────┘
       │ picks Skill           │                       │
       ▼                       ▼                       ▼
  ┌──────────┐            ┌──────────┐            ┌──────────┐
  │  MOTOR   │            │  MOTOR   │            │  MOTOR   │
  │  (state  │            │ (state   │            │  (state  │
  │ machine) │            │ machine) │            │ machine) │
  └────┬─────┘            └────┬─────┘            └────┬─────┘
       │ atomic action         │                       │
       └───────────────────────┴───────────────────────┘
                                 │
                                 ▼
                  ┌────────────────────────────┐
                  │  GFOOTBALL ENV (3D physics)│
                  │  + PERCEPTION FILTER       │
                  │    (god view → egocentric  │
                  │     observations per player)│
                  └────────────────────────────┘
```

**Three layers per player**, mirroring human cognition:
| Layer | Frequency | Implementation | Maps to |
|---|---|---|---|
| Brain (LLM) | every 5-10 env ticks (~0.5-1s) | Anthropic-compatible LLM with tool use | Prefrontal cortex / strategic thinking |
| Motor (state machines) | every env tick (~0.1s) | Python state machines per skill | Motor cortex + cerebellum |
| Body (gfootball) | every env tick | gfootball's C++ engine | Skeletal muscle / physics |

The **Perception module** filters the env's god-view (full state of all 22 players + ball) into what THIS player can realistically see (FOV cone, distance, occlusion, attention cap) — preventing "wallhack AI" play.

---

## 3. Stack decisions (locked)

| Concern | Choice | Rationale |
|---|---|---|
| Sim env | **gfootball 2.10.2** in WSL Ubuntu 22.04 | Only open-source 11v11 3D football env; smoke-tested working |
| LLM provider | **火山方舟 (Volcengine ARK)** default; **MiniMax** fallback | User's available API keys; both serve OpenAI-compatible API |
| LLM model | **Doubao-Seed-1.6** or **DeepSeek-V3.2** via 火山方舟 | Strong function-calling, fast (~1-2s), Chinese-friendly |
| LLM SDK | **`openai` Python SDK** (point at provider-specific `base_url`) | Provider-agnostic, supports tool use uniformly |
| Language | **Python 3.10** | Matches gfootball / Ubuntu 22.04 default |
| Config | **`.env` file** + `python-dotenv` | Standard, keys never in code |
| Decision rate | **LLM decision every 5 env ticks** (~0.5s) | Balances LLM latency vs reactivity. Tunable. |

---

## 4. Module layout

```
football/
├── DEV_PLAN.md                     # this file
├── requirements.txt                # pip deps
├── .env.example                    # template for API keys (copy to .env, never commit)
├── setup_wsl_env.sh                # ✅ existing — one-shot WSL bootstrap
├── smoke_test_gfootball.py         # ✅ existing — confirms gfootball engine works
└── football_agents/                # main package
    ├── __init__.py                 # architecture overview docstring
    ├── perception.py               # 👁️  Observation dataclass + EgocentricFilter
    ├── skills.py                   # 🦵  Skill protocol + 12 v0 skills
    ├── motor.py                    # 🧠  Motor controllers (Skill → atomic action)
    ├── env.py                      # gfootball wrapper exposing our API
    ├── llm_client.py               # OpenAI-SDK wrapper for 火山方舟 / MiniMax
    ├── prompts.py                  # System prompts per role (CB, CM, ST...)
    ├── agent.py                    # LLMPlayer: perceive → think → act loop
    └── runner.py                   # Match runner: orchestrate N agents
└── scripts/
    └── demo_single_agent.py        # entry: 1 LLM player vs gfootball bots
```

---

## 5. Implementation phases

### Phase 1 — Contracts (THIS SESSION)
**Files:** `requirements.txt`, `football_agents/__init__.py`, `football_agents/perception.py`, `football_agents/skills.py`
**Goal:** typed data structures + Skill protocol; no behavior yet.
**Success criteria:**
- Imports cleanly in venv
- `from football_agents.skills import ALL_SKILLS` lists 12 skills
- `Observation` dataclass round-trips through dict serialization

### Phase 2 — Motor layer
**Files:** `football_agents/motor.py`
**Goal:** state machines that translate Skills into gfootball's `Discrete(19)` action sequences. Implement 4 highest-value controllers: `MoveTo`, `DribbleToward`, `PassTo`, `Shoot`.
**Success criteria:**
- `MoveToController(MoveTo(target=(0.5, 0.0))).step(obs_dict)` returns sensible directional actions
- Standalone test: random Skill stream → no crashes over 1000 ticks

### Phase 3 — Env wrapper + perception filter
**Files:** `football_agents/env.py`, complete the `EgocentricFilter` in `perception.py`
**Goal:** clean API for agents: `env.observe(player_id) → Observation`, `env.act(player_id, skill) → status`
**Success criteria:**
- A scripted "always pass to nearest teammate" agent runs `academy_3_vs_1_with_keeper` to completion

### Phase 4 — LLM agent loop
**Files:** `football_agents/llm_client.py`, `football_agents/prompts.py`, `football_agents/agent.py`
**Goal:** `LLMPlayer` class — perceives, calls LLM with skill tool defs, parses choice, dispatches.
**Success criteria:**
- Demo: 1 LLM player on `academy_empty_goal_close` scores within 200 ticks (proves loop > random)
- Logs show LLM reasoning and skill choices each decision tick

### Phase 5 — Multi-agent
**Files:** `football_agents/runner.py`, expand `scripts/`
**Goal:** scale 1 → 2 → 5 → 11 LLM agents, observe emergent coordination
**Success criteria:**
- 2 LLM agents same team complete a verified pass-and-shoot on `academy_3_vs_1_with_keeper`
- 11v11 LLM team is competitive vs gfootball's "easy" built-in opponent

### Phase 6 — Polish + investor demo
- Bridge gfootball state stream → teammate's Three.js renderer
- Match clock, score display, agent reasoning overlay
- Replay mode

---

## 6. Open design questions

1. **Skill granularity** — Currently intent-level (`pass_to(player_id)`). Too coarse → LLM has no creativity; too fine → can't keep up with tick rate. Will adjust based on observed play in Phase 4.
2. **Communication channel** — `Call()` skill broadcasts to teammates within ~30m. Should opponents "overhear" within ~10m? Toggle for realism.
3. **Memory horizon** — Default short-term memory: 30 ticks (~3s). Too short → "amnesiac" play; too long → stale info. Tunable.
4. **Set pieces** — kickoff, throw-in, free kick. Phase 5+ concern; for now skip the gfootball scenarios that involve them.
5. **LLM cost / rate limits** — User said cost is not a concern, but provider rate limits matter for 11v11 (≥11 calls per decision tick). May need batching or a smaller "tactical brain" shared across nearby teammates.

---

## 7. Project success criteria (north star)

The project is "successful" when **at least 3** of these are demonstrable:

1. ✅ Single LLM agent scores on `academy_empty_goal_close` (proves perceive-think-act loop)
2. 2 LLM agents complete a verified pass-and-shoot (proves cooperation)
3. 11v11 LLM team beats gfootball's "easy" built-in opponent (proves tactical depth)
4. Coordination is **interpretable** from `Call()` chat logs (LLM advantage over RL: explainability)
5. Demo video bridged through Three.js renderer (investor-ready)

---

## 8. Out of scope (v0)

- Training / fine-tuning LLMs (we use them off-the-shelf)
- Vision-language perception (renders + VLM) — too slow, sticking to filtered structured observations
- Player fatigue/injury modeling beyond gfootball's built-in stamina
- Coach/manager-level meta-agents (could be Phase 7+)
- Real-match data integration

---

## 9. Glossary

- **Skill** — an intent-level action chosen by the LLM (e.g., `pass_to(7, 'short')`), not a low-level button press.
- **Motor controller** — state machine per Skill that emits gfootball atomic actions over multiple ticks.
- **Egocentric observation** — the filtered view of the world from one player's perspective, not god view.
- **Tick** — one step of the gfootball env (~0.1s sim time).
- **Decision tick** — one LLM decision cycle (default every 5 env ticks).
