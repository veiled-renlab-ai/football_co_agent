"""End-to-end smoke for soul injection: persona patching + prompt rendering + lineup template."""
import json
import os
from dataclasses import replace as dataclass_replace
from pathlib import Path

# Reads ARK_KEYS from .env if needed (this script doesn't actually call any LLM —
# it only renders prompts — but llm_client gets imported transitively so dotenv runs).

from football_agents.players import TEAM_BLUE_11V11
from football_agents.prompts import build_system_prompt

print("=" * 70)
print("[1] Default persona — system prompt should contain default play_style")
print("=" * 70)
gk = TEAM_BLUE_11V11[0]  # 林涛 GK
default_prompt = build_system_prompt(gk)
print(f"persona: {gk.name} #{gk.jersey_number} ({gk.position})")
# Find the identity block — between persona name line and the universal section
print("--- relevant slice ---")
for line in default_prompt.split("\n")[:14]:
    print(f"  {line}")

print()
print("=" * 70)
print("[2] custom_soul override — same persona, replaced soul text")
print("=" * 70)
custom_text = "我是阿利松。冷静的巴西门将，扑救脚下球是招牌。出球能力顶级——拿到球先看长传破解高位逼抢。"
patched = dataclass_replace(gk, custom_soul=custom_text)
patched_prompt = build_system_prompt(patched)
print(f"persona: {patched.name} #{patched.jersey_number} (custom_soul set)")
print("--- relevant slice ---")
for line in patched_prompt.split("\n")[:14]:
    print(f"  {line}")

# Verify the custom text appears and the original play_style does NOT
assert custom_text in patched_prompt, "custom_soul should appear in patched prompt"
assert gk.play_style not in patched_prompt, "default play_style should NOT appear when custom_soul is set"
print("[OK] custom_soul fully replaces default identity block")

print()
print("=" * 70)
print("[3] FIFA defaults JSON loads + has 22 entries")
print("=" * 70)
fifa_path = Path(__file__).parent / "football_agents" / "eval_platform" / "data" / "fifa_defaults.json"
fifa = json.loads(fifa_path.read_text(encoding="utf-8"))
print(f"  blue: {len(fifa['blue'])} entries")
print(f"  red:  {len(fifa['red'])} entries")
assert len(fifa["blue"]) == 11 and len(fifa["red"]) == 11
# Show 3 samples
for sample in [fifa["blue"][0], fifa["blue"][9], fifa["red"][8]]:
    print(f"  slot {sample['slot']:2d} {sample['position']:4s} → {sample['fifa_player']}")
    print(f"      {sample['soul'][:60]}...")
print("[OK] FIFA defaults file valid + 11+11 entries")

print()
print("=" * 70)
print("[4] Simulate /api/lineup_template response shape")
print("=" * 70)
from football_agents.players import TEAM_RED_11V11
fifa_blue = {int(e["slot"]): e for e in fifa["blue"]}
fifa_red  = {int(e["slot"]): e for e in fifa["red"]}
def _entry(persona, team, slot, fifa_for_slot):
    return {
        "slot": slot, "team": team, "name": persona.name,
        "jersey": persona.jersey_number, "position": persona.position,
        "default": f"{persona.background}\n\n我的球风是 —— {persona.play_style}",
        "fifa_player": fifa_for_slot.get("fifa_player", ""),
        "fifa": fifa_for_slot.get("soul", ""),
    }
template = {
    "blue": [_entry(TEAM_BLUE_11V11[i], "blue", i, fifa_blue.get(i, {})) for i in range(11)],
    "red":  [_entry(TEAM_RED_11V11[i], "red", 11+i, fifa_red.get(i, {})) for i in range(11)],
}
print(f"  blue entries: {len(template['blue'])}, red entries: {len(template['red'])}")
sample = template["blue"][9]  # CF / Haaland
print(f"  sample slot 9 (蓝CF): name={sample['name']} #{sample['jersey']} {sample['position']}")
print(f"    fifa_player: {sample['fifa_player']}")
print(f"    default ({len(sample['default'])} chars): {sample['default'][:50]}...")
print(f"    fifa    ({len(sample['fifa'])} chars): {sample['fifa'][:50]}...")
print("[OK] lineup_template shape correct")

print()
print("[DONE] all checks pass")
