"""Baseline and self-play evaluation utilities for simple battle logic."""

import time
import numpy as np
from typing import Dict, Any, List, Tuple
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

RAW_FUNCTIONS = actions.RAW_FUNCTIONS
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL

WORKER_TYPES = {
    units.Terran.SCV,
    units.Protoss.Probe,
    units.Zerg.Drone,
    units.Terran.MULE,
}

STRUCTURE_TYPES = {
    units.Terran.CommandCenter,
    units.Terran.OrbitalCommand,
    units.Terran.PlanetaryFortress,
    units.Terran.SupplyDepot,
    units.Terran.Barracks,
    units.Terran.EngineeringBay,
    units.Terran.Factory,
    units.Terran.Starport,
    units.Terran.Bunker,
    units.Terran.MissileTurret,
    units.Terran.SensorTower,
    units.Terran.GhostAcademy,
    units.Terran.Armory,
    units.Terran.FusionCore,
    units.Terran.Refinery,
    units.Terran.BarracksReactor,
    units.Terran.BarracksTechLab,
    units.Terran.FactoryReactor,
    units.Terran.FactoryTechLab,
    units.Terran.StarportReactor,
    units.Terran.StarportTechLab,
    units.Protoss.Nexus,
    units.Protoss.Pylon,
    units.Protoss.Assimilator,
    units.Protoss.Gateway,
    units.Protoss.WarpGate,
    units.Protoss.Forge,
    units.Protoss.CyberneticsCore,
    units.Protoss.PhotonCannon,
    units.Protoss.ShieldBattery,
    units.Protoss.RoboticsFacility,
    units.Protoss.Stargate,
    units.Protoss.TwilightCouncil,
    units.Protoss.RoboticsBay,
    units.Protoss.FleetBeacon,
    units.Protoss.TemplarArchive,
    units.Protoss.DarkShrine,
    units.Zerg.Hatchery,
    units.Zerg.Lair,
    units.Zerg.Hive,
    units.Zerg.Extractor,
    units.Zerg.SpawningPool,
    units.Zerg.EvolutionChamber,
    units.Zerg.RoachWarren,
    units.Zerg.BanelingNest,
    units.Zerg.HydraliskDen,
    units.Zerg.InfestationPit,
    units.Zerg.Spire,
    units.Zerg.GreaterSpire,
    units.Zerg.UltraliskCavern,
    units.Zerg.NydusNetwork,
    units.Zerg.NydusCanal,
    units.Zerg.LurkerDen,
    units.Zerg.SpineCrawler,
    units.Zerg.SporeCrawler,
    units.Zerg.CreepTumor,
    units.Zerg.CreepTumorQueen,
    units.Zerg.CreepTumorBurrowed,
}

MINERAL_FIELD_TYPES = {
    units.Neutral.MineralField,
    units.Neutral.MineralField750,
    units.Neutral.RichMineralField,
    units.Neutral.RichMineralField750,
    units.Neutral.LabMineralField,
    units.Neutral.LabMineralField750,
    units.Neutral.PurifierMineralField,
    units.Neutral.PurifierMineralField750,
    units.Neutral.PurifierRichMineralField,
    units.Neutral.PurifierRichMineralField750,
    units.Neutral.BattleStationMineralField,
    units.Neutral.BattleStationMineralField750,
}

NON_COMBAT_TARGET_TYPES = {
    106,   # Zerg.Overlord
    129,   # Zerg.Overseer
    1912,  # Zerg.OverseerOversightMode
}


class SimpleBattleLogicWrapper:
    """A lightweight heuristic policy for existing single-side adapter env."""

    def __init__(self, config: Dict[str, Any], obs_space: Any = None, act_space: Any = None, device=None):
        self.config = config

    def select_actions(self, observation_dict: Dict[str, np.ndarray], deterministic: bool = False) -> Dict[str, np.ndarray]:
        action_dict = {}
        for agent_id, obs in observation_dict.items():
            obs = np.asarray(obs, dtype=np.float32)
            health = float(obs[2]) if obs.shape[0] > 2 else 1.0
            enemy_signal = int(np.count_nonzero((obs[8:] if obs.shape[0] > 8 else obs) > 0.01))

            if health < 0.25:
                action = 0
            elif enemy_signal > 6:
                action = 1
            else:
                action = 1 if (deterministic or np.random.random() < 0.55) else 0

            action_dict[agent_id] = np.array([action], dtype=np.int64)
        return action_dict

    def update_policy(self, rollouts: List[Tuple]) -> Dict[str, float]:
        return {"policy_loss": 0.0}

    def save_model(self, path: str):
        payload = {"type": "simplebattle_logic", "config": self.config}
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(payload))

    def load_model(self, path: str):
        return


class RawUnitSimpleBattleAgent(base_agent.BaseAgent):
    """Raw-unit controller for worker mining and combat-unit search/attack."""

    def __init__(self, logger=None, name: str = "agent"):
        super().__init__()
        self.iteration = 0
        self.last_worker_cycle = -9999
        self.last_combat_cycle = -9999
        self.logger = logger
        self.name = name

    def _debug(self, msg: str):
        if self.logger is not None and self.iteration % 8 == 0:
            self.logger.info(f"RAW_DEBUG[{self.name}] {msg}")

    def _self_units(self, obs):
        return [u for u in obs.observation.raw_units if u.alliance == _PLAYER_SELF]

    def _enemy_units(self, obs):
        return [u for u in obs.observation.raw_units if u.alliance == _PLAYER_ENEMY]

    def _effective_enemy_units(self, obs):
        enemies = self._enemy_units(obs)
        return [
            u for u in enemies
            if int(u.unit_type) not in NON_COMBAT_TARGET_TYPES
            and int(u.unit_type) not in WORKER_TYPES
            and int(u.unit_type) not in STRUCTURE_TYPES
        ]

    def _neutral_units(self, obs):
        return [u for u in obs.observation.raw_units if u.alliance == _PLAYER_NEUTRAL]

    def _is_worker(self, unit):
        return unit.unit_type in WORKER_TYPES

    def _is_structure(self, unit):
        return unit.unit_type in STRUCTURE_TYPES

    def _combat_units(self, obs):
        return [u for u in self._self_units(obs) if not self._is_worker(u) and not self._is_structure(u)]

    def _worker_units(self, obs):
        return [u for u in self._self_units(obs) if self._is_worker(u)]

    def _mineral_fields(self, obs):
        return [u for u in self._neutral_units(obs) if u.unit_type in MINERAL_FIELD_TYPES]

    def _distance(self, a, b):
        return float(np.hypot(a.x - b.x, a.y - b.y))

    def _closest_enemy_to_group(self, group, enemies):
        if not group or not enemies:
            return None
        center_x = sum(u.x for u in group) / len(group)
        center_y = sum(u.y for u in group) / len(group)
        return min(enemies, key=lambda e: (e.x - center_x) ** 2 + (e.y - center_y) ** 2)

    def _enemy_search_point(self, obs):
        allies = self._combat_units(obs) or self._self_units(obs)
        enemies = self._effective_enemy_units(obs) or self._enemy_units(obs)
        if enemies:
            target = self._closest_enemy_to_group(allies, enemies)
            return [target.x, target.y]
        if not allies:
            return [32, 32]

        xs = [u.x for u in obs.observation.raw_units]
        ys = [u.y for u in obs.observation.raw_units]
        if not xs or not ys:
            return [32, 32]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center_x = sum(u.x for u in allies) / len(allies)
        center_y = sum(u.y for u in allies) / len(allies)

        mirror_x = min_x + max_x - center_x
        mirror_y = min_y + max_y - center_y
        mirror_x = max(min_x, min(max_x, mirror_x))
        mirror_y = max(min_y, min(max_y, mirror_y))
        return [mirror_x, mirror_y]

    def _issue_worker_harvest(self, obs):
        workers = [u for u in self._worker_units(obs) if u.order_length == 0]
        minerals = self._mineral_fields(obs)
        if not workers or not minerals:
            self._debug(f"worker idle={len(workers)} minerals={len(minerals)} -> no worker action")
            return None

        workers = workers[:8]
        actions_to_try = []
        target = None
        for worker in workers:
            target = min(minerals, key=lambda m: self._distance(worker, m))
            actions_to_try.append(RAW_FUNCTIONS.Harvest_Gather_unit("now", [worker.tag], target.tag))

        if actions_to_try and target is not None:
            self._debug(f"worker action=Harvest_Gather_unit workers={len(workers)} target_tag={int(target.tag)}")
        return actions_to_try[0] if actions_to_try else None

    def _issue_combat_command(self, obs):
        combat_units = self._combat_units(obs)
        if not combat_units:
            self._debug("combat_units=0 -> no combat action")
            return None

        all_enemies = self._enemy_units(obs)
        enemies = self._effective_enemy_units(obs)

        if enemies:
            target = self._closest_enemy_to_group(combat_units, enemies)
            nearby_units = [u for u in combat_units if self._distance(u, target) <= 10.0]
            self._debug(
                f"combat_units={len(combat_units)} enemies={len(all_enemies)} effective={len(enemies)} "
                f"target_tag={int(target.tag)} target_type={int(target.unit_type)} nearby={len(nearby_units)}"
            )
            if nearby_units:
                self._debug("combat action=Attack_unit")
                return RAW_FUNCTIONS.Attack_unit("now", [u.tag for u in nearby_units], target.tag)
            self._debug("combat action=Move_pt toward enemy")
            return RAW_FUNCTIONS.Move_pt("now", [u.tag for u in combat_units], [target.x, target.y])

        search_point = self._enemy_search_point(obs)
        self._debug(
            f"combat_units={len(combat_units)} enemies={len(all_enemies)} effective=0 action=Move_pt search=({search_point[0]:.1f},{search_point[1]:.1f})"
        )
        return RAW_FUNCTIONS.Move_pt("now", [u.tag for u in combat_units], search_point)

    def step(self, obs):
        super().step(obs)
        self.iteration += 1

        if self.iteration - self.last_worker_cycle >= 16:
            worker_action = self._issue_worker_harvest(obs)
            self.last_worker_cycle = self.iteration
            if worker_action is not None:
                return worker_action

        if self.iteration - self.last_combat_cycle >= 1:
            combat_action = self._issue_combat_command(obs)
            self.last_combat_cycle = self.iteration
            if combat_action is not None:
                return combat_action

        self._debug("action=no_op")
        return RAW_FUNCTIONS.no_op()


def evaluate_simplebattle_self_play(
    map_name: str,
    step_mul: int,
    max_steps: int,
    visualize: bool,
    hold_seconds: int,
    logger,
    race: str = "terran",
) -> Dict[str, Any]:
    """Run one visible self-play battle where both sides use same simplebattle logic."""

    race_map = {
        "terran": sc2_env.Race.terran,
        "protoss": sc2_env.Race.protoss,
        "zerg": sc2_env.Race.zerg,
        "random": sc2_env.Race.random,
    }
    selected_race = race_map.get(str(race).lower(), sc2_env.Race.terran)

    with sc2_env.SC2Env(
        map_name=map_name,
        players=[
            sc2_env.Agent(selected_race),
            sc2_env.Agent(selected_race),
        ],
        agent_interface_format=sc2_env.AgentInterfaceFormat(
            feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
            use_raw_units=True,
            use_raw_actions=True,
        ),
        step_mul=step_mul,
        game_steps_per_episode=0,
        visualize=visualize,
    ) as env:
        agents = [RawUnitSimpleBattleAgent(logger=logger, name="p1"), RawUnitSimpleBattleAgent(logger=logger, name="p2")]
        timesteps = env.reset()

        p1_reward, p2_reward = 0.0, 0.0
        steps = 0

        def _count_combat(raw_units, alliance):
            return len([
                u for u in raw_units
                if u.alliance == alliance
                and int(u.unit_type) not in WORKER_TYPES
                and int(u.unit_type) not in STRUCTURE_TYPES
                and int(u.unit_type) not in NON_COMBAT_TARGET_TYPES
            ])

        battle_winner = "ongoing"
        while steps < max_steps and battle_winner == "ongoing":
            steps += 1
            acts = [agents[0].step(timesteps[0]), agents[1].step(timesteps[1])]
            timesteps = env.step(acts)
            p1_reward += float(timesteps[0].reward)
            p2_reward += float(timesteps[1].reward)

            p1_units = _count_combat(timesteps[0].observation.raw_units, _PLAYER_SELF)
            p2_units = _count_combat(timesteps[1].observation.raw_units, _PLAYER_SELF)
            if p1_units <= 0 and p2_units <= 0:
                battle_winner = "draw"
            elif p2_units <= 0:
                battle_winner = "player1"
            elif p1_units <= 0:
                battle_winner = "player2"
            elif timesteps[0].last() or timesteps[1].last():
                battle_winner = "draw"

        p1_score = float(timesteps[0].observation["score_cumulative"][0]) if "score_cumulative" in timesteps[0].observation else 0.0
        p2_score = float(timesteps[1].observation["score_cumulative"][0]) if "score_cumulative" in timesteps[1].observation else 0.0

        winner = battle_winner if battle_winner != "ongoing" else "draw"

        logger.info(
            f"EVAL_BATTLE winner={winner} p1_reward={p1_reward:.4f} p2_reward={p2_reward:.4f} "
            f"p1_score={p1_score:.1f} p2_score={p2_score:.1f} steps={steps}"
        )

        if visualize and hold_seconds > 0:
            logger.info(f"Holding game window for {hold_seconds}s after battle end...")
            time.sleep(hold_seconds)

        return {
            "winner": winner,
            "p1_reward": p1_reward,
            "p2_reward": p2_reward,
            "p1_score": p1_score,
            "p2_score": p2_score,
            "steps": steps,
            "avg_reward": (p1_reward + p2_reward) / 2.0,
            "avg_win_rate": 1.0 if winner != "draw" else 0.5,
            "avg_length": float(steps),
        }
