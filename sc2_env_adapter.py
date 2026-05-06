"""PySC2 multi-agent adapter for training/evaluation algorithms."""

import numpy as np
from typing import Dict, List, Tuple, Any
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from gym.spaces import Box, Discrete
from simplebattle_adapter import WORKER_TYPES, STRUCTURE_TYPES

RAW_FUNCTIONS = actions.RAW_FUNCTIONS
_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_ENEMY = features.PlayerRelative.ENEMY
NON_COMBAT_TARGET_TYPES = {106, 129, 1912}


class MultiAgentEnv(object):
    def step(self, actions):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        raise NotImplementedError

    def get_obs_size(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        raise NotImplementedError

    def get_total_actions(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_obs_alone_size(self):
        return self.get_obs_size()

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "obs_alone_shape": self.get_obs_alone_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }


class SC2MultiAgentEnv(MultiAgentEnv):
    def __init__(self, map_name: str = "Simple64", step_mul: int = 8, agent_interface_format=None, visualize: bool = False, obs_size: int = 128):
        super().__init__()
        if agent_interface_format is None:
            agent_interface_format = sc2_env.AgentInterfaceFormat(
                feature_dimensions=sc2_env.Dimensions(screen=84, minimap=64),
                use_raw_units=True,
                use_raw_actions=True,
            )

        self.sc2_env = sc2_env.SC2Env(
            map_name=map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.easy),
            ],
            agent_interface_format=agent_interface_format,
            step_mul=step_mul,
            visualize=visualize,
        )

        self.episode_limit = 1000
        self.n_agents = 0
        self.last_timestep = None
        self.last_battle_info = {"self_combat": 0, "enemy_combat": 0, "battle_winner": "ongoing"}
        self._seen_self_combat = False
        self._seen_enemy_combat = False
        self.obs_size = int(obs_size)
        self.state_size = 128
        self.n_actions = 5
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]
        self.observation_space = [Box(low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32) for _ in range(self.n_agents)]

    def _ordered_self_units(self, obs) -> List[Any]:
        units = [u for u in obs.raw_units if u.alliance == _PLAYER_SELF and int(u.unit_type) not in WORKER_TYPES and int(u.unit_type) not in STRUCTURE_TYPES]
        units.sort(key=lambda u: int(u.tag))
        return units

    def _enemy_units(self, obs) -> List[Any]:
        return [u for u in obs.raw_units if u.alliance == _PLAYER_ENEMY]

    def _effective_enemy_units(self, obs) -> List[Any]:
        enemies = self._enemy_units(obs)
        return [
            u for u in enemies
            if int(u.unit_type) not in NON_COMBAT_TARGET_TYPES
            and int(u.unit_type) not in WORKER_TYPES
            and int(u.unit_type) not in STRUCTURE_TYPES
        ]

    def _battle_status(self, obs) -> Dict[str, Any]:
        self_combat = len(self._ordered_self_units(obs))
        enemy_combat = len(self._effective_enemy_units(obs))

        if self_combat > 0:
            self._seen_self_combat = True
        if enemy_combat > 0:
            self._seen_enemy_combat = True

        if not (self._seen_self_combat and self._seen_enemy_combat):
            winner = "ongoing"
        elif self_combat <= 0 and enemy_combat <= 0:
            winner = "draw"
        elif enemy_combat <= 0:
            winner = "self"
        elif self_combat <= 0:
            winner = "enemy"
        else:
            winner = "ongoing"

        return {"self_combat": int(self_combat), "enemy_combat": int(enemy_combat), "battle_winner": winner}

    def _unit_val(self, unit, key: str, default: float = 0.0) -> float:
        try:
            return float(unit[key])
        except Exception:
            try:
                return float(getattr(unit, key))
            except Exception:
                return float(default)

    def _clamp_world(self, pt: List[float]) -> List[float]:
        return [max(0.0, min(255.0, float(pt[0]))), max(0.0, min(255.0, float(pt[1])))]

    def reset(self) -> Dict[str, np.ndarray]:
        timestep = self.sc2_env.reset()
        self.last_timestep = timestep
        self._update_agent_count(timestep)
        self._seen_self_combat = False
        self._seen_enemy_combat = False
        self.last_battle_info = self._battle_status(timestep[0].observation)
        return self._extract_observations(timestep)

    def step(self, actions_dict: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, Any]]:
        sc2_actions = self._convert_actions(actions_dict)
        # PySC2 的 step 期望“每个玩家一组动作”。
        # 我们只有1个受控玩家，因此需要把多单位动作打包成外层长度为1的列表。
        timestep = self.sc2_env.step([sc2_actions])
        self.last_timestep = timestep
        self.last_battle_info = self._battle_status(timestep[0].observation)
        return self._extract_observations(timestep), self._extract_rewards(timestep), self._extract_dones(timestep), self._extract_info(timestep)

    def _update_agent_count(self, timestep):
        self.n_agents = len(self._ordered_self_units(timestep[0].observation))
        self.action_space = [Discrete(self.n_actions) for _ in range(self.n_agents)]
        self.observation_space = [Box(low=-1.0, high=1.0, shape=(self.obs_size,), dtype=np.float32) for _ in range(self.n_agents)]

    def _norm_pos(self, x: float, lo: float, hi: float) -> float:
        return float((x - lo) / max(1e-6, hi - lo))

    def _extract_observations(self, timestep) -> Dict[str, np.ndarray]:
        obs = timestep[0].observation
        all_units = list(obs.raw_units)
        self_units = self._ordered_self_units(obs)
        enemy_units = self._effective_enemy_units(obs)

        if all_units:
            xs = [u.x for u in all_units]
            ys = [u.y for u in all_units]
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        else:
            min_x, max_x, min_y, max_y = 0.0, 64.0, 0.0, 64.0

        obs_dict = {}
        for i, unit in enumerate(self_units):
            feats = [
                self._norm_pos(unit.x, min_x, max_x),
                self._norm_pos(unit.y, min_y, max_y),
                min(1.0, self._unit_val(unit, "health", 0.0) / 200.0),
                min(1.0, self._unit_val(unit, "shield", 0.0) / 200.0),
                min(1.0, self._unit_val(unit, "weapon_cooldown", 0.0) / 15.0),
                float(self._unit_val(unit, "order_length", 0.0) > 0.0),
            ]

            ally_dists = [np.hypot(unit.x - a.x, unit.y - a.y) for a in self_units if int(a.tag) != int(unit.tag)]
            feats.append(min(1.0, min(ally_dists) / 40.0) if ally_dists else 1.0)

            if enemy_units:
                nearest = sorted(enemy_units, key=lambda e: (unit.x - e.x) ** 2 + (unit.y - e.y) ** 2)[:5]
                for e in nearest:
                    feats.extend([
                        max(-1.0, min(1.0, (e.x - unit.x) / 40.0)),
                        max(-1.0, min(1.0, (e.y - unit.y) / 40.0)),
                        min(1.0, self._unit_val(e, "health", 0.0) / 200.0),
                        float(e.unit_type) / 3000.0,
                    ])
            else:
                feats.extend([0.0] * 20)

            arr = np.asarray(feats, dtype=np.float32)
            obs_dict[str(i)] = np.pad(arr, (0, max(0, self.obs_size - arr.size)))[: self.obs_size]

        return obs_dict

    def _extract_rewards(self, timestep) -> Dict[str, float]:
        winner = self.last_battle_info.get("battle_winner", "ongoing")
        reward = 1.0 if winner == "self" else (-1.0 if winner == "enemy" else (0.0 if winner == "draw" else float(timestep[0].reward)))
        return {str(i): reward for i in range(self.n_agents)}

    def _extract_dones(self, timestep) -> Dict[str, bool]:
        battle_done = self.last_battle_info.get("battle_winner", "ongoing") != "ongoing"
        done = bool(timestep[0].last()) or battle_done
        return {str(i): done for i in range(self.n_agents)}

    def _extract_info(self, timestep) -> Dict[str, Any]:
        info = {"timestep": timestep[0]}
        info.update(self.last_battle_info)
        return info

    def _convert_actions(self, actions_dict: Dict[str, np.ndarray]) -> List[actions.FunctionCall]:
        if not self.last_timestep:
            return [RAW_FUNCTIONS.no_op()]

        obs = self.last_timestep[0].observation
        self_units = self._ordered_self_units(obs)
        enemies = self._effective_enemy_units(obs)
        if not self_units:
            return [RAW_FUNCTIONS.no_op()]

        ally_center = [float(sum(u.x for u in self_units) / len(self_units)), float(sum(u.y for u in self_units) / len(self_units))]
        enemy_center = [float(sum(e.x for e in enemies) / len(enemies)), float(sum(e.y for e in enemies) / len(enemies))] if enemies else [63.0 - ally_center[0], 63.0 - ally_center[1]]
        ally_center, enemy_center = self._clamp_world(ally_center), self._clamp_world(enemy_center)

        sc2_actions = []
        for agent_id, action in actions_dict.items():
            if not str(agent_id).isdigit():
                continue
            idx = int(agent_id)
            if idx < 0 or idx >= len(self_units):
                continue

            unit = self_units[idx]
            if isinstance(action, np.ndarray):
                if action.size == 0:
                    act = 0
                elif action.ndim == 0 or action.size == 1:
                    act = int(action.reshape(-1)[0])
                else:
                    # 兼容离散动作的多种输出格式：
                    # - 标量动作id
                    # - shape=(n_actions,) 的one-hot/概率/logits
                    act = int(np.argmax(action.reshape(-1)))
            else:
                act = int(action)
            tag = [int(unit.tag)]

            # 为避免策略输出大量0导致“多数单位原地不动”，将0也映射为有效动作。
            if act == 0:
                if enemies:
                    target = min(enemies, key=lambda e: (e.x - unit.x) ** 2 + (e.y - unit.y) ** 2)
                    sc2_actions.append(RAW_FUNCTIONS.Attack_unit("now", tag, int(target.tag)))
                else:
                    sc2_actions.append(RAW_FUNCTIONS.Move_pt("now", tag, enemy_center))
            elif act == 1:
                sc2_actions.append(RAW_FUNCTIONS.Move_pt("now", tag, enemy_center))
            elif act == 2:
                if enemies:
                    target = min(enemies, key=lambda e: (e.x - unit.x) ** 2 + (e.y - unit.y) ** 2)
                    sc2_actions.append(RAW_FUNCTIONS.Attack_unit("now", tag, int(target.tag)))
                else:
                    sc2_actions.append(RAW_FUNCTIONS.Move_pt("now", tag, enemy_center))
            elif act == 3:
                sc2_actions.append(RAW_FUNCTIONS.Move_pt("now", tag, ally_center))
            else:
                jitter = [float(unit.x + np.random.uniform(-4.0, 4.0)), float(unit.y + np.random.uniform(-4.0, 4.0))]
                sc2_actions.append(RAW_FUNCTIONS.Move_pt("now", tag, self._clamp_world(jitter)))

        return sc2_actions if sc2_actions else [RAW_FUNCTIONS.no_op()]

    def get_obs(self) -> List[np.ndarray]:
        return list(self._extract_observations(self.last_timestep).values())

    def get_obs_agent(self, agent_id: int) -> np.ndarray:
        obs_list = self.get_obs()
        return obs_list[agent_id] if 0 <= agent_id < len(obs_list) else np.zeros(self.obs_size, dtype=np.float32)

    def get_obs_size(self) -> int:
        return self.obs_size

    def get_obs_alone_size(self):
        return self.obs_size

    def get_state(self) -> np.ndarray:
        obs_list = self.get_obs()
        return np.mean(np.stack(obs_list, axis=0), axis=0).astype(np.float32) if obs_list else np.zeros(self.state_size, dtype=np.float32)

    def get_state_size(self) -> int:
        return self.state_size

    def get_avail_actions(self) -> List[List[int]]:
        return [[1] * self.n_actions for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        avail = self.get_avail_actions()
        return avail[agent_id] if 0 <= agent_id < len(avail) else [0] * self.n_actions

    def get_total_actions(self) -> int:
        return self.n_actions

    def render(self):
        pass

    def close(self):
        self.sc2_env.close()

    def seed(self, seed: int):
        self.sc2_env._seed = seed

    def save_replay(self, replay_path: str):
        self.sc2_env.save_replay(replay_path)
