"""
HARL Agent Wrapper for Multi-Agent Reinforcement Learning.

This module provides a wrapper around HARL algorithms to interface with custom multi-agent environments.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.mappo import MAPPO
from simplebattle_adapter import SimpleBattleLogicWrapper


class HARLAgentWrapper:
    """Wrapper for HARL algorithms and simple baseline logic."""

    def __init__(self, config: Dict[str, Any], obs_space: Any, act_space: Any, device: torch.device = torch.device("cpu")):
        self.config = config
        self.device = device
        self.obs_space = obs_space
        self.act_space = act_space

        algo_name = config.get("algo", "happo")
        if algo_name == "happo":
            self.algo_class = HAPPO
        elif algo_name == "mappo":
            self.algo_class = MAPPO
        elif algo_name == "simplebattle":
            self.algo_class = None
            self.baseline_agent = SimpleBattleLogicWrapper(config, obs_space, act_space, device)
            self.num_agents = config.get("num_agents", 1)
            self.actors = []
            self.rnn_states_actor = []
            self.masks = np.ones((1, 1), dtype=np.float32)
            return
        else:
            raise ValueError(f"Algorithm {algo_name} not found in HARL registry.")

        self.num_agents = config.get("num_agents", 1)
        self.actors = []

        if isinstance(obs_space, list):
            obs_spaces = obs_space
            act_spaces = act_space if isinstance(act_space, list) else [act_space] * self.num_agents
        else:
            obs_spaces = [obs_space] * self.num_agents
            act_spaces = [act_space] * self.num_agents

        for i in range(self.num_agents):
            actor = self.algo_class(config, obs_spaces[i], act_spaces[i], device)
            self.actors.append(actor)

        self.rnn_states_actor = [
            np.zeros((1, config.get("recurrent_n", 1), config.get("recurrent_hidden_size", 64)), dtype=np.float32)
            for _ in range(self.num_agents)
        ]
        self.masks = np.ones((1, 1), dtype=np.float32)

    def select_actions(self, observation_dict: Dict[str, np.ndarray], deterministic: bool = False) -> Dict[str, np.ndarray]:
        if self.config.get("algo") == "simplebattle":
            return self.baseline_agent.select_actions(observation_dict, deterministic=deterministic)

        action_dict = {}
        for agent_id, obs in observation_dict.items():
            idx = int(agent_id) if isinstance(agent_id, str) and agent_id.isdigit() else 0
            if idx >= self.num_agents:
                continue

            obs = np.array(obs).reshape(1, -1)
            rnn_state = self.rnn_states_actor[idx]

            with torch.no_grad():
                actions, action_log_probs, rnn_states_actor = self.actors[idx].get_actions(
                    obs, rnn_state, self.masks, deterministic=deterministic
                )

            action_np = actions.detach().cpu().numpy().reshape(-1)
            rnn_state_np = rnn_states_actor.detach().cpu().numpy()

            # 统一输出为标量离散动作id，避免 ndarray 形状差异导致后续解析异常。
            if action_np.size == 0:
                action_out = 0
            elif action_np.size == 1:
                action_out = int(action_np[0])
            else:
                action_out = int(np.argmax(action_np))

            action_dict[agent_id] = action_out
            self.rnn_states_actor[idx] = rnn_state_np

        return action_dict

    def update_policy(self, rollouts: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Dict[str, bool]]]) -> Dict[str, float]:
        if self.config.get("algo") == "simplebattle":
            return self.baseline_agent.update_policy(rollouts)

        train_infos = []
        for idx in range(self.num_agents):
            obs_list = []
            action_list = []
            reward_list = []
            done_list = []

            for obs_dict, action_dict, reward_dict, done_dict in rollouts:
                if str(idx) in obs_dict:
                    obs_list.append(obs_dict[str(idx)])
                    action_list.append(action_dict[str(idx)])
                    reward_list.append(reward_dict[str(idx)])
                    done_list.append(done_dict[str(idx)])

            if obs_list:
                returns = []
                G = 0
                for reward, done in zip(reversed(reward_list), reversed(done_list)):
                    if done:
                        G = 0
                    G = reward + 0.99 * G
                    returns.insert(0, G)

                obs_np = np.asarray(obs_list, dtype=np.float32)
                action_np = np.asarray(action_list)
                return_np = np.asarray(returns, dtype=np.float32)

                obs_batch = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
                action_batch = torch.tensor(action_np, dtype=torch.float32, device=self.device)
                return_batch = torch.tensor(return_np, dtype=torch.float32, device=self.device)

                train_info = self._update_actor(idx, obs_batch, action_batch, return_batch)
                train_infos.append(train_info)

        if train_infos:
            avg_train_info = {}
            for key in train_infos[0].keys():
                avg_train_info[key] = np.mean([info[key] for info in train_infos])
            return avg_train_info
        return {}

    def _update_actor(self, agent_idx: int, obs_batch: torch.Tensor, action_batch: torch.Tensor, return_batch: torch.Tensor) -> Dict[str, float]:
        self.actors[agent_idx].actor_optimizer.zero_grad()
        action_log_probs, _, _ = self.actors[agent_idx].evaluate_actions(
            obs_batch, self.rnn_states_actor[agent_idx], action_batch, self.masks
        )
        loss = -torch.mean(action_log_probs * return_batch)
        loss.backward()
        self.actors[agent_idx].actor_optimizer.step()
        return {"policy_loss": loss.item()}

    def save_model(self, path: str):
        if self.config.get("algo") == "simplebattle":
            self.baseline_agent.save_model(path)
            return

        state_dict = {
            "actors": [actor.actor.state_dict() for actor in self.actors],
            "config": self.config,
        }
        torch.save(state_dict, path)

    def load_model(self, path: str):
        if self.config.get("algo") == "simplebattle":
            self.baseline_agent.load_model(path)
            return

        if os.path.isdir(path):
            self._load_model_dir(path)
            return

        state_dict = torch.load(path, map_location=self.device)
        actor_states = state_dict.get("actors", [])
        if len(actor_states) < self.num_agents:
            raise ValueError(
                f"Model file contains {len(actor_states)} actors, but current env needs {self.num_agents}."
            )
        for i in range(self.num_agents):
            self._safe_load_actor_state(self.actors[i].actor, actor_states[i], source=f"file:{path} actor[{i}]")

    def _load_model_dir(self, model_dir: str):
        """
        Load per-agent actor weights from a directory.

        Expected filenames (any existing subset will be used, with priority):
        - actor_agent0.pt, actor_agent1.pt, ... (HARL常见导出)
        - actor_0.pth, actor_1.pth, ...
        - actor0.pth, actor1.pth, ...
        - 0.pth, 1.pth, ...

        Each file can be either:
        - raw actor.state_dict()
        - wrapper dict containing key: actor / actor_state_dict / model / state_dict
        """
        for i in range(self.num_agents):
            candidates = [
                os.path.join(model_dir, f"actor_agent{i}.pt"),
                os.path.join(model_dir, f"actor_{i}.pth"),
                os.path.join(model_dir, f"actor{i}.pth"),
                os.path.join(model_dir, f"{i}.pth"),
            ]
            actor_file = next((p for p in candidates if os.path.exists(p)), None)
            if actor_file is None:
                raise FileNotFoundError(
                    f"Cannot find actor weight for agent {i} in {model_dir}. "
                    f"Tried: {', '.join(os.path.basename(p) for p in candidates)}"
                )

            data = torch.load(actor_file, map_location=self.device)
            if isinstance(data, dict):
                if "actor" in data and isinstance(data["actor"], dict):
                    actor_state = data["actor"]
                elif "actor_state_dict" in data and isinstance(data["actor_state_dict"], dict):
                    actor_state = data["actor_state_dict"]
                elif "model" in data and isinstance(data["model"], dict):
                    actor_state = data["model"]
                elif "state_dict" in data and isinstance(data["state_dict"], dict):
                    actor_state = data["state_dict"]
                else:
                    actor_state = data
            else:
                actor_state = data

            self._safe_load_actor_state(self.actors[i].actor, actor_state, source=f"dir:{model_dir} actor[{i}]")

    def _safe_load_actor_state(self, actor_module: torch.nn.Module, actor_state: dict, source: str = ""):
        """Load actor weights with compatibility filtering.

        - Removes keys that do not exist in current model (e.g. feature_norm when disabled).
        - Removes keys with mismatched shape.
        - Uses strict=False to tolerate architectural minor differences.
        """
        if not isinstance(actor_state, dict):
            raise TypeError(f"Actor checkpoint from {source} is not a state_dict dict.")

        model_state = actor_module.state_dict()
        filtered = {}
        dropped_keys = []

        for k, v in actor_state.items():
            if k not in model_state:
                dropped_keys.append((k, "missing_in_model"))
                continue
            if hasattr(v, "shape") and hasattr(model_state[k], "shape") and tuple(v.shape) != tuple(model_state[k].shape):
                dropped_keys.append((k, f"shape_mismatch {tuple(v.shape)} != {tuple(model_state[k].shape)}"))
                continue
            filtered[k] = v

        missing, unexpected = actor_module.load_state_dict(filtered, strict=False)

        if dropped_keys:
            preview = ", ".join([f"{k}({reason})" for k, reason in dropped_keys[:5]])
            print(f"[HARL LOAD] {source}: dropped {len(dropped_keys)} keys, e.g. {preview}")
        if unexpected:
            print(f"[HARL LOAD] {source}: unexpected keys after filtering: {len(unexpected)}")
        if missing:
            # missing keys are acceptable in compatibility mode, but still surface count for debugging.
            print(f"[HARL LOAD] {source}: missing keys after filtering: {len(missing)}")
