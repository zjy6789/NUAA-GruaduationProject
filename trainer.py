"""Training Pipeline for HARL-based Multi-Agent Reinforcement Learning."""

import os
import numpy as np
from typing import Dict, List, Tuple, Any
from harl_wrapper import HARLAgentWrapper
from sc2_env_adapter import SC2MultiAgentEnv
import logging


class Trainer:
    def __init__(self, agent: HARLAgentWrapper, env: SC2MultiAgentEnv, config: Dict[str, Any]):
        self.agent = agent
        self.env = env
        self.config = config

        self.num_episodes = config.get("num_episodes", 1000)
        self.max_steps_per_episode = config.get("max_steps_per_episode", 1000)
        self.update_interval = config.get("update_interval", 1000)
        self.save_interval = config.get("save_interval", 100)
        self.save_dir = config.get("save_dir", ".")
        self.control_file = config.get("control_file")
        self.num_collect_envs = max(1, int(config.get("num_collect_envs", 1)))

        self.logger = logging.getLogger(__name__)
        self.episode_rewards = []
        self.win_rates = []
        self.should_stop = False
        os.makedirs(self.save_dir, exist_ok=True)

        self.collect_envs = [env]
        for _ in range(self.num_collect_envs - 1):
            self.collect_envs.append(
                SC2MultiAgentEnv(
                    map_name=config.get("map_name", "Simple64"),
                    step_mul=config.get("step_mul", 8),
                    visualize=False,
                )
            )

    def close(self):
        for e in self.collect_envs[1:]:
            try:
                e.close()
            except Exception:
                pass

    def _checkpoint_path(self, episode: int) -> str:
        return os.path.join(self.save_dir, f"model_episode_{episode}.pth")

    def _read_control_command(self) -> str:
        if not self.control_file or not os.path.exists(self.control_file):
            return ""
        try:
            with open(self.control_file, "r", encoding="utf-8") as f:
                cmd = f.read().strip().lower()
            os.remove(self.control_file)
            return cmd
        except Exception:
            return ""

    def _maybe_handle_control_command(self, episode: int):
        cmd = self._read_control_command()
        if cmd == "save_and_stop":
            save_path = self._checkpoint_path(episode)
            self.agent.save_model(save_path)
            self.logger.info(f"Manual save done: {save_path}")
            self.should_stop = True
        elif cmd == "save":
            save_path = self._checkpoint_path(episode)
            self.agent.save_model(save_path)
            self.logger.info(f"Manual save done: {save_path}")

    def collect_rollouts(self, num_steps: int, episode_idx: int) -> List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Dict[str, bool]]]:
        rollouts = []

        obs_list = [env.reset() for env in self.collect_envs]
        ep_rewards = [0.0 for _ in self.collect_envs]
        ep_lengths = [0 for _ in self.collect_envs]

        steps_per_env = max(1, num_steps // len(self.collect_envs))
        for _ in range(steps_per_env):
            self._maybe_handle_control_command(episode_idx)
            if self.should_stop:
                break

            for i, env in enumerate(self.collect_envs):
                action_dict = self.agent.select_actions(obs_list[i], deterministic=False)
                next_obs, reward_dict, done_dict, _ = env.step(action_dict)
                rollouts.append((obs_list[i], action_dict, reward_dict, done_dict))

                ep_rewards[i] += sum(reward_dict.values())
                ep_lengths[i] += 1
                obs_list[i] = next_obs

                if any(done_dict.values()):
                    self.episode_rewards.append(ep_rewards[i])
                    self.win_rates.append(1.0 if ep_rewards[i] > 0 else 0.0)
                    self.logger.info(f"Episode ended(env={i}). Reward: {ep_rewards[i]:.2f}, Length: {ep_lengths[i]}")
                    obs_list[i] = env.reset()
                    ep_rewards[i] = 0.0
                    ep_lengths[i] = 0

        return rollouts

    def update_policy(self, rollouts: List[Tuple]) -> Dict[str, float]:
        return self.agent.update_policy(rollouts)

    def train(self):
        self.logger.info(f"Starting training... (parallel envs={len(self.collect_envs)})")
        total_steps = 0

        for episode in range(self.num_episodes):
            self._maybe_handle_control_command(episode)
            if self.should_stop:
                break

            rollouts = self.collect_rollouts(self.max_steps_per_episode, episode)
            total_steps += len(rollouts)

            if self.should_stop:
                break

            if total_steps >= self.update_interval and len(rollouts) > 0:
                train_info = self.update_policy(rollouts)
                self.logger.info(f"Episode {episode}, Train Info: {train_info}")
                total_steps = 0

            if episode % self.save_interval == 0:
                save_path = self._checkpoint_path(episode)
                self.agent.save_model(save_path)
                self.logger.info(f"Model saved at episode {episode}: {save_path}")

            if len(self.episode_rewards) > 0:
                avg_reward = float(np.mean(self.episode_rewards[-10:]))
                avg_win_rate = float(np.mean(self.win_rates[-10:]))
                self.logger.info(f"TRAIN_METRIC episode={episode} avg_reward={avg_reward:.4f} avg_win_rate={avg_win_rate:.4f}")

        if self.should_stop:
            self.logger.info("Training stopped by user command.")
        self.logger.info("Training completed.")

    def evaluate(self, num_episodes: int = 1) -> Dict[str, float]:
        self.logger.info("Starting evaluation...")
        eval_rewards, eval_win_rates, eval_lengths = [], [], []

        for _ in range(num_episodes):
            obs_dict = self.env.reset()
            episode_reward = 0.0
            done, steps = False, 0

            while not done and steps < self.max_steps_per_episode:
                action_dict = self.agent.select_actions(obs_dict, deterministic=True)

                if self.config.get("debug_force_spread", False) and steps < 30:
                    # 调试模式：前30步强制让每个单位执行不同移动模式，快速验证“是否每单位可独立控制”。
                    forced = {}
                    for k in sorted(obs_dict.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                        idx = int(k) if k.isdigit() else 0
                        # 在现有动作语义中：1=前进, 3=回撤, 4=抖动机动
                        forced[k] = [1, 3, 4][idx % 3]
                    action_dict = forced

                if steps < 5:
                    self.logger.info(
                        "EVAL_DEBUG step=%d obs_agents=%d action_agents=%d keys=%s",
                        steps,
                        len(obs_dict),
                        len(action_dict),
                        sorted(list(action_dict.keys()))[:10],
                    )
                next_obs_dict, reward_dict, done_dict, _ = self.env.step(action_dict)
                episode_reward += sum(reward_dict.values())
                done = any(done_dict.values())
                obs_dict = next_obs_dict
                steps += 1

            eval_rewards.append(episode_reward)
            eval_win_rates.append(1.0 if episode_reward > 0 else 0.0)
            eval_lengths.append(steps)

        eval_stats = {
            "avg_reward": float(np.mean(eval_rewards)) if eval_rewards else 0.0,
            "std_reward": float(np.std(eval_rewards)) if eval_rewards else 0.0,
            "avg_win_rate": float(np.mean(eval_win_rates)) if eval_win_rates else 0.0,
            "avg_length": float(np.mean(eval_lengths)) if eval_lengths else 0.0,
            "episode_rewards": eval_rewards,
            "episode_lengths": eval_lengths,
        }

        self.logger.info(f"EVAL_METRIC avg_reward={eval_stats['avg_reward']:.4f} avg_win_rate={eval_stats['avg_win_rate']:.4f} avg_length={eval_stats['avg_length']:.2f}")
        self.logger.info(f"Evaluation Results: {eval_stats}")
        return eval_stats
