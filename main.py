"""Main Simulation Entry Point for Drone Swarm Collaborative Confrontation."""

import argparse
import logging
import yaml
import torch
import sys
import os
from datetime import datetime

harl_path = os.path.join(os.path.dirname(__file__), "HARL")
sys.path.insert(0, harl_path)

from harl_wrapper import HARLAgentWrapper
from sc2_env_adapter import SC2MultiAgentEnv
from trainer import Trainer
from simplebattle_adapter import evaluate_simplebattle_self_play
from gym.spaces import Box, Discrete


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_config() -> dict:
    return {
        "algo": "happo",
        "num_agents": 5,
        "lr": 5e-4,
        "clip_param": 0.2,
        "ppo_epoch": 10,
        "actor_num_mini_batch": 1,
        "entropy_coef": 0.01,
        "use_valuenorm": True,
        "recurrent_hidden_size": 64,
        "recurrent_layers": 1,
        "data_chunk_length": 10,
        "use_policy_active_masks": True,
        "action_aggregation": "prod",
        "num_episodes": 1000,
        "max_steps_per_episode": 1000,
        "update_interval": 1000,
        "save_interval": 100,
        "obs_dim": 10,
        "act_dim": 5,
        "hidden_sizes": [128, 128],
        "activation_func": "relu",
        "use_recurrent_policy": False,
        "use_naive_recurrent_policy": False,
        "recurrent_n": 1,
        "use_feature_normalization": False,
        "initialization_method": "orthogonal",
        "weight_decay": 0,
        "opti_eps": 1e-5,
        "gain": 0.01,
        "std_x_coef": 1.0,
        "std_y_coef": 0.5,
        "use_max_grad_norm": True,
        "max_grad_norm": 10.0,
        "use_clipped_value_loss": True,
        "clip_value": True,
        "value_loss_coef": 1,
        "use_huber_loss": True,
        "huber_delta": 10.0,
        "use_popart": False,
        "popart_beta": 0.99999,
        "state_type": "EP",
        "share_param": False,
    }


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Train HARL agent for drone swarm simulation.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--map", type=str, default="Simple64")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None, choices=["happo", "mappo", "simplebattle"])
    parser.add_argument("--num-collect-envs", type=int, default=1)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--clip-param", type=float, default=None)
    parser.add_argument("--update-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--step-mul", type=int, default=8)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--hold-seconds", type=int, default=8)
    parser.add_argument("--simplebattle-race", type=str, default="terran", choices=["terran", "protoss", "zerg", "random"])
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--control-file", type=str, default=None)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--debug-force-spread", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    config = load_config(args.config) if args.config else create_config()

    config["num_episodes"] = args.episodes
    if args.algo:
        config["algo"] = args.algo
    if args.lr is not None:
        config["lr"] = args.lr
    if args.clip_param is not None:
        config["clip_param"] = args.clip_param
    if args.update_interval is not None:
        config["update_interval"] = args.update_interval
    if args.save_interval is not None:
        config["save_interval"] = args.save_interval
    if args.max_steps is not None:
        config["max_steps_per_episode"] = args.max_steps
    if args.num_agents is not None:
        config["num_agents"] = args.num_agents

    config["num_collect_envs"] = max(1, args.num_collect_envs)
    config["map_name"] = args.map
    config["step_mul"] = args.step_mul
    config["debug_force_spread"] = bool(args.debug_force_spread)

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # Only allow StarCraft II rendering in evaluation mode.
    visualize = bool(args.eval and args.visualize)
    if args.visualize and not args.eval:
        logger.info("Visualization requested during training, but it is disabled unless --eval is set.")

    # Hide native SC2 client windows during training (or any non-visual run).
    if visualize:
        os.environ.pop("PYSC2_HIDE_WINDOW", None)
    else:
        os.environ["PYSC2_HIDE_WINDOW"] = "1"

    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable, fallback to CPU.")
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Simplebattle self-play eval path: both sides same algorithm.
    if args.eval and config.get("algo") == "simplebattle":
        if args.num_agents is not None:
            logger.warning("num_agents does not change map unit count in simplebattle self-play; unit count is map-defined.")
        logger.info("Running simplebattle self-play evaluation...")
        result = evaluate_simplebattle_self_play(
            map_name=args.map,
            step_mul=args.step_mul,
            max_steps=config["max_steps_per_episode"],
            visualize=visualize,
            hold_seconds=args.hold_seconds,
            logger=logger,
            race=args.simplebattle_race,
        )
        logger.info(
            "EVAL_METRIC avg_reward=%.4f avg_win_rate=%.4f avg_length=%.2f",
            result["avg_reward"],
            result["avg_win_rate"],
            result["avg_length"],
        )
        logger.info(f"Final Evaluation: {result}")
        return

    run_name = args.run_name if args.run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    config["save_dir"] = save_dir
    config["control_file"] = args.control_file

    logger.info("Initializing SC2 environment...")
    env = SC2MultiAgentEnv(
        map_name=args.map,
        step_mul=args.step_mul,
        visualize=visualize,
        obs_size=int(config.get("obs_dim", 10)),
    )
    _ = env.reset()

    detected_agents = env.n_agents
    if detected_agents <= 0:
        raise RuntimeError("No controllable self units detected on map. Cannot initialize multi-agent policy.")

    if args.num_agents is not None and args.num_agents != detected_agents:
        logger.warning(
            f"Requested num_agents={args.num_agents}, but map detected {detected_agents}. "
            "For per-unit control, using detected map unit count."
        )
    config["num_agents"] = detected_agents

    obs_space = env.observation_space[0] if env.observation_space else Box(low=0, high=1, shape=(config["obs_dim"],))
    act_space = env.action_space[0] if env.action_space else Discrete(config["act_dim"])

    logger.info(f"Initializing agent for {detected_agents} units...")
    agent = HARLAgentWrapper(config, obs_space, act_space, device)

    if args.model_path:
        if os.path.exists(args.model_path):
            logger.info(f"Loading pretrained model from: {args.model_path}")
            try:
                agent.load_model(args.model_path)
            except Exception as e:
                logger.error(f"Failed to load model from {args.model_path}: {e}")
                raise
        else:
            logger.warning(f"Model path does not exist, skip loading: {args.model_path}")

    trainer = Trainer(agent, env, config)

    if args.eval:
        logger.info("Running evaluation...")
        eval_stats = trainer.evaluate(num_episodes=args.eval_episodes)
        logger.info(f"Final Evaluation: {eval_stats}")
    else:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Running final evaluation...")
        eval_stats = trainer.evaluate(num_episodes=args.eval_episodes)
        logger.info(f"Final Evaluation: {eval_stats}")

    trainer.close()
    env.close()


if __name__ == "__main__":
    main()
