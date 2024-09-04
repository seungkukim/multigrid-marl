import os
from datetime import datetime
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_multigrid.envs.augmented_collect_game import AugCollectGameEnv

def parse_args():
    parser = argparse.ArgumentParser(description="Self-Play Training Script")
    parser.add_argument("--model_name", type=str, default=f"marl-collectgame-self-play-{datetime.now().strftime('%y%m%d-%H%M')}",
                        help="Name of the model to be saved")
    parser.add_argument("--save_dir", type=str, default="ckpt", help="Directory to save models")
    parser.add_argument("--training_steps", type=int, default=int(1e5), help="Number of training steps before saving a new opponent")
    parser.add_argument("--self_play_turns", type=int, default=20, help="Number of self-play turns")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--grid_size", type=int, default=10, help="Grid size for CollectGameEnv")
    parser.add_argument("--num_balls", type=int, default=5, help="Number of balls in CollectGameEnve")

    return parser.parse_args()

def make_dummy_env(size, num_balls):
    env = AugCollectGameEnv(
        size=size,
        num_balls=[num_balls],
        agents_index=[0],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=False,
        partial_obs=False,
        opponent_policy=[],
    )
    return env

def make_env(seed, size, num_balls, opponent):
    def _init():
        env = AugCollectGameEnv(
            size=size,
            num_balls=[num_balls],
            agents_index=[0, 0],
            balls_index=[0],
            balls_reward=[1],
            zero_sum=False,
            partial_obs=False,
            opponent_policy=[opponent],
        )
        env.seed(seed)
        return env

    return _init

if __name__ == "__main__":
    args = parse_args()

    # Create checkpoint directory
    save_dir = os.path.join(args.save_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Create initial random policy to start self-play training
    dummy_env = make_dummy_env(args.grid_size, args.num_balls)
    opponent_model = PPO(
        "MlpPolicy",
        dummy_env,
        batch_size=args.batch_size,
        verbose=2,
    )
    initial_model_path = os.path.join(save_dir, "iter_0.zip")
    opponent_model.save(initial_model_path)

    # Self-play training
    for ix in range(args.self_play_turns):
        prev_model_path = os.path.join(save_dir, f"iter_{ix}.zip")

        opponent_model = PPO.load(prev_model_path)

        # Create environment (opponent is part of environment)
        # TODO: vectorize environment
        env = make_env(0, args.grid_size, args.num_balls, opponent_model)()

        # Agent to train
        model = PPO.load(
            prev_model_path,
            env,
            batch_size=args.batch_size,
            verbose=2,
        )

        # Train the agent
        model.learn(
            total_timesteps=args.training_steps,
        )

        # Evaluate the trained agent
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(
            f"[After Iteration {ix}] mean_reward:{mean_reward:.2f}, std_reward {std_reward:.2f}"
        )

        # Save trained model
        next_model_path = os.path.join(save_dir, f"iter_{ix+1}.zip")
        model.save(next_model_path)
