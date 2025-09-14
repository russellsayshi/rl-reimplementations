import torch
import gym
from network import FeedForwardNN # Changed from Actor, Critic


def evaluate_policy(actor, env_name, num_episodes=10):
    env = gym.make(env_name, render_mode="human") # Added render_mode
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = actor(torch.tensor(obs, dtype=torch.float)).detach().numpy() # Get action from actor
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {i+1} finished with reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    # Configuration
    env_name = "Pendulum-v1"  # Or any other environment used for training
    actor_path = "ppo_actor.pth"
    critic_path = "ppo_critic.pth"

    # Create a dummy environment to get dimensions
    dummy_env = gym.make(env_name)
    obs_dim = dummy_env.observation_space.shape[0]
    action_dim = dummy_env.action_space.shape[0]
    dummy_env.close()

    # Initialize actor and load state dict
    actor = FeedForwardNN(obs_dim, action_dim)
    actor.load_state_dict(torch.load(actor_path))
    actor.eval() # Set actor to evaluation mode

    # Evaluate policy
    print(f"Evaluating policy for {env_name} with {5} episodes.")
    evaluate_policy(actor, env_name, num_episodes=5)
