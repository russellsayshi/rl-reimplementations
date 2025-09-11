import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
from network import Network
from colorama import Fore, Style
from helpers import get_categorical_distribution, get_action, compute_loss

# Following the OpenAI spinning up tutorial
# https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5_000, render=False):
	# Create the environment
	env = gym.make(env_name, render_mode=(None if not render else 'human'))
	assert isinstance(env.observation_space, Box), "Must have continuous state space"
	assert isinstance(env.action_space, Discrete), "Must have discrete action space"

	obs_dim = env.observation_space.shape[0]
	n_actions = env.action_space.n

	# Create the network
	logits_net = Network(sizes=[obs_dim] + hidden_sizes + [n_actions])

	# Make optimizer
	optimizer = Adam(logits_net.parameters(), lr=lr)

	# For training
	def train_one_epoch():
		# Lists for logging
		batch_obs = []
		batch_actions = []
		batch_weights = []
		batch_rets = []
		batch_lens = []

		# Env-specific variables
		obs, _ = env.reset()
		done = False
		ep_rews = []

		# Render first episode of every epoch
		finished_rendering_this_epoch = False

		# Main training loop
		while True:
			# Rendering
			if not (finished_rendering_this_epoch) and render:
				env.render()

			# Save observations
			batch_obs.append(obs.copy())

			# Act in the environment
			action = get_action(logits_net, torch.as_tensor(obs, dtype=torch.float32))
			obs, rew, terminated, truncated, _ = env.step(action)
			done = terminated or truncated

			# Save action & reward
			batch_actions.append(action)
			ep_rews.append(rew)

			if done:
				# If the episode has finished, let's record the data we have
				ep_ret, ep_len = sum(ep_rews), len(ep_rews)
				batch_rets.append(ep_ret)
				batch_lens.append(ep_len)

				# The weight for each logprob(a|s) is R(tau)
				batch_weights += [ep_ret] * ep_len

				# Reset episode vars
				obs, _ = env.reset()
				done = False
				ep_rews = []

				# No more rendering this epoch
				finished_rendering_this_epoch = True

				# End experience loop if we have enough of it
				if len(batch_obs) > batch_size:
					break

		# Take a single policy gradient update step
		optimizer.zero_grad()
		batch_loss = compute_loss(
			logits_net = logits_net,
			obs = torch.as_tensor(batch_obs, dtype=torch.float32),
			actions = torch.as_tensor(batch_actions, dtype=torch.float32),
			weights = torch.as_tensor(batch_weights, dtype=torch.float32),
		)
		batch_loss.backward()
		optimizer.step()
		return batch_loss, batch_rets, batch_lens

	# Training loop!
	for i in range(epochs):
		batch_loss, batch_rets, batch_lens = train_one_epoch()
		print(
			f"{Fore.CYAN}Epoch: {i:3d}{Style.RESET_ALL} "
			f"{Fore.YELLOW}Loss: {batch_loss:.3f}{Style.RESET_ALL} "
			f"{Fore.GREEN}Return: {np.mean(batch_rets):.3f}{Style.RESET_ALL} "
			f"{Fore.MAGENTA}Ep Len: {np.mean(batch_lens):.3f}{Style.RESET_ALL}"
		)

