import argparse
from pg import train

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--lr', type=float, default=1e-2)
	args = parser.parse_args()

	train(env_name = args.env_name, render = args.render, lr = args.lr)
