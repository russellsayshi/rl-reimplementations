from torch.distributions import Categorical

# Compute distribution over actions
def get_categorical_distribution(logits_net, obs):
	logits = logits_net(obs)
	return Categorical(logits=logits)

# Get action from obs
def get_action(logits_net, obs):
	return get_categorical_distribution(logits_net, obs).sample().item()

# Compute loss function
# The gradient of this function approximates the policy gradient
def compute_loss(logits_net, obs, actions, weights):
	logprobs = get_categorical_distribution(logits_net, obs).log_prob(actions)
	return -(logprobs * weights).mean()	
