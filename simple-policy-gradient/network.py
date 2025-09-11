import torch
import torch.nn as nn
from typing import List, Optional

class Network(nn.Module):
	def __init__(self, sizes: List[int], activation=nn.Tanh, output_activation=nn.Identity):
		super().__init__()
		self.layers = nn.ModuleList()
		self.activations = nn.ModuleList()
		for i in range(len(sizes) - 1):
			self.layers.append(nn.Linear(sizes[i], sizes[i+1]))
			if i < len(sizes) - 2:
				self.activations.append(activation())
			else:
				self.activations.append(output_activation())

	def forward(self, input_tensor):
		x = input_tensor
		for layer, activation in zip(self.layers, self.activations):
			x = layer(x)
			x = activation(x)
		return x

if __name__ == "__main__":
	net = Network((5, 10, 6, 5))
	print(net)
	print(net.layers)
	print(net.activations)

	test_data = torch.rand(10, 8, 5)
	print(test_data)
	print("*" * 80)
	print(net(test_data))
