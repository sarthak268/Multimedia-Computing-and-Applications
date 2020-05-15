import torch
import torch.nn as nn
import torch.nn.functional as F


class word2vec_network(nn.Module):
	def __init__(self, inp_size, emb_size):
		super(word2vec_network, self).__init__()
		self.fc1 = nn.Linear(inp_size, emb_size)
		self.fc2 = nn.Linear(emb_size, inp_size)
		self.softmax = nn.Softmax()
		
	def forward(self, x, batch_size):
		embedding = self.fc1(x.view(batch_size, -1))
		#output = self.softmax(self.fc2(embedding))
		output = F.log_softmax(self.fc2(embedding))

		return embedding, output