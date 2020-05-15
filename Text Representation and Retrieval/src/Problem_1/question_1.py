import nltk
import torch
import torchvision
from torch.autograd import Variable
import torch.optim as optim
import random
from operator import itemgetter
import numpy as np
from networks import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity


def train(vocab_size, embedding_size, data, word2index, batch_size=64):

	model = word2vec_network(inp_size=vocab_size, emb_size=embedding_size).cuda()
	optimz = optim.Adam(list(model.parameters()), lr=0.00001, betas=(0.5, 0.999))
	nll_loss = nn.NLLLoss().cuda()
	#data = data[:int((len(data)/batch_size)/batch_size)*batch_size]
	data_size = len(data)

	for epochs in range(num_epochs):
		optimz.zero_grad()
		total_loss = 0.0

		for iteration in range(data_size):

			entire_batch = np.asarray(data[iteration*batch_size:iteration*batch_size+batch_size])
				
			if (entire_batch.shape[0] == batch_size):
				
				target = entire_batch[:, 0]
				target_index = itemgetter(*list(target))(word2index)
				target_index = Variable(torch.LongTensor(list(target_index))).cuda()

				neighbouring = entire_batch[:, 1]
				neighbouring_index = itemgetter(*list(neighbouring))(word2index)
				neighbouring_index = Variable(torch.LongTensor(list(neighbouring_index))).cuda()
				
				neighbouring_value = torch.zeros((batch_size, vocab_size)).cuda()
				for idx in range(neighbouring.shape[0]):
					neighbouring_value[:, neighbouring_index[idx]] = 1
				
				embedding, probability = model(neighbouring_value, batch_size)
				loss = nll_loss(probability, target_index)

				loss.backward()
				optimz.step()

				total_loss += loss.item()

				if (iteration % 1000 == 1):
					print ('Loss at iteration ' + str(iteration) + ' of epoch ' + str(epochs) + ' is ' + str(total_loss/(batch_size*iteration)))

		torch.save(model.state_dict(), './weights/model_' + str(epochs) + '.pth')

def visualize(model, vocab_size, embedding_size, vocab, epoch):

	latent = np.zeros((vocab_size, embedding_size))
	list_voc = list(vocab)

	for i in range(len(list_voc)):
		w = list_voc[i]
		
		target_index = w2i[w]
		target_index = Variable(torch.LongTensor([target_index])).cuda()

		v = torch.zeros((1, vocab_size)).cuda()
		for j in range(target_index.shape[0]):
			v[:, target_index[j]] = 1

		em, pr = model(v, 1)

		latent[i, :] = em.cpu().data.numpy()

	latent = np.asarray(latent)
	latent_2 = TSNE(n_components=2).fit_transform(latent)

	plt.scatter(latent_2[:, 0], latent_2[:, 1], c='b', s=3)
	plt.savefig('./plots/epoch_' + str(epoch) + '.png')
	plt.close()

def word2embedding(word, model, w2i, vocab_size):

	target = w2i[word]
	target = Variable(torch.LongTensor([target])).cuda()
	
	a = torch.zeros(1, vocab_size).cuda()
	for i in range(target.shape[0]):
		a[:, target[i]] = 1
	
	emb, out = model(a, 1)
	emb = emb.cpu().data.numpy()

	return emb

def visualize_similar_words(model, vocab_size, embedding_size, l_vocab, key_words, epoch, n, top_n, w2i):
	
	vocab_embeddings = np.zeros((vocab_size, embedding_size))
	key_word_embeddings = np.zeros((n, embedding_size))

	for k in range(len(key_words)):
		key = key_words[k]
		key_emb = word2embedding(key, model, w2i, vocab_size)
		key_word_embeddings[k, :] = key_emb

	for w in range(len(l_vocab)):
		word = l_vocab[w]
		word_emb = word2embedding(word, model, w2i, vocab_size)
		vocab_embeddings[w, :] = word_emb

	sim_matrix = cosine_similarity(key_word_embeddings, vocab_embeddings)

	embedding_clusters = []
	word_clusters = []
	
	for idx, word in enumerate(key_words):
		embeddings = []
		words = []

		similar_words = np.argsort(sim_matrix[idx, :])[-top_n:]
		
		for similar_word in similar_words:
			words.append(l_vocab[similar_word])
			embeddings.append(vocab_embeddings[similar_word, :])
		embedding_clusters.append(embeddings)
		word_clusters.append(words)

	embedding_clusters = np.array(embedding_clusters)
	n, m, k = embedding_clusters.shape
	tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
	embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

	filename = 'plot_similar_words_epoch_' + str(epoch)

	plot_similar_words('TSNE for Similar Words', key_words, embeddings_en_2d, word_clusters, 0.7, filename)

def plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename):
    
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')

if (__name__ == "__main__"):

	c_dim = 2
	e_dim = 32
	training = False
	num_epochs = 3
	n = 10
	top_n = 15

	corpus = nltk.corpus.abc.raw()
	corpus_splitted = corpus.split()
	vocabulary = set(corpus_splitted)
	vocabulary_size = len(vocabulary)

	w2i = {}
	for i, w in enumerate(vocabulary):
		w2i[w] = i

	selected_words = []
	l_voc = list(vocabulary)
	for i in range(n):
		selected_words.append(l_voc[random.randint(0, vocabulary_size)])
		
	skip_gram_data = []
	for i in range(c_dim, len(corpus_splitted)-c_dim-1):
		current_word = corpus_splitted[i]

		for w in range(-c_dim, c_dim+1):
			if(w!=0):
				context_window_word = corpus_splitted[i+w]
				skip_gram_data.append((current_word, context_window_word)) # appending positive samples

	
	if (training == True):
		train(vocabulary_size, e_dim, skip_gram_data, w2i)

	else:
		for e in range(num_epochs):
			model = word2vec_network(inp_size=vocabulary_size, emb_size=e_dim).cuda()
			model.load_state_dict(torch.load('./weights/model_' + str(e) + '.pth'))
			#visualize(model, vocabulary_size, e_dim, vocabulary, e)
			visualize_similar_words(model, vocabulary_size, e_dim, l_voc, selected_words, e, n, top_n, w2i)

