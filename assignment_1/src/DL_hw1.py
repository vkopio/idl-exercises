"""
   Introduction to Deep Learning (LDA-T3114)
   Skeleton Code for Assignment 1: Sentiment Classification on a Feed-Forward Neural Network

   Hande Celikkanat & Miikka Silfverberg
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from data_semeval import *
from paths import data_dir

N_CLASSES = len(LABEL_INDICES)
N_EPOCHS = 10
LEARNING_RATE = 0.5
BATCH_SIZE = 100
REPORT_EVERY = 1
IS_VERBOSE = False

def make_bow(tweet, indices):
    feature_ids = list(indices[tok] for tok in tweet['BODY'] if tok in indices)
    bow_vec = torch.zeros(len(indices))
    bow_vec[feature_ids] = 1
    return bow_vec.view(1, -1)

def generate_bow_representations(data):
    vocab = set(token for tweet in data['training'] for token in tweet['BODY'])
    vocab_size = len(vocab) 
    indices = {w:i for i, w in enumerate(vocab)}
  
    for split in ["training","development.input","development.gold",
                  "test.input","test.gold"]:
        for tweet in data[split]:
            tweet['BOW'] = make_bow(tweet,indices)

    return indices, vocab_size

# Convert string label to pytorch format.
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])

def make_target(tweets):
    list_of_targets = list(
        map(lambda tweet: label_to_idx(tweet['SENTIMENT']), tweets)
    )

    return torch.cat(list_of_targets)

def make_batch(tweets):
    list_of_bows = list(
        map(lambda tweet: tweet['BOW'], tweets)
    )

    return torch.cat(list_of_bows)

class FFNN(nn.Module):
    def __init__(self, vocab_size, n_classes, extra_arg_1=None, extra_arg_2=None):
        super(FFNN, self).__init__()
        self.hidden = nn.Linear(vocab_size, 6)
        self.output = nn.Linear(6, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.log_softmax(self.output(x), dim=1)

        return x

data = read_semeval_datasets(data_dir)
indices, vocab_size = generate_bow_representations(data)

model = FFNN(vocab_size, N_CLASSES)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#--- training ---
for epoch in range(N_EPOCHS):
    total_loss = 0

    random.shuffle(data['training'])

    iterations = int(len(data['training'])/BATCH_SIZE)

    for i in range(iterations):
        print("{} / {}".format(i, iterations), end="\r", flush=True)

        tweet_batch = data['training'][i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

        batch = make_batch(tweet_batch)
        target = make_target(tweet_batch)

        prediction = model(batch)
        output = loss_function(prediction, target)

        total_loss += output.data.item()

        output.backward()
        optimizer.step()
        optimizer.zero_grad()
                              
    if ((epoch + 1) % REPORT_EVERY) == 0:
        print('epoch: %d, loss: %.4f' % (epoch + 1, total_loss / len(data['training'])))

#--- test ---
correct = 0
with torch.no_grad():
    for tweet in data['test.gold']:
        gold_class = label_to_idx(tweet['SENTIMENT'])
        prediction = model(tweet['BOW'])
        predicted = torch.argmax(prediction).data.item()

        correct += int(predicted == gold_class.data[0].item())

        if IS_VERBOSE:
            print('TEST DATA: %s, GOLD LABEL: %s, GOLD CLASS %d, OUTPUT: %d' % 
                 (' '.join(tweet['BODY'][:-1]), tweet['SENTIMENT'], gold_class, predicted))

    print('test accuracy: %.2f' % (100.0 * correct / len(data['test.gold'])))
