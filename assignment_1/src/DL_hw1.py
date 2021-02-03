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


#--- hyperparameters ---
N_CLASSES = len(LABEL_INDICES)
N_EPOCHS = 10
LEARNING_RATE = 0.05
BATCH_SIZE = 1
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

#--- model ---
class FFNN(nn.Module):
    def __init__(self, vocab_size, n_classes, extra_arg_1=None, extra_arg_2=None):
        super(FFNN, self).__init__()
        self.hidden = nn.Linear(vocab_size, 256)
        self.output = nn.Linear(256, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = F.log_softmax(self.output(x), dim=1)

        return x

#--- data loading ---
data = read_semeval_datasets(data_dir)
indices, vocab_size = generate_bow_representations(data)

#--- set up ---
model = FFNN(vocab_size, N_CLASSES) #add extra arguments here if you use
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

#--- training ---
for epoch in range(N_EPOCHS):
    total_loss = 0
    # Generally speaking, it's a good idea to shuffle your
    # datasets once every epoch.
    random.shuffle(data['training'])

    iterations = int(len(data['training'])/BATCH_SIZE)

    for i in range(iterations):
        print("{} / {}".format(i, iterations), end="\r", flush=True)

        minibatch = data['training'][i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        target = label_to_idx(minibatch[0]['SENTIMENT'])

        pred = model(minibatch[0]['BOW'])
        output = loss_function(pred, target)

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
