# -*- coding: utf-8 -*-

import tensorflow as tf
import zipfile
import pdb
import collections
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# global variables
data_index = 0

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()

    return data

def build_dataset(words,vocabulary_size):
    # INPUT
    # words: list of words
    # vocabulary_size: number of words collected in the dictionary
    # OUTPUT
    # data: index
    # count: count of words frequencies
    # dictionary: vocabulary with words & count
    # reverse_dictionary: index:words

    count = [["UNK",-1]] # UNK is Unknown
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()

    for word,_ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0

    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1

        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    
    return data,count,dictionary,reverse_dictionary

def generate_batch(batch_size,num_skips,skip_window):
    # Skip-Gram Module
    # num_skips: number of samples generated with one word, less than 2*skipwindow
    # skip_window: distance of two words in a pair sample
    global data_index
    
    assert batch_size % num_skips == 0 #make sure each batch has all possible samples for a word
    assert num_skips <= 2 * skip_window
    
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels =  np.ndarray(shape=(batch_size,1),dtype=np.int32)

    # number of words required while creating samples with one word
    span = 2 * skip_window + 1
    
    # `deque` dual direction query, max length is span 
    buffer = collections.deque(maxlen=span)
    
    # fill the deque 
    for _ in range(span):
        buffer.append(data[data_index])
        data_index += 1 # move data_index of the targeted word

    for i in range(batch_size // num_skips):
        # generate samples with one word
        # target is the targeted word, on the skip_window
        target = skip_window
        # exclude the target word itself
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            # each sample
            while target in targets_to_avoid:
                # generate random number until it is not in list of targets_to_avoid
                target = random.randint(0,span-1)
            # add the used word in the list of targets_to_avoid
            targets_to_avoid.append(target)
            # feature is the target word itself
            batch[i * num_skips + j] = buffer[skip_window]
            # label is buffer[target]
            labels[i * num_skips + j,0] = buffer[target]
        
        # buffer deque read the next word
        buffer.append(data[data_index])
        data_index += 1
    
    return batch,labels

def plot_with_labels(low_dim_embs,labels,filename="tsne.png"):
    assert low_dim_embs.shape[0] >= len(labels)
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y = low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,xy=(x,y),xytext=(5,2),
            textcoords="offset points",
            ha="right",va="bottom")
    plt.savefig(filename)

if __name__ == "__main__":
    
    filename = "text8.zip"
    words = read_data(filename) # the `words` is a list of 17,000,000 words
    print("Data size",len(words))

    vocabulary_size = 50000

    data,count,dictionary,reverse_dictionary = build_dataset(words,vocabulary_size)
    
    del words
    print("Most Common Words (+UNK)",count[:5])
    print("Sample data",data[:10],[reverse_dictionary[i] for i in data[:10]])
    
    '''
    # test function generate_batch
    batch,labels = generate_batch(batch_size=8,num_skips=2,skip_window=1)
    for i in range(8):
        print(batch[i],reverse_dictionary[batch[i]],"->",labels[i,0],
            reverse_dictionary[labels[i,0]])

    pdb.set_trace()
    '''
    
    # define variables
    batch_size = 128
    embedding_size = 128 # dim of vocabulary vector
    skip_window = 1
    num_skips = 2
    
    valid_size = 16 # number of words extracted for validation
    valid_window = 100 # validation words are only extracted from top 100 frequent words
    valid_examples = np.random.choice(valid_window,valid_size,replace=False)
    num_sampled = 64 # number of negative samples

    graph = tf.Graph()
    
    with graph.as_default():
        
        train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
        train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
        valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

        # generate word vectors for all words (50,000,128)
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.,1.))
        # look up vectors what represent train_inputs
        embed = tf.nn.embedding_lookup(embeddings,train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size,embedding_size],
            stddev=1./math.sqrt(embedding_size)))

        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        
        # nce loss
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases =nce_biases,
                                             labels =train_labels,
                                             inputs =embed,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))
        
        optimizer = tf.train.GradientDescentOptimizer(1.).minimize(loss)
        
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
        normalized_embeddings = embeddings/norm

        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
        similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

        init = tf.global_variables_initializer()

        num_steps = 100001

    with tf.Session(graph=graph) as sess:
        init.run()
            
        average_loss = 0
        for step in range(num_steps):
            batch_inputs,batch_labels = generate_batch(batch_size,num_skips,skip_window)

            feed_dict = {train_inputs:batch_inputs,train_labels:batch_labels}

            _,loss_val = sess.run([optimizer,loss],feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step ",step,": ",average_loss)
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i,:]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s:"%valid_word
               
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," %(log_str,close_word)

                    print(log_str)
            
        final_embeddings = normalized_embeddings.eval(session=sess)
    
    from sklearn.manifold import TSNE
    tsne = TSNE(perplexity=30,n_components=2,init="pca",n_iter=5000)

    plot_only = 100
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs,labels)








