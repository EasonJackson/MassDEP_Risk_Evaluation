import tensorflow as tf
import re
import numpy as np
import collections
import os


# Constants
batch_size = 128      # Number of input word sampled per mini batch
sentence_span = 8     # Number of input word sampled per sentence
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a context.

valid_size = 16       # Random set of words to evaluate similarity on.
valid_window = 100    # Only pick dev samples in the head of the distribution.
num_sampled = 60      # NCE sampling rate



def get_vocabulary(data_raw):
    '''
    Argument
    data_raw - list of [list of strings] cleaned sentences

    Return
    vocab - list of string, word collections
    '''
    vocab = []
    for sentence in data_raw:
        word_list = sentence.split()
        for word in word_list:
            word = word.lower()
            word = re.sub('\W', '', word)
            if word not in vocab:
                vocab.append(word)
    return vocab


def word_int_mapping(vocab, data_raw):
    '''
    Arguments
    vocab - list of string, word collections
    data_raw - list of [list of strings] sentences

    Return
    word_to_int - dict, word to int index mapping
    int_to_word - dict, int index to word mapping
    data - list of [list of int], vectorized raw data sequences
    '''
    word_to_int = {}
    int_to_word = {}
    for index, word in enumerate(vocab):
        word_to_int[word] = index + 1
        int_to_word[index] = word

    data = []
    for sentence in data_raw:
        tmp = []
        for word in sentence.split():
            if word in word_to_int:
                tmp.append(word_to_int[word])
            else:
                tmp.append(0)
        data.append(tmp)

    return word_to_int, int_to_word, data


def generate_batch_sentence(sentence):
    '''
    Arguments:
    sentence - list of int, vectorized sentence 

    Return:
    (batch, context) - tuple of batch input words and context words as labels
    batch - np array of shape (batch_size, )
    context - np array of shape (batch_size, 1)
    '''

    #assert batch_size % num_skips == 0
    #assert num_skips <= 2 * skip_window

    batch = []
    context = []

    # [ skip_window input_word skip_window ]
    # Randomly select input word index
    for _ in range(sentence_span // num_skips):
        # Exclude the first skip_window words and last skip_window words
        input_index = random.randint(skip_window, len(sentence) - skip_window - 1)
        for _ in range(num_skips):
            batch.append(input_index)
            context_index = random.randint(input_index - skip_window, input_index + skip_window)
            # Exclude the input word
            while context_index == input_index:
                context_index = random.randint(input_index - skip_window, input_index + skip_window)
            context.append(context_index)

    return batch, context


def generate_batch(data):
    '''
    Argument
    data
    batch - np array of shape (batch_size, )
    context - np array of shape (batch_size, 1)
    '''
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    select = np.random.choice(len(data), batch_size // sentence_span, replace=False)
    tmp_count = 0
    for index in select:
        tmp_batch, tmp_context = generate_batch_sentence(data[index])
        for input_word, context_word in zip(tmp_batch, tmp_context):
            batch[tmp_count] = input_word
            context_word[tmp_count] = context_word
            tmp_count += 1

    return batch, context


def create_validate():
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    return valid_examples


def create_tensorflow_network(vocabulary_size):
    valid_examples = create_validate()

    # Build input layer
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Build embedding layer
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embedding_layer = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Build output softmax layer
    ###
    # weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    # biases = tf.Variable(tf.zeros([vocabulary_size]))
    # output = tf.matmul(embed, tf.transpose(weights)) + biases
    ###
    # Optimize time efficiency with noise contrastive estimation
    nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Trasfer labels into one-hot coded vectors
    # Calculate cross entropy through output and training labels
    ###
    # train_one_hot = tf.one_hot(train_labels, vocabulary_size)
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=train_one_hot))
    ###
    nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=embedding_layer,
                                             num_sampled=num_sampled,
                                             num_classes=vocabulary_size))

    # Construct the mini batch GD optimizer using a learning rate of 1.0.
    ###
    # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)
    ###
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(nce_loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    return optimizer, nce_loss, valid_examples, normalized_embeddings, similarity


def run(data_raw, num_steps=10000):
    with tf.Session() as session:
        init_op = tf.global_variables_initializer()
        session.run(init_op)

        optimizer, loss, valid_examples, normalized_embeddings, similarity = create_tensorflow_network()
        print("Complete neural network structure")

        vocab = get_vocabulary(data_raw)
        vocabulary_size = len(vocab)

        word_to_int, int_to_word, data = word_int_mapping(vocab, data_raw)

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_context = generate_batch(data)
            feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = int_to_word[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        close_word = int_to_word[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)

        final_embeddings = normalized_embeddings.eval()


def get_raw_text():
    files_n = os.listdir('Data/Train/N')
    files_y = os.listdir('Data/Train/Y')
    files = []
    for file in files_n:
        files.append('Data/Train/N/' + file)

    for file in files_y:
       files.append('Data/Train/Y/' + file)

    final_text = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            print('Read file: {0}'.format(file))
            doc = f.read()
            sentences = doc.split('.')
            for sentence in sentences:
                #if re.match(r'^[\w]+$', sentence) is None:
                sentence = sentence.strip()
                if len(sentence) > 0:
                    final_text.append(sentence)

    output = open('Data/final_text.txt', 'w')
    for sentence in final_text:
        output.write(sentence + '\r')


if __name__ == "__main__":
    
    #run()
    word_to_int = {}
    int_to_word = {}
    data = []
    with open('Data/word_to_int.txt', 'r') as f:
        for line in f:
            tmp = line.split(':')
            word_to_int[tmp[0]] = int(tmp[1])
            int_to_word[tmp[1].strip()] = tmp[0]

    with open('Data/data_vec.txt', 'r') as f:
        for line in f:
            tmp = [int(x.strip()) for x in line.strip()[1 : -1].split(',')]
            data.append(tmp)

