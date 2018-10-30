import tensorflow as tf
import re
import numpy as np


def get_volcabulary(data):
    word_list = data.split()
    volcab = []
    for word in word_list:
        word = lower(word)
        word = re.sub(r'[~a-z]', "", word)
        if word not in volcab:
            volcab.append(word)
    return volcab


def word_int_mapping(volcab):
    word_to_int = {}
    int_to_word = {}
    for index, word in enumerate(volcab):
        word_to_int[word] = index
        int_to_word[index] = word

    return word_to_int, int_to_word


data_index = 0
# generate batch data
def generate_batch(data, batch_size, num_skips, skip_window):
    '''
    Arguments:
    data - list of strings as text data
    batch_size - int, number of input word in one batch
    num_skips - int, number of skipping word to pick as prediction
    skip_window - int, size of skipping window

    Return:
    (batch, context) - tuple of batch input words and context words as labels
    batch - np array of shape (batch_size, )
    context - np array of shape (batch_size, 1)
    '''
    global data_index
    #assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window input_word skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # input word at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]  # this is the input word
            context[i * num_skips + j, 0] = buffer[target]  # these are the context words
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, context


def create_validate():
    valid_size = 16     # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    return valid_examples


def create_tensorflow_network():
    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1       # How many words to consider left and right.
    num_skips = 2         # How many times to reuse an input to generate a context.
    valid_examples = create_validate()

    # Build input layer
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Build embedding layer
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embedding_layer = tf.nn.embedding_lookup(embeddings, train_inputs)

    # Build output softmax layer
    weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocabulary_size]))
    output = tf.matmul(embed, tf.transpose(weights)) + biases

    # Trasfer labels into one-hot coded vectors
    # Calculate cross entropy through output and training labels
    train_one_hot = tf.one_hot(train_labels, vocabulary_size)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=train_one_hot))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(cross_entropy)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    return optimizer, cross_entropy, valid_examples, normalized_embeddings, similarity


def run():
    with tf.Session(graph=graph) as session:
        init_op = tf.global_variables_initializer()
        session.run(init_op)
        print('Initialized')

        optimizer, cross_entropy, valid_examples, normalized_embeddings, similarity = create_tensorflow_network()

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_context = generate_batch(data, batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_context: batch_context}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, cross_entropy], feed_dict=feed_dict)
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

