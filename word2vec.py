import tensorflow as tf
import keras
from keras import layers
import tqdm #not really functional just makes a pretty loop progress bar
import numpy as np
import datetime
import re
import string
import io
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def custom_processor(sentence):
    #Split the sentence into words based on spaces and remove punctuation:
    sentence = tf.strings.lower(sentence)
    sentence = tf.strings.regex_replace(sentence, '[%s]' % re.escape(string.punctuation), '')


    return sentence

def load_dataset(vocab_size, sequence_length):
    file_path = tf.keras.utils.get_file('shakespeare.txt', \
        'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    #Make dataset of non-empty lines in the data set
    text = tf.data.TextLineDataset(file_path).filter(lambda x: tf.strings.length(x) > 0)

    vectorize_layer = layers.TextVectorization(
        standardize=custom_processor,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    batch_size = 1024
    vectorize_layer.adapt(text.batch(batch_size))

    inverse_vocab = vectorize_layer.get_vocabulary()

    text_vectorized = text.batch(batch_size).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()
    sequences = list(text_vectorized.as_numpy_iterator())

    #for s in sequences[:20]:
    #    print(f"{s} => {[inverse_vocab[t] for t in s]}")
    return sequences, inverse_vocab

def generate_training_data_epicfail(sequences, window_size, num_ns, vocab_size, seed):
    """
    Generate positive skip grams and num_ns negative skip gams from sequences
    """
    #generate skip grams and negative skip grams using the relevant tensorflow functions
    targets, contexts, labels = [], [], []

    #Make sampling table based on Zipf's law so we don't screw shit up with "is" and "the"
    sampling_table = keras.preprocessing.sequence.make_sampling_table(vocab_size)

    for sequence in tqdm.tqdm(sequences):
        skip_grams, sg_labels = keras.preprocessing.sequence.skipgrams(
            sequence, 
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=num_ns
            )
        if skip_grams: #Make sure it ain't empty 
            #targets.append(skip_grams[0][0]) #Get the token we're looking at
            sg_contexts = list(np.reshape(np.array(skip_grams), (-1, num_ns + 1, 2))) 
            # split skip grams into 5s, 2 on the end to keep the pair relationship
            sg_labels = list(np.reshape(sg_labels, (-1, num_ns + 1))) # same thing
            for i, (sgc, scl) in enumerate(zip(sg_contexts, sg_labels)):
                targets.append(tf.constant(skip_grams[0][0], dtype="int64"))
                contexts.append(tf.constant(sgc, dtype="int64"))
                print(sgc)
                labels.append(tf.constant(scl, dtype="int64"))

        #print(skip_grams, sg_labels)
        #for sg, label in zip(skip_grams, sg_labels):
        #    targets.append(target:=sg[0]) #Walrus operator used to make it clear what sg[] are
        #    contexts.append(context:=sg[1])
        #    labels.append(label)

    #print(len(contexts), len(targets), len(labels))
    return np.array(targets), np.array(contexts), np.array(labels)

def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return np.array(targets), np.array(contexts), np.array(labels)

#All the tasks to get the data ready for the model to train on and create a model
def pipeline(embedding_dim, vocab_size=4096, sequence_length=10, window_size=2,num_ns=4,batch_size=1024,buffer_size=10000):
    sequences, inverse_vocab = load_dataset(vocab_size, sequence_length)

    targets, contexts, labels = \
        generate_training_data(sequences, window_size, num_ns, vocab_size, SEED)

    #for t, c, l in zip(targets, contexts, labels):
    #    print(c)
    #    print(f"{inverse_vocab[t]} {([inverse_vocab[a] for a in c ])}" )

    dataset = tf.data.Dataset.from_tensor_slices(((targets,contexts), labels))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return dataset, Word2Vec(vocab_size, embedding_dim, num_ns), inverse_vocab

#Word2Vec model
class Word2Vec(keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(Word2Vec, self).__init__()

        self.target_embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name = "w2v_embedding"
        )

        self.context_embedding = layers.Embedding(
            vocab_size,
            embedding_dim,
            input_length=num_ns+1
        )
        #print(num_ns+1)

    def call(self, pair):
        target,context = pair
        #target, context = tf.map_fn(lambda x: x, elems=pair, dtype=(tf.int64, tf.int64))
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)


        word_embedding = self.target_embedding(target)
        context_embedding = self.context_embedding(context)
        #print(context.shape)
        dots = tf.einsum("be,bce->bc", word_embedding, context_embedding)
        # \sigma_{batch} \sigma{embedding}
        # word_embeddding[batch][embedding] * context_embedding[batch][context][embedding] =
        # Tensor(batch, context)
        # For each context, compute the dot product with the target embedding
        # Dot product will be higher if the words share the same context
        # In other words if they have similar values at each the dot product will be higher
        # If they are orthogonal on many dimensions then they aren't similar at all

        return dots

#def custom_loss(x_logit, y_true):
#    y_true = tf.cast(y_true, dtype=tf.int64)
#    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


def main():
    dataset, model, vocab = pipeline(248, window_size=4)

    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
        metrics=['accuracy'])
    
    log_dir = "logs/word2vec/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),
        keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=1e-4,patience=10),
    ]

    model.fit(dataset, epochs=10000, callbacks=callbacks)

    weights = model.get_layer("w2v_embedding").get_weights()[0]
    
    vectors = io.open('vectors.tsv', 'w', encoding='utf-8')
    words = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.

        vec = weights[index] #Get embedding from target embedding layer
        vectors.write('\t'.join([str(x) for x in vec]) + "\n")
        words.write(word + "\n")
    
    vectors.close()
    words.close()

    #model.save("Basic_Models/word2vec_shakespeare.h5", save_format="tf")

def cosine_similarity(a,b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def vector_test():
    vectors = []
    with open('vectors.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            vector = np.fromstring(line.strip(), sep='\t')
            vectors.append(vector)
            #print(vector)

    # Load the saved metadata (words)
    words = []
    with open('metadata.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            words.append(word)

    def word_to_vec(word):
        return vectors[words.index(word)]

    word1 = word_to_vec(input("1st word: "))
    word2 = word_to_vec(input("2nd word: "))
    while True:
        wordc = input(f"Word to compare to 1st + 2nd: ")
        print(wordc)
        print(cosine_similarity(word_to_vec(wordc), (word1 + word2)))

    
if __name__ == "__main__":
    main()
    #vector_test()