import tensorflow as tf
import os
import re
import string
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras import layers
from tensorflow.keras import losses


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
        '[%s]' % re.escape(string.punctuation),'')

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

BATCH_SIZE = 32
SEED = 42
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 16
EPOCHS = 10

dataset = tf.keras.utils.get_file("https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", url, untar=True,
    cache_dir='', cache_subdir="")
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory( 'aclImdb/train', batch_size=BATCH_SIZE, 
    validation_split=0.2, subset='training', seed=SEED)
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/train', batch_size=BATCH_SIZE, 
    validation_split=0.2, subset='validation', seed=SEED)
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory('aclImdb/test', batch_size=BATCH_SIZE)

vectorize_layer = TextVectorization(standardize=custom_standardization,max_tokens=MAX_FEATURES,
    output_mode='int', output_sequence_length=SEQUENCE_LENGTH)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
  layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

#model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS)

""" loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy) """

export_model = tf.keras.Sequential([vectorize_layer, model, layers.Activation('sigmoid')])
export_model.compile(loss=losses.BinaryCrossentropy(from_logits=False), 
    optimizer="adam", metrics=['accuracy'])

test_reviews= []
predictions = export_model.predict(test_reviews)