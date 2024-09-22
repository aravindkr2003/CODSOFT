import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Add
from PIL import Image
import matplotlib.pyplot as plt

resnet_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')

def extract_image_features(img_path):
    img = Image.open(img_path).resize((224, 224))
    img = np.expand_dims(image.img_to_array(img), axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    features = resnet_model.predict(img)
    return features

captions = [
    "A dog is running",
    "A man is riding a bike",
    "A cat is sleeping on the couch",
    "A woman is playing tennis"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
sequences = tokenizer.texts_to_sequences(captions)
max_len = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
vocab_size = len(tokenizer.word_index) + 1  

def build_captioning_model(vocab_size, max_len):
    image_input = Input(shape=(2048,))
    img_features = Dense(256, activation='relu')(image_input)
   
    caption_input = Input(shape=(max_len,))
    caption_embedding = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(caption_input)
    caption_lstm = LSTM(256)(caption_embedding)
   
    combined = Add()([img_features, caption_lstm])
   
    output = Dense(vocab_size, activation='softmax')(combined)
   
    model = Model(inputs=[image_input, caption_input], outputs=output)
    return model

model = build_captioning_model(vocab_size, max_len)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

def train_model(model, image_features, padded_sequences, epochs=10):
    for epoch in range(epochs):
        for i, caption in enumerate(padded_sequences):
           
            X_image = np.array([image_features[i]])
            X_caption = np.array([caption[:-1]])  
            y = tf.keras.utils.to_categorical(caption[1:], vocab_size)
           
            model.train_on_batch([X_image, X_caption], y)
        print(f"Epoch {epoch + 1}/{epochs} completed.")

dummy_image_path = 'path_to_your_image.jpg'
image_features = np.vstack([extract_image_features(dummy_image_path) for _ in captions])

train_model(model, image_features, padded_sequences, epochs=5)

def generate_caption(image_path, model, tokenizer, max_len):
    image_features = extract_image_features(image_path)
    caption_input = [tokenizer.word_index['a']]
   
    for _ in range(max_len):
        caption_input_padded = pad_sequences([caption_input], maxlen=max_len, padding='post')
        prediction = model.predict([image_features, caption_input_padded], verbose=0)
        predicted_word_index = np.argmax(prediction)
       
        if predicted_word_index == 0:
            break
        caption_input.append(predicted_word_index)
   
    return ' '.join([tokenizer.index_word.get(idx, '') for idx in caption_input])

test_image_path = 'path_to_your_image.jpg'
generated_caption = generate_caption(test_image_path, model, tokenizer, max_len)
print("Generated Caption:", generated_caption)

img = Image.open(test_image_path)
plt.imshow(img)
plt.axis('off')
plt.title(generated_caption)
plt.show()