import os
import random
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load and preprocess audio, padding or truncating as needed
def load_and_preprocess_audio(file_path, sr=16000, duration=3):
    audio_data, _ = librosa.load(file_path, sr=sr)
    target_length = sr * duration

    if len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:target_length]

    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec)

    return log_mel_spec

# Create triplets: anchor, positive, negative
def create_triplets(dataset_paths, labels):
    triplets = []
    unique_labels = set(labels)

    for label in unique_labels:
        label_indices = [i for i, l in enumerate(labels) if l == label]
        non_label_indices = [i for i, l in enumerate(labels) if l != label]

        for idx in label_indices:
            anchor = dataset_paths[idx]
            positive = random.choice([dataset_paths[i] for i in label_indices if i != idx])
            negative = random.choice([dataset_paths[i] for i in non_label_indices])
            triplets.append((anchor, positive, negative))

    return triplets

# Base model to extract embeddings from mel-spectrograms
def create_base_model(input_shape):
    input = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu')(input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)

    embedding = layers.Dense(128)(x)

    return Model(input, embedding)

# Triplet loss function
def triplet_loss(margin=0.2):
    def loss_fn(y_true, y_pred):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + margin, 0.0)
        return tf.reduce_mean(loss)
    return loss_fn

# Create the triplet loss network
def create_triplet_network(input_shape):
    base_model = create_base_model(input_shape)

    anchor_input = layers.Input(shape=input_shape)
    positive_input = layers.Input(shape=input_shape)
    negative_input = layers.Input(shape=input_shape)

    anchor_embedding = base_model(anchor_input)
    positive_embedding = base_model(positive_input)
    negative_embedding = base_model(negative_input)

    triplet_model = Model(inputs=[anchor_input, positive_input, negative_input], 
                          outputs=[anchor_embedding, positive_embedding, negative_embedding])
    
    return triplet_model

# Preprocess the dataset into mel-spectrograms
def preprocess_dataset(dataset_paths, sr=16000, duration=3):
    spectrograms = []
    for file_path in dataset_paths:
        mel_spec = load_and_preprocess_audio(file_path, sr, duration)
        mel_spec = np.expand_dims(mel_spec, axis=-1)  # Add channel dimension
        spectrograms.append(mel_spec)

    return np.array(spectrograms)

# Main training function
def train_triplet_network(dataset_paths, labels, epochs=10, batch_size=4, sr=16000, duration=3):
    # Preprocess dataset into mel-spectrograms
    spectrograms = preprocess_dataset(dataset_paths, sr, duration)
    input_shape = spectrograms[0].shape

    # Create triplets (anchor, positive, negative)
    triplets = create_triplets(dataset_paths, labels)

    anchors, positives, negatives = [], [], []
    for triplet in triplets:
        anchors.append(load_and_preprocess_audio(triplet[0], sr, duration))
        positives.append(load_and_preprocess_audio(triplet[1], sr, duration))
        negatives.append(load_and_preprocess_audio(triplet[2], sr, duration))

    anchors = np.array(anchors)
    positives = np.array(positives)
    negatives = np.array(negatives)

    # Define the triplet network
    triplet_network = create_triplet_network(input_shape)

    # Compile the model with the custom triplet loss
    triplet_network.compile(optimizer=Adam(), loss=triplet_loss())

    # Create dummy labels (array of zeros)
    dummy_labels = np.zeros((anchors.shape[0], 1))

    # Train the network using the triplets
    triplet_network.fit([anchors, positives, negatives], dummy_labels, epochs=epochs, batch_size=batch_size)

    # Save the trained model
    triplet_network.save("triplet_audio_model.h5")
    print("Model saved as triplet_audio_model.h5")

if __name__ == "__main__":
    # Example dataset paths and labels (replace with actual data)
    dataset_paths = ["audio1.wav", "audio2.wav", "audio3.wav", "audio4.wav"]  # Replace with actual file paths
    labels = [0, 0, 1, 1]  # Replace with actual labels for each audio file
    
    # Train the triplet network
    train_triplet_network(dataset_paths, labels, epochs=10, batch_size=4, sr=16000, duration=3)
