import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dotenv import load_dotenv
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s:%(message)s')

# Load configuration from .env file
load_dotenv('.env')  # Loads environment variables from the .env file
FILESYSTEM_PREF = os.getenv('FILESYSTEM_PREF')  # Retrieves the FILESYSTEM_PREF variable

# Configuration
CSV_FILE = 'path/to/your/csv_file.csv'  # Replace with the actual path to your CSV file
IMAGE_FOLDER = FILESYSTEM_PREF          # Use FILESYSTEM_PREF from .env as the image folder
BATCH_SIZE = 128                        # Adjust batch size as needed
NUM_EPOCHS = 5                          # Adjust number of epochs as needed
LEARNING_RATE = 0.0001                  # Learning rate for the optimizer
IMAGE_SIZE = (224, 224)                 # Input image size for the model
SEED = 42                               # Seed for random number generators to ensure reproducibility
AUTOTUNE = tf.data.experimental.AUTOTUNE  # For optimizing data pipeline
NUM_PARALLEL_CALLS = 8                  # Number of parallel calls for data loading

# Development mode flag
DEVELOPMENT_MODE = True  # Set to True for development/testing, False for full training

# Check for GPU availability
if tf.config.list_physical_devices('GPU'):
    logging.info("GPU is available. Training will use the GPU.")
else:
    logging.info("GPU is not available. Training will use the CPU.")

# Set random seed for reproducibility
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Step 1: Load and preprocess the data

try:
    # Read the CSV file into a pandas DataFrame
    logging.info(f"Loading data from CSV file: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    logging.error(f"CSV file not found at the specified path: {CSV_FILE}")
    exit(1)
except Exception as e:
    logging.error(f"An error occurred while reading the CSV file: {e}")
    exit(1)

# Filter out rows with missing image paths
df = df[df['image'].notnull()]

# **Development Mode: Use only the first 100 images if DEVELOPMENT_MODE is True**
if DEVELOPMENT_MODE:
    logging.info("DEVELOPMENT_MODE is ON. Using a subset of 100 images for training.")
    df = df.head(100)  # Select the first 100 rows

# Combine all tags into a single list and create a mapping from tag names to indices
tag_columns = [f'tag{i}' for i in range(1, 6)]
all_tags = df[tag_columns].values.flatten()
unique_tags = set(tag for tag in all_tags if pd.notnull(tag))
tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
NUM_CLASSES = len(unique_tags)

logging.info(f"Number of unique tags (classes): {NUM_CLASSES}")

# Function to encode tags into multi-hot vectors
def encode_tags(row):
    """
    Converts the tags in a row into a multi-hot encoded vector.
    """
    # Extract tags from the row
    tags = [row[f'tag{i}'] for i in range(1, 6) if pd.notnull(row[f'tag{i}'])]
    # Convert tags to indices
    indices = [tag_to_idx[tag] for tag in tags]
    # Create a multi-hot vector
    multi_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
    multi_hot[indices] = 1.0
    return multi_hot

# Apply the encoding function to each row in the DataFrame
logging.info("Encoding tags into multi-hot vectors...")
df['labels'] = df.apply(encode_tags, axis=1)

# Step 2: Create TensorFlow Dataset

# Create lists of image paths and labels
logging.info("Preparing image paths and labels...")
image_paths = df['image'].apply(lambda x: os.path.join(IMAGE_FOLDER, x)).values
labels = np.stack(df['labels'].values)

# Create a TensorFlow Dataset from the image paths and labels
logging.info("Creating TensorFlow Dataset...")
dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

# Function to load and preprocess images
def load_and_preprocess_image(path, label):
    """
    Loads an image from a file path and preprocesses it for the model.
    """
    try:
        # Read the image file
        image = tf.io.read_file(path)
        # Decode the image into a tensor
        image = tf.image.decode_jpeg(image, channels=3)
        # Resize the image to the desired size
        image = tf.image.resize(image, IMAGE_SIZE)
        # Preprocess the image using ResNet50's preprocessing function
        image = keras.applications.resnet50.preprocess_input(image)
        return image, label
    except Exception as e:
        logging.error(f"Error loading image {path}: {e}")
        # Skip the problematic image by returning None
        return None

# Apply the preprocessing function to the dataset
logging.info("Applying data preprocessing...")
dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=NUM_PARALLEL_CALLS)

# **Filter out None values resulting from failed image loads**
dataset = dataset.filter(lambda x, y: x is not None)

# Shuffle and batch the dataset
logging.info("Configuring dataset for performance...")
dataset = dataset.shuffle(buffer_size=1000, seed=SEED)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=AUTOTUNE)

# Step 3: Define the model

# Create the base model from the pre-trained ResNet50
logging.info("Setting up the model...")
base_model = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))

# Freeze the base model to prevent its weights from being updated during training
base_model.trainable = False

# Add custom layers on top of the base model
inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)  # Pass inputs through the base model
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)  # Add dropout for regularization
outputs = layers.Dense(NUM_CLASSES, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

# Step 4: Compile the model

# Compile the model with an appropriate optimizer and loss function
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',
              metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])

logging.info("Model compiled successfully.")

# Step 5: Set up callbacks for checkpointing and monitoring

# Directory to save checkpoints
checkpoint_dir = './training_checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Callback to save the model at regular intervals
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'ckpt_{epoch}'),
    save_weights_only=True,
    save_freq='epoch'
)

# Callback for early stopping if validation loss does not improve
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# Callback for TensorBoard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs', update_freq='batch')

# Step 6: Train the model

logging.info("Starting model training...")
try:
    history = model.fit(
        dataset,
        epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping, tensorboard_callback]
    )
except Exception as e:
    logging.error(f"An error occurred during training: {e}")
    exit(1)

logging.info("Model training completed.")

# Step 7: Save the trained model

try:
    model.save('trained_model.h5')
    logging.info("Model saved to 'trained_model.h5'")
except Exception as e:
    logging.error(f"An error occurred while saving the model: {e}")
    exit(1)