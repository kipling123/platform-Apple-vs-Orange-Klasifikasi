import os
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from PIL import Image

def build_datasets(data_dir, img_size=(150, 150), batch_size=32, seed=42):
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        class_names=['apple', 'orange'],
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
        validation_split=0.2,
        subset='training'
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='binary',
        class_names=['apple', 'orange'],
        color_mode='rgb',
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
        validation_split=0.2,
        subset='validation'
    )
    return train_ds, val_ds

def train_scratch_cnn(train_ds, val_ds, img_size=(150, 150)):
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Scratch_CNN")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training Scratch CNN...")
    start_time = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=8, verbose=1)
    train_duration = time.time() - start_time
    
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    return model, val_acc, train_duration

def train_mobilenetv2(train_ds, val_ds, img_size=(150, 150)):
    base_model = MobileNetV2(input_shape=(img_size[0], img_size[1], 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    
    print("Training MobileNetV2 Feature Extractor...")
    start_time = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=6, verbose=1)
    
    # Fine-tune last 20 layers
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=4, verbose=1)
    train_duration = time.time() - start_time
    
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    return model, val_acc, train_duration

def main():
    data_dir = 'c:/week2'
    os.makedirs('models', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    
    print("Loading image dataset from:", data_dir)
    train_ds, val_ds = build_datasets(data_dir)
    
    scratch_model, scratch_acc, scratch_time = train_scratch_cnn(train_ds, val_ds)
    scratch_path = 'models/apple_orange_scratch_cnn.keras'
    scratch_model.save(scratch_path)
    print(f"Scratch CNN saved to {scratch_path}. Val Acc: {scratch_acc:.4f}")
    
    mobilenet_model, mobilenet_acc, mobilenet_time = train_mobilenetv2(train_ds, val_ds)
    mobilenet_path = 'models/apple_orange_mobilenetv2.keras'
    mobilenet_model.save(mobilenet_path)
    print(f"MobileNetV2 saved to {mobilenet_path}. Val Acc: {mobilenet_acc:.4f}")
    
    vision_metrics = {
        'scratch_cnn': {
            'accuracy': float(scratch_acc),
            'val_loss': float(scratch_acc),
            'train_time_sec': float(scratch_time),
            'architecture': '4-Layer Conv2D + MaxPooling + Dense'
        },
        'mobilenetv2': {
            'accuracy': float(mobilenet_acc),
            'val_loss': float(mobilenet_acc),
            'train_time_sec': float(mobilenet_time),
            'architecture': 'Pre-trained MobileNetV2 + Fine-tuning'
        },
        'classes': ['apple', 'orange'],
        'img_size': [150, 150]
    }
    
    with open('artifacts/vision_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(vision_metrics, f, indent=4)
    print("Vision metrics saved to artifacts/vision_metrics.json")

if __name__ == '__main__':
    main()
