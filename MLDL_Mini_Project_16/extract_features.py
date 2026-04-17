import os
import numpy as np
from tqdm import tqdm
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# Load model
model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

# Dataset path
dataset_path = "dataset"

features = []
image_names = []

def extract_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((224, 224))
        img = np.array(img)

        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        result = model.predict(img, verbose=0)
        return result.flatten()

    except:
        return None

# Loop through dataset
for file in tqdm(os.listdir(dataset_path)):
    img_path = os.path.join(dataset_path, file)

    feat = extract_features(img_path)

    if feat is not None:
        features.append(feat)
        image_names.append(file)

# Save
np.save("features/features.npy", np.array(features))
np.save("features/names.npy", np.array(image_names))

print("✅ Feature extraction completed!")