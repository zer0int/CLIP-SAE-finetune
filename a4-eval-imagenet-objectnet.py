import torch
from torch.utils.data import Dataset, DataLoader
import clip as clip
from clip.model import CLIP
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os

"""
For more eval scripts, please see my repo: https://github.com/zer0int/CLIP-fine-tune
"""


# Download from https://objectnet.dev/mvt/ 
# Load csv labels file from dataset:
csv_file = 'path/to/dataset-difficulty-CLIP/data_release_2023/human_responses.csv'
# Path to the image folder that contains ALL images from the MVT dataset:
image_folder = 'path/to/dataset-difficulty-CLIP/data_release_2023/all/'

clipmodel = 'ViT-L/14'
model_path = 'ft-checkpoints/clip_ft_20_backtoweight.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

_, preprocess = clip.load(clipmodel, device=device)

model_to_eval = torch.load(model_path)
model_to_eval = model_to_eval.to(device)

class CroppedImageCSVFileDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = self.data.iloc[idx]['image']
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = self.data.iloc[idx]['label']

        return image, label

dataset = CroppedImageCSVFileDataset(csv_file, image_folder, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=48, shuffle=True)

def evaluate_model(model, dataloader):
    correct = 0
    total = 0

    for batch_images, batch_labels in tqdm(dataloader):
        batch_images = batch_images.to(device)
        batch_texts = clip.tokenize(batch_labels).to(device)

        with torch.no_grad():
            image_embeddings = model.encode_image(batch_images)
            text_embeddings = model.encode_text(batch_texts)
            logits_per_image = (image_embeddings @ text_embeddings.T).softmax(dim=-1)

            # Get the top predictions
            _, top_indices = logits_per_image.topk(1, dim=-1)

            for i, label in enumerate(batch_labels):
                if label == batch_labels[top_indices[i]]:
                    correct += 1
                total += 1

    accuracy = correct / total
    return accuracy

model_accuracy = evaluate_model(model_to_eval, dataloader)
print(f"Fine-tuned Model Accuracy on MVT ImageNet/ObjectNet: {model_accuracy}")
