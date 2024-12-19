import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from longgmp import longclip
from torch.optim.lr_scheduler import OneCycleLR
import random
from colorama import Fore, Style
from tqdm import tqdm
from adabelief_pytorch import AdaBelief


training_losses = []
validation_losses = []
print("\n")

# Save training plots with matplotlib to:
plots_folder = 'ft-plots'
os.makedirs(plots_folder, exist_ok=True)

# Save model .pt files to: 
ft_checkpoints_folder = 'ft-checkpoints'
os.makedirs(ft_checkpoints_folder, exist_ok=True)

# Save verbose text / training logs to:
text_logs_folder = 'ft-logs'
os.makedirs(text_logs_folder, exist_ok=True)


"""
METRICS
"""
def adjust_unfreeze_rate(epoch, adjust_after=12, increase_rate=2):
    if epoch < adjust_after:
        return 1  # Initial slower unfreeze rate
    else:
        return increase_rate  # Increased rate after initial pass

def unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=False):
    if unfreeze_all:
        print("All params require gradient")
        for param in model.parameters():
            param.requires_grad = True
    else:
        unfreeze_every_n_epochs = adjust_unfreeze_rate(epoch)
        layers_to_unfreeze = (epoch // unfreeze_every_n_epochs) % total_layers
        layers_to_unfreeze = min(layers_to_unfreeze, total_layers)
        for i, (name, param) in enumerate(model.named_parameters()):
            if i >= total_layers - layers_to_unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False

def monitor_gradient_norms(gradient_norms, threshold=1e-5):
    alert_messages = []
    for name, norms in gradient_norms.items():
        mean_norm = sum(norms) / len(norms)
        if mean_norm < threshold:  # Vanishing gradient
            alert_messages.append(Fore.RED + f"Vanishing gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
        elif mean_norm > 1000:  # Exploding gradient
            alert_messages.append(Fore.RED + f"Exploding gradient detected in {name} with mean norm {mean_norm:.2e}" + Style.RESET_ALL)
    if alert_messages:
        for message in alert_messages:
            print(message)

def cap_gradients(model, max_value=1e6):
    for name, param in model.named_parameters():
        if param.grad is not None:  # Check if the parameter has gradients
            grad_norm = param.grad.norm().item()
            if grad_norm > max_value:  # Cap only gradients exceeding max_value
                param.grad.data = param.grad.data * (max_value / grad_norm)

def plot_gradient_norms(gradient_norms, epoch, use_log_scale=True):
    plt.figure(figsize=(20, 10))
    
    cmap = plt.get_cmap('Spectral')
    sorted_layers = sorted(gradient_norms.items(), key=lambda item: max(item[1]), reverse=True)
    colors = cmap(range(len(sorted_layers)))
    
    for (layer_name, norms), color in zip(sorted_layers, colors):
        plt.plot(norms, label=layer_name, color=color)

    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    plt.legend(loc='upper right', fontsize='small')
    
    if use_log_scale:
        plt.yscale('log')
        plt.title(f'Gradient Norms for Epoch {epoch}{" - Log Scale" if use_log_scale else ""}')
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}_log.png")
    else:
        plt.savefig(f"{plots_folder}/gradient_norms_epoch_{epoch}.png")
    
    plt.close()

def plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts):
    epochs_x = range(1, epoch + 2)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    if len(training_losses) == len(epochs_x):
        plt.plot(epochs_x, training_losses, label='Training Loss')
    if len(validation_losses) == len(epochs_x):
        plt.plot(epochs_x, validation_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    if len(logits_images) == len(epochs_x):
        plt.plot(epochs_x, logits_images, label='Average Logits')
    if len(logits_texts) == len(epochs_x):
        plt.plot(epochs_x, logits_texts, label='Average Logits')
    plt.title('Average Logits Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Logits')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plots_folder}/combined_plot_epoch_{epoch + 1}.png")
    plt.close()

def calculate_metrics(logits, ground_truth):
    preds = torch.argmax(logits, dim=1)
    acc = accuracy_score(ground_truth.cpu(), preds.cpu())
    f1 = f1_score(ground_truth.cpu(), preds.cpu(), average='weighted')
    return acc, f1


"""
DATASETS
"""
class AttackDataset(Dataset):
    def __init__(self, attack_folder, transform=None):
        self.attack_folder = attack_folder
        self.transform = transform
        self.attack_images = []
        self.attack_texts = []

        for filename in os.listdir(attack_folder):
            if filename.endswith(".png"):
                self.attack_images.append(filename)
                corresponding_txt = filename.replace(".png", ".txt")
                with open(os.path.join(attack_folder, corresponding_txt), 'r') as f:
                    label = f.readline().strip()
                    self.attack_texts.append(label)

    def __len__(self):
        return len(self.attack_images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.attack_folder, self.attack_images[idx])
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        text = longclip.tokenize([self.attack_texts[idx]])
        return image, text.squeeze(0)

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        
        if len(labels) == 3:
            label = random.choice([labels[0], labels[1], labels[2]])
        elif len(labels) == 2:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[0]  # Fallback to the first label if less than 2 are available
        else:
            label = ''  # Fallback if no labels are available

        text = longclip.tokenize([label])  # Tokenize the label

        return image, text.squeeze(0)  # Remove the extra dimension

class LoopingDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]

    def __len__(self):
        return len(self.dataset)  # Return a large number to loop


"""
LOSS
"""
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, smoothing=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.smoothing = smoothing

    def forward(self, logits_per_image, logits_per_text):
        # Normalize the features to avoid overflow or underflow
        logits_per_image = F.normalize(logits_per_image, p=2, dim=1)
        logits_per_text = F.normalize(logits_per_text, p=2, dim=1)

        # Calculate logits
        logits = torch.matmul(logits_per_image, logits_per_text.t()) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Apply label smoothing
        N = logits.size(0)
        smoothed_labels = torch.full_like(logits, self.smoothing / (N - 1))
        smoothed_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)

        # Calculate loss manually using log-softmax and smoothed labels
        log_probs = F.log_softmax(logits, dim=1)
        loss_img = -(smoothed_labels * log_probs).sum(dim=1).mean()

        log_probs = F.log_softmax(logits.t(), dim=1)
        loss_txt = -(smoothed_labels * log_probs).sum(dim=1).mean()

        return (loss_img + loss_txt) / 2


"""
HOOKS
"""
class DynamicFeatureScalerHook:
    def __init__(self, model, scale_factor):
        self.model = model
        self.scale_factor = scale_factor
        self.handles = []

    def capture_top_neurons(self, activations, num_neurons):
        # Sort and capture top neurons dynamically
        top_indices = torch.argsort(activations.mean(dim=0), descending=True)[:num_neurons]
        return top_indices.tolist()

    def register_dynamic_hooks(self, layers_to_hook):
        for layer_idx in layers_to_hook:
            layer = self.model.visual.transformer.resblocks[layer_idx].mlp.c_fc

            def hook_fn(module, input, output):
                # Dynamically adjust top neurons during forward pass
                top_neurons = self.capture_top_neurons(output, 8 if layer_idx < 21 else 4)
                for idx in top_neurons:
                    output[:, :, idx] *= self.scale_factor
                return output

            handle = layer.register_forward_hook(hook_fn)
            self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# Dont mind this tiny func
def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()
        

"""
CONFIGURATION
"""      
    
contrastive_loss = ContrastiveLoss(temperature=0.07)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# Get Long-CLIP original models: https://github.com/beichenzbc/Long-CLIP
clipmodel = 'path/to/LongCLIP/checkpoints/longclip-L.pt'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load(clipmodel, device=device)


unfreeze_all = True

EPOCHS = 10
max_learning_rate = 3e-6
learning_rate = 3e-7
batch_size = 30



# Training dataset and dataloader: COCO-SPRIGHT, cropped to square.
# Get the images here: https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco (you already have the labels .json via my repo)

# NOTE! These mixed .json files contain LONG labels, >77 tokens; using them to fine-tune a normal (not-Long) CLIP will throw an error!
dataset1 = ImageTextDataset("path/to/COCO/data-square", "long_triple-coco-SPRIGHT-train-0_9.json", transform=preprocess)
concatenated_dataset = ConcatDataset([dataset1])  


# Dataset 2/2: Download from https://huggingface.co/datasets/zer0int/CLIP-adversarial-typographic-attack_text-image and put in "attack" subfolder
num_attacks = 12
attack_folder = "attack"
attack_dataset = AttackDataset(attack_folder, transform=preprocess)
concat_attack_dataset = ConcatDataset([attack_dataset])
looping_attack_dataset = LoopingDataset(concat_attack_dataset)
attack_dataloader = DataLoader(looping_attack_dataset, batch_size=num_attacks, shuffle=True)

# Concat all train data
combined_dataset = ConcatDataset([concatenated_dataset, concat_attack_dataset])
train_dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)


# Validation dataset and dataloader
val_dataset1 = ImageTextDataset("path/to/COCO/data-square", "long_triple-coco-SPRIGHT-val-10_11.json", transform=preprocess)
concatenated_valdataset = ConcatDataset([val_dataset1])
val_dataloader = DataLoader(concatenated_valdataset, batch_size=batch_size, shuffle=False)

total_steps = len(train_dataloader) * EPOCHS



# Define parameter groups for different learning rates
visual_parameters = [p for p in model.visual.transformer.parameters() if p.requires_grad]
transformer_parameters = [p for p in model.transformer.parameters() if p.requires_grad]

param_groups = [
    {'params': visual_parameters, 'lr': 3e-7},
    {'params': transformer_parameters, 'lr': 3e-7},
    {'params': model.token_embedding.parameters(), 'lr': 3e-7},
    {'params': [model.positional_embedding, model.visual.positional_embedding, model.visual.class_embedding], 'lr': 1e-7},
    {'params': [model.visual.proj, model.text_projection], 'lr': 3e-8},
    {'params': [model.visual.ln_pre.weight, model.visual.ln_pre.bias, model.visual.ln_post.weight, model.visual.ln_post.bias], 'lr': 3e-8},
    {'params': [model.ln_final.weight, model.ln_final.bias, model.visual.conv1.weight], 'lr': 3e-8}
]

accumulation_steps = 2  # Effective batch size will be batch_size * accumulation_steps

optimizer = AdaBelief(param_groups, lr=learning_rate, eps=1e-14, betas=(0.9, 0.999), weight_decay=1e-3, weight_decouple=True, rectify=True, print_change_log=False)

scheduler = OneCycleLR(optimizer, max_lr=max_learning_rate, total_steps=total_steps, pct_start=0.3, anneal_strategy='cos')

model = model.float()

print(f"Precision: {model.dtype}")
print(f'Total batches: {len(train_dataloader)} @ Batch Size: {batch_size}')
print("== START == \n")


"""
TRAIN
"""    

def trainloop():
    contrastive_loss = ContrastiveLoss(temperature=0.07).to(device)
    logits_images = []
    logits_texts = []
    logits_per_image = []
    logits_per_text = []

    scaler = GradScaler()
    stopping_epoch = 8  # Stop hook manipulation after this epoch
    accumulation_steps = 2

    for epoch in range(EPOCHS):
        gradient_norms = {}
        unfreeze_layers(model, epoch, total_layers=24, unfreeze_all=unfreeze_all)
        model.train()
        total_train_loss = 0.0
        train_accs, train_f1s, val_accs, val_f1s = [], [], [], []
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{EPOCHS}', leave=True)

        optimizer.zero_grad()

        attack_iter = iter(attack_dataloader)

        for batch_idx, (images, texts) in progress_bar:
            batch_logits_images = []
            batch_logits_texts = []
            # Fetch attack samples
            try:
                attack_batch = next(attack_iter)
            except StopIteration:
                attack_iter = iter(attack_dataloader)  # Restart the iterator
                attack_batch = next(attack_iter)

            attack_images, attack_texts = attack_batch
            attack_images, attack_texts = attack_images.to(device), attack_texts.to(device)

            # Fetch normal samples
            normal_images, normal_texts = images.to(device), texts.to(device)

            with autocast():
                # Note to self: kind of double-do to check for stopping_epoch AND here...
                if epoch < stopping_epoch:
                    logits_per_image_attack, logits_per_text_attack = model(attack_images, attack_texts)
                    
                    if epoch <1:
                        # Initially tried 'slow warm up' (scale factor 10), doesn't improve result, alas:
                        dynamic_hook = DynamicFeatureScalerHook(model, scale_factor=100)
                        dynamic_hook.register_dynamic_hooks(layers_to_hook=[14, 16, 18, 19, 20, 21, 22])
                    elif epoch >=1 and epoch <9:
                        # Register dynamic hooks for neuron manipulation
                        dynamic_hook = DynamicFeatureScalerHook(model, scale_factor=100)
                        dynamic_hook.register_dynamic_hooks(layers_to_hook=[14, 16, 18, 19, 20, 21, 22])
                    else:
                        # Leaving this here in case needs modifying, albeit scaling to 1 = doing nothing
                        dynamic_hook = DynamicFeatureScalerHook(model, scale_factor=1)
                        dynamic_hook.register_dynamic_hooks(layers_to_hook=[14, 16, 18, 19, 20, 21, 22])                        

                    # Calculate loss for attack samples
                    attack_loss = contrastive_loss(logits_per_image_attack, logits_per_text_attack)

                    # Backward pass with manipulated activations
                    scaler.scale(attack_loss).backward()

                    # Cap gradients specifically for attack samples
                    # Nope, didn't need that for Long-CLIP (unlike for 77-tokens CLIP!)
                    #cap_gradients(model, max_value=1e5)

                    # Remove hooks after backward
                    dynamic_hook.remove_hooks()
                else:
                    attack_loss = 0  # No attack loss after stopping_epoch

                # Handle normal loss without gradient manipulation
                logits_per_image_normal, logits_per_text_normal = model(normal_images, normal_texts)
                normal_loss = contrastive_loss(logits_per_image_normal, logits_per_text_normal)
                             
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        grad_norm = parameter.grad.norm().item()
                        gradient_norms.setdefault(name, []).append(grad_norm)

                # Comment out this if getting too much red warning spam about gradients
                monitor_gradient_norms(gradient_norms)

            # Combine losses
            total_loss = attack_loss + normal_loss
            scaler.scale(normal_loss).backward()

            batch_logits_images.append(logits_per_image_normal.mean().item())
            batch_logits_texts.append(logits_per_text_normal.mean().item())

            # Step optimizer after accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
           
            # Track batch losses
            total_train_loss += total_loss.item()
            train_accs.append(accuracy_score(torch.arange(logits_per_image_normal.size(0), device=device).cpu(), logits_per_image_normal.argmax(dim=1).cpu()))
            train_f1s.append(f1_score(torch.arange(logits_per_image_normal.size(0), device=device).cpu(), logits_per_image_normal.argmax(dim=1).cpu(), average='weighted'))

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{total_train_loss / (batch_idx + 1):.4f}', 'attack': f'{attack_loss:.4f}', 'normal': f'{normal_loss:.4f}'})
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)

        epoch_avg_logits_image = sum(batch_logits_images) / len(batch_logits_images)
        epoch_avg_logits_text = sum(batch_logits_texts) / len(batch_logits_texts)
        logits_images.append(epoch_avg_logits_image)
        logits_texts.append(epoch_avg_logits_text)

        plot_gradient_norms(gradient_norms, epoch)

        epoch_train_acc = sum(train_accs) / len(train_accs)
        epoch_train_f1 = sum(train_f1s) / len(train_f1s)
        with open(f"{text_logs_folder}/log_details_train.txt", "a", encoding='utf-8') as f:
            f.write(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_train_loss:.4f}, Training Acc: {epoch_train_acc:.4f}, Training F1: {epoch_train_f1:.4f}\n")

        # Validation
        model.eval()
        total_val_loss = 0.0
        print("Running Validation...")
        with torch.no_grad():
            for images, texts in val_dataloader:
                current_batch_size = images.size(0)
                ground_truth = torch.arange(current_batch_size, device=device)
                images, texts = images.to(device), texts.to(device)
                logits_per_image, logits_per_text = model(images, texts)
                val_loss = contrastive_loss(logits_per_image, logits_per_text)
                total_val_loss += val_loss.item()
                val_acc, val_f1 = calculate_metrics(logits_per_image, ground_truth)
                val_accs.append(val_acc)
                val_f1s.append(val_f1)

        avg_val_loss = total_val_loss / len(val_dataloader)
        validation_losses.append(avg_val_loss)
        if epoch >= 1:
            plot_training_info(epoch, training_losses, validation_losses, logits_images, logits_texts)

        epoch_val_acc = sum(val_accs) / len(val_accs)
        epoch_val_f1 = sum(val_f1s) / len(val_f1s)

        if epoch >= 1:
            plt.figure(figsize=(10, 5))
            plt.plot(range(1, epoch + 2), training_losses, label='Training Loss')
            plt.plot(range(1, epoch + 2), validation_losses, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Over Epochs')
            plt.legend()
            plt.savefig(f"{plots_folder}/loss_plot_epoch_{epoch + 1}.png")
            plt.close()

        print(Fore.YELLOW + "======================== STATS =============================")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}")
        print(Fore.YELLOW + f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print(Fore.YELLOW + "============================================================" + Style.RESET_ALL)

        with open(f"{text_logs_folder}/log_training.txt", "a", encoding='utf-8') as f:
            f.write("======================== STATS =============================\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Validation Acc: {epoch_val_acc:.4f}, Validation F1: {epoch_val_f1:.4f}\n")
            f.write(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
            f.write("============================================================\n")

        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            model_path = f"{ft_checkpoints_folder}/clip_ft_{epoch+1}.pt"
            torch.save(model, model_path)
            print(Fore.GREEN + f"Model saved: {model_path}" + Style.RESET_ALL)

trainloop()
