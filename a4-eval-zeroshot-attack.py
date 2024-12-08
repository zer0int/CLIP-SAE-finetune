import torch
import clip
from PIL import Image

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the image paths
image_paths = [
    "bwcat_cat.png",
    "bwcat_dog.png",
    "bwcat_notext.png",  # Add as many image paths as needed
]

# Define the choices
choices = ["a photo of a cat", "a photo of a dog", "a photo of a text"]

# Load the original CLIP model
original_model, preprocess = clip.load("ViT-L/14", device=device)

# This = my own / from https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14
torch_model_path = "ViT-L-14-GmP-SAE-pickle-OpenAI.pt"
torch_model = torch.load(torch_model_path).to(device)

# Models to evaluate
models = {
    "Original CLIP": original_model,
    "Torch-Loaded Model": torch_model,
}

# Tokenize text once (this is shared between all models and images)
text_tokens = clip.tokenize(choices).to(device)

# Define variants
variants = {
    "standard (exp)": lambda model, img_emb, txt_emb: torch.matmul(img_emb, txt_emb.T) * model.logit_scale.exp(),
    "non-exp scale": lambda model, img_emb, txt_emb: torch.matmul(img_emb, txt_emb.T) * model.logit_scale,
}

# Function to compute results
def compute_results(model, image, text_embeddings):
    with torch.no_grad():
        # Compute image embeddings
        image_embeddings = model.encode_image(image)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        
        # Evaluate logits for variants
        results = []
        for _, compute_logits in variants.items():
            logits = compute_logits(model, image_embeddings, text_embeddings)
            probs = logits.softmax(dim=-1).squeeze()
            top_prob, top_index = probs.topk(1)
            results.append((choices[top_index.item()], top_prob.item(), probs))
        return results

# Process each image
for image_path in image_paths:
    try:
        # Preprocess image
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        
        # Compute text embeddings once per model
        with torch.no_grad():
            original_text_embeddings = original_model.encode_text(text_tokens)
            original_text_embeddings /= original_text_embeddings.norm(dim=-1, keepdim=True)
            
            torch_text_embeddings = torch_model.encode_text(text_tokens)
            torch_text_embeddings /= torch_text_embeddings.norm(dim=-1, keepdim=True)

        # Compute results for both models
        original_results = compute_results(original_model, image, original_text_embeddings)
        torch_results = compute_results(torch_model, image, torch_text_embeddings)

        # Print results side-by-side
        print(f"\nImage: {image_path}")
        print(f"{'Model: Fine-tune':<25}\t\t{'Model: ViT-L/14':<25}")
        for idx, (torch_res, original_res) in enumerate(zip(torch_results, original_results)):
            torch_choice, torch_prob, torch_probs = torch_res
            original_choice, original_prob, original_probs = original_res
            print(f"Variant {idx + 1}:")
            print(f"{torch_choice} ({torch_prob:.4f})\t\t{original_choice} ({original_prob:.4f})")
            
            print("All probabilities:")
            for i, (torch_p, original_p) in enumerate(zip(torch_probs, original_probs)):
                print(f"{choices[i]:<25}: {torch_p.item():.4f}\t\t{choices[i]:<25}: {original_p.item():.4f}")

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")