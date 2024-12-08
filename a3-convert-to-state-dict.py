import torch
import os

# Load the fine-tuned model AFTER converting back to weight with script (2)!
device = 'cuda'
THE_FINETUNED_MODEL = torch.load('ft-checkpoints/clip_ft_20_backtoweight.pt', map_location=device)

# Save only the state dictionary of the fine-tuned model
# This can be used e.g. with ComfyUI
torch.save(THE_FINETUNED_MODEL.state_dict(), 'ft-checkpoints/clip_ft_20_state-dict.pt')

