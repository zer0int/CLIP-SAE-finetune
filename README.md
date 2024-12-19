### CLIP finetune: SAE-informed adversarial training 💥🤖💫

- ⚠️ This is EXPERIMENTAL code / a repo for messing with CLIP + Sparse Autoencoders (SAE)
- For 'good, known-working' code (and more scripts + info), please see [zer0int/CLIP-fine-tune](https://github.com/zer0int/CLIP-fine-tune)!
-----
## Changes 19/DEC/2024:
- New (best) SAE-informed Long-CLIP model with 90% ImageNet/ObjectNet accuracy.
- Code is here, model is at my HF 🤗: [https://huggingface.co/zer0int/LongCLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-SAE-ViT-L-14)
-----
🔨
- Contains the code used to fine-tune my model [HF: zer0int/CLIP-SAE-ViT-L-14](https://huggingface.co/zer0int/CLIP-SAE-ViT-L-14) 🤗
- See the "attack" folder to obtain datasets required / used in 'a1-finetune.py'
- Gradients will be very large throughout training. Comment out 'monitor_gradient_norms' as needed
- Use a2 to convert GmP model back to .weight after fine-tune -> normal CLIP model (use in any 'import clip' downstream tasks)
- Use a4 to quickly zero-shot test the 3 typographic attack test images provided
-----
🔎
- The [attack dataset](https://huggingface.co/datasets/zer0int/CLIP-adversarial-typographic-attack_text-image) was curated via SAE
- Selected for typographic attack salience (i.e. CLIP's 'text obsession' -> misclassifies image, as text is highly salient to model)
- Fine-tune: Geometric Parametrization (GmP) + scaling of 'text salient' neurons top stimulating images (via SAE)
- For details about GmP, see my other repo: [zer0int/CLIP-fine-tune](https://github.com/zer0int/CLIP-fine-tune)
-----
🔬
- Info: [Toy Models of Superposition | Perturbing a single feature](https://transformer-circuits.pub/2022/toy_model/index.html#geometry-perturb)
- Reasoning: Brute-force snap those geometric bonds, hoping to force CLIP model to find better (less text obsessed) solution 😅
- ...Until I learn / find out what I am actually doing here (with regard to Sparse Autoencoders), at least. =)
- Sparse Autoencoder inspiration:
- Anthropic.AI research ["Golden Gate Claude"](https://transformer-circuits.pub/2024/scaling-monosemanticity/) + [SAE details](https://transformer-circuits.pub/2024/april-update/index.html#training-saes)
- OpenAI: Top-K activation function (replace ReLU in Sparse Autoencoders), [arxiv](https://arxiv.org/abs/2406.04093v1)
-----
💡❓
- My SAE: Encoder-Decoder, tied weights + Top-K (puzzled together from the above!)
- Is this a good autoencoder for CLIP? I don't know. 🤔
- Small hidden dimension + low Top-K => very sparse -> will learn concepts from CLIP that [with SAE-reconstructed embeds] retrieve images of very narrow concepts, e.g. ONLY stop signs.
- Huge hidden dimension (e.g. 8192) -> not so sparse, accuracy drops, more (seemingly) random encoded concepts (judging via image retrieval)
- Intermediate -> Learns complex, surprising, but meaningful concepts that are 'totally an AI-thing to encode'
- Alas: SAE empirically shown to be 'working', but is it good? What is BEST? 🤔
- Should I be using projection? Going 'back up' in the model with pinv? Hook into residual stream? I don't (yet) know! 🤷
- I will publish the code for the SAE once I am more confident in that I know what I am actually doing (and cleaned up the mess of a code 😂).
-----
🤪
For now, here's a fun concept of "things on the back of other things" in CLIP ViT-L/14 that the SAE learned:

![6](https://github.com/user-attachments/assets/2a4521b8-3a18-4c56-b68e-2e09d9280697)

Example of the effect of images the SAE had chosen as salient typographic attacks for CLIP.

![8](https://github.com/user-attachments/assets/ec3205e2-1420-4baa-a3a2-1e3100776865)

And zero-shot results via script (4):

![results-zeroshot](https://github.com/user-attachments/assets/ed3a6c24-3c49-4d27-969b-7802fe17e35f)
