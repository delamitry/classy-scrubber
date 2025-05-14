import os
import threading
import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM

# Hugging Face model references
CLIP_PATH = "google/siglip-so400m-patch14-384"  # Vision model (SigLIP/CLIP) path on HF
LLAMA_PATH = "John6666/Llama-3.1-8B-Lexi-Uncensored-V2-nf4"  # LLM path on HF (8B LLaMA, 4-bit)
# Default caption prompt (will be formatted with a word count)
WORDS = 200
PROMPT_TEMPLATE = (
    "In one paragraph, write a very descriptive caption for this image, describe all objects, "
    "characters and their actions, describe in detail what is happening and their emotions. "
    "Include information about lighting, the style of this image and information about camera angle "
    "within {word_count} words. Don't create any title for the image."
)  # Default prompt template:contentReference[oaicite:4]{index=4}

# Allowed image file extensions
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Flag to indicate if models are loaded, and a dictionary to hold them
models_loaded = False
models = {}  # will store 'vision_model', 'tokenizer', 'text_model', 'image_adapter'

# Define the ImageAdapter module (to integrate image features into the LLM)
import torch.nn as nn
class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract
        if self.deep_extract:
            input_features = input_features * 5  # if deep extraction, we’ll use multiple layers
        # Linear projection layers to map vision features to language model hidden size
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        # Optional layer normalization (not used in our case)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        # Optional positional embeddings for image tokens (not used here)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        # Special tokens for <|image_start|>, <|image_end|>, and <|eot_id|> (end-of-text) as learnable embeddings
        self.other_tokens = nn.Embedding(3, output_features)
        nn.init.normal_(self.other_tokens.weight, mean=0.0, std=0.02)  # initialize tokens

    def forward(self, vision_hidden_states: torch.Tensor):
        # vision_hidden_states is a tuple of hidden states from the vision model's layers
        if self.deep_extract:
            # Concatenate selected layers for deeper feature extraction (not used by default)
            x = torch.concat((
                vision_hidden_states[-2],
                vision_hidden_states[3],
                vision_hidden_states[7],
                vision_hidden_states[13],
                vision_hidden_states[20],
            ), dim=-1)
        else:
            # By default, take the second-to-last layer’s output as image features
            x = vision_hidden_states[-2]
        # Apply layer norm if enabled
        x = self.ln1(x)
        # Add positional embedding if enabled
        if self.pos_emb is not None:
            x = x + self.pos_emb
        # Project to language model hidden size
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        # Prepare <|image_start|> and <|image_end|> tokens
        # other_tokens[0] = <|image_start|>, other_tokens[1] = <|image_end|>
        batch_size = x.shape[0]
        start_end_tokens = self.other_tokens(torch.tensor([0, 1], device=x.device)).unsqueeze(0).expand(batch_size, -1, -1)
        # Insert the image embeddings between <|image_start|> and <|image_end|>
        x = torch.cat((start_end_tokens[:, :1, :], x, start_end_tokens[:, 1:2, :]), dim=1)
        return x

    def get_eot_embedding(self):
        # Get embedding for <|eot_id|> (end-of-text special token)
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)

def load_models():
    """Load the vision model, tokenizer, language model, and image adapter (one-time setup)."""
    global models_loaded, models
    if models_loaded:
        return models  # already loaded, reuse

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Load vision encoder (SigLIP/CLIP model for images)
    print("Loading vision model...")
    processor = AutoProcessor.from_pretrained(CLIP_PATH)        # For preprocessing (not heavily used in this script)
    clip_model = AutoModel.from_pretrained(CLIP_PATH)           # Load the combined CLIP model (text + vision)
    vision_model = clip_model.vision_model.to(device)           # Use only the vision sub-model
    # Load custom fine-tuned weights for the vision model (provided in LLM-CAPTION checkpoint)
    clip_checkpoint_path = os.path.join("checkpoint", "clip_model.pt")
    if os.path.exists(clip_checkpoint_path):
        print("Applying custom vision model weights...")
        checkpoint = torch.load(clip_checkpoint_path, map_location='cpu')
        # Adjust keys if necessary (strip any module prefixes)
        checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
        vision_model.load_state_dict(checkpoint)
        del checkpoint

    # 2. Load tokenizer for the language model (with special tokens for images)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("checkpoint", "text_model"), use_fast=True)

    # 3. Load language model (8B LLaMA) – use device_map=0 to load on GPU 0 if available
    print("Loading language model... (This may take a while)")
    text_model = AutoModelForCausalLM.from_pretrained(
        LLAMA_PATH, device_map=0 if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    text_model.eval()

    # 4. Load the image adapter to integrate vision features into the LLM
    print("Loading image adapter...")
    image_adapter = ImageAdapter(
        input_features=vision_model.config.hidden_size,   # CLIP vision hidden size
        output_features=text_model.config.hidden_size,    # LLM hidden size
        ln1=False, pos_emb=False, num_image_tokens=38, deep_extract=False
    )
    adapter_path = os.path.join("checkpoint", "image_adapter.pt")
    if os.path.exists(adapter_path):
        state = torch.load(adapter_path, map_location='cpu')
        image_adapter.load_state_dict(state)
    image_adapter.to(device)
    image_adapter.eval()

    # Store loaded components for reuse
    models = {
        "vision_model": vision_model,
        "processor": processor,
        "tokenizer": tokenizer,
        "text_model": text_model,
        "image_adapter": image_adapter,
        "device": device
    }
    models_loaded = True
    return models

def generate_caption_for_image(image_path):
    """Generate a caption for a single image file path using the loaded models."""
    # Ensure models are loaded
    m = load_models()
    device = m["device"]
    vision_model = m["vision_model"]
    text_model = m["text_model"]
    tokenizer = m["tokenizer"]
    image_adapter = m["image_adapter"]

    # Open and preprocess image
    img = Image.open(image_path).convert("RGB")
    # Resize and normalize image to 384x384 as expected by SigLIP model
    img_resized = img.resize((384, 384), Image.LANCZOS)
    pixel_values = torch.tensor([[[p/255.0 for p in img_resized.getdata(band)] for band in range(len(img_resized.getbands()))]])
    # The above creates a tensor of shape (1, Channels, Height*Width), we should reshape to (1, C, H, W)
    pixel_values = pixel_values.view(1, len(img_resized.getbands()), img_resized.height, img_resized.width)
    # Normalize pixel values (SigLIP expects mean 0.5, std 0.5)
    pixel_values = (pixel_values - 0.5) / 0.5
    pixel_values = pixel_values.to(device)

    # Run vision model to get hidden states (features) from the image
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        vision_outputs = vision_model(pixel_values=pixel_values, output_hidden_states=True)
    vision_hidden_states = vision_outputs.hidden_states  # tuple of hidden layers

    # Use the image adapter to get embedding vectors for the image (with <|image_start|> and <|image_end|>)
    image_embeds = image_adapter(vision_hidden_states)               # shape: (batch=1, N_image_tokens+2, hidden_size)
    image_embeds = image_embeds.to(device)

    # Build the conversation context with system and user (prompt) roles
    prompt_str = PROMPT_TEMPLATE.format(word_count=WORDS)
    conversation = [
        {"role": "system", "content": "You are a helpful image captioner."},
        {"role": "user", "content": prompt_str}
    ]
    # Format the conversation to a single string using the tokenizer's chat template (if available)
    if hasattr(tokenizer, "apply_chat_template"):
        convo_str = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback: simple role formatting if no special method
        convo_str = f"<s>[SYSTEM]{conversation[0]['content']}[/SYSTEM]\n[USER]{conversation[1]['content']}[/USER]\n"

    # Tokenize the conversation and prompt
    convo_tokens = tokenizer.encode(convo_str, return_tensors="pt", add_special_tokens=False)
    prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False)
    convo_tokens = convo_tokens.squeeze(0); prompt_tokens = prompt_tokens.squeeze(0)

    # Determine where to insert the image tokens in the token sequence:
    # Find the second <|eot_id|> token (end-of-text token indicating end of system prompt)
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>") if "<|eot_id|>" in tokenizer.vocab else tokenizer.eos_token_id
    eot_positions = (convo_tokens == eot_token_id).nonzero(as_tuple=True)[0].tolist()
    if len(eot_positions) >= 2:
        # Position right before the user prompt begins
        preamble_len = eot_positions[1] - prompt_tokens.shape[0]
    else:
        preamble_len = 0  # fallback if template is different

    # Embed all conversation tokens (text) into vectors
    convo_embeds = text_model.model.embed_tokens(convo_tokens.to(device).unsqueeze(0))
    # Construct input embeddings by inserting image embeddings at the determined position
    input_embeds = torch.cat([
        convo_embeds[:, :preamble_len, :],                         # embeddings before image
        image_embeds.to(dtype=convo_embeds.dtype),                 # embedded image tokens
        convo_embeds[:, preamble_len:, :]                          # embeddings for prompt (after image tokens)
    ], dim=1)
    # Construct input token ids with placeholders (zeros) for image tokens
    image_token_count = image_embeds.shape[1]
    input_ids = torch.cat([
        convo_tokens[:preamble_len].unsqueeze(0),
        torch.zeros((1, image_token_count), dtype=torch.long),     # dummy token IDs for image embeddings
        convo_tokens[preamble_len:].unsqueeze(0)
    ], dim=1).to(device)
    # Attention mask: 1 for all tokens (including image placeholders)
    attention_mask = torch.ones_like(input_ids)

    # Generate caption text using the language model
    output_ids = text_model.generate(
        input_ids=input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask,
        max_new_tokens=300, do_sample=True, temperature=0.6, top_p=0.9
    )
    # Slice off the input part from the generated output to get only new tokens (the caption)
    output_ids = output_ids[:, input_ids.shape[1]:]
    # Remove trailing EOS or <|eot_id|> if present
    if output_ids[0, -1].item() in [tokenizer.eos_token_id, eot_token_id]:
        output_ids = output_ids[:, :-1]
    # Decode to string
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip().strip('\"')  # strip quotes or whitespace

# Set up the Tkinter GUI
root = tk.Tk()
root.title("LLM-Caption Image Captioning")

# Directory selection and display
dir_path_var = tk.StringVar()
dir_label = tk.Label(root, text="Select an image directory:")
dir_label.pack(padx=10, pady=(10, 0))
dir_display = tk.Label(root, textvariable=dir_path_var, fg="blue")
dir_display.pack(padx=10, pady=(0, 10))

# Text box for output captions
output_text = ScrolledText(root, width=80, height=20)
output_text.pack(padx=10, pady=10)

def browse_directory():
    path = filedialog.askdirectory(title="Choose Image Directory")
    if path:
        dir_path_var.set(path)

def start_captioning():
    folder = dir_path_var.get()
    if not folder:
        output_text.insert(tk.END, "Please select a directory first.\n")
        output_text.see(tk.END)
        return
    # Clear previous output
    output_text.delete(1.0, tk.END)
    # Get list of image files in directory
    image_files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    if not image_files:
        output_text.insert(tk.END, "No images found in the selected directory.\n")
        return
    output_text.insert(tk.END, f"Generating captions for {len(image_files)} images...\n")
    output_text.see(tk.END)
    # Process each image
    for idx, filename in enumerate(image_files, start=1):
        image_path = os.path.join(folder, filename)
        output_text.insert(tk.END, f"[{idx}/{len(image_files)}] Processing {filename}...\n")
        output_text.see(tk.END)
        try:
            caption = generate_caption_for_image(image_path)
            output_text.insert(tk.END, f"→ {filename}: {caption}\n\n")
            output_text.see(tk.END)
            # Save caption to a .txt file next to the image
            txt_path = os.path.splitext(image_path)[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption)
        except Exception as e:
            output_text.insert(tk.END, f"Error processing {filename}: {e}\n")
            output_text.see(tk.END)
    output_text.insert(tk.END, "Done.\n")
    output_text.see(tk.END)

# Buttons for browsing and starting caption generation
browse_btn = tk.Button(root, text="Browse...", command=browse_directory)
browse_btn.pack(pady=5)
generate_btn = tk.Button(root, text="Generate Captions", command=start_captioning)
generate_btn.pack(pady=(0, 10))

root.mainloop()
