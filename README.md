SauerkrautLM Chatbot Script.

This Python script provides an interactive command-line chatbot powered by Hugging Face Transformers and VAGOsolutions’ Llama-3 SauerkrautLM models.
It supports real-time conversation, optimized inference with PyTorch 2.0+, and performance metrics like tokens-per-second.

✨ Features
- Supports Multiple Model Sizes
  - Default: VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct
  - Optional: VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct

- Optimized Loading
  - Runs in bfloat16 for faster inference on modern GPUs
  - Automatic device placement with device_map="auto"

- Optional Model Compilation
  - Uses torch.compile (PyTorch 2.0+) to speed up execution

- Chat History Memory
  - Maintains conversation context between turns

- Performance Metrics
  - Response time
  - Tokens per second (TPS)

- Customizable Prompting
  - Adjustable sampling parameters: temperature, top_p, and repetition_penalty

📦 Requirements
- Python 3.9+
- PyTorch 2.0+ with GPU acceleration
- Transformers
- A compatible GPU (recommended: at least 16 GB VRAM for 8B model, 48 GB+ for 70B model)

🚀 Usage
1. Clone this repository (or save the script).
2. Run the script


⚙️ Configuration
- Switch Models
  In the script (mark and unmark "#" for the model):
  - model_id = "VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct"
  - #model_id = "VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct"

- Adjust Generation Parameters
  - temperature=0.7
  - top_p=0.9
  - repetition_penalty=1.2
  - max_new_tokens=256
