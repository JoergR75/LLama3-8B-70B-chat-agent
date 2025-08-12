import warnings
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

# Choose your model
model_id = "VAGOsolutions/Llama-3-SauerkrautLM-8b-Instruct"
# model_id = "VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct"

# Load tokenizer and model with optimized settings
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# (Optional) Compile the model for better performance (PyTorch 2.0+ only)
model = torch.compile(model)

# Initialize the chat history
messages = [
    {"role": "system", "content": "You are a knowledgeable and articulate chatbot that provides concise and clear answers."},
]

# Function to build the prompt from history
def build_prompt(messages):
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
        else:
            prompt += f"{content}\n"
    prompt += "Assistant: "
    return prompt

# Chat loop using generate
def chat_with_bot():
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        prompt = build_prompt(messages)

        # Tokenize input and move to the right device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed_time = time.time() - start_time

        # Decode response and calculate TPS
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        bot_response = generated_text[len(prompt):].strip()

        tokens_generated = outputs.shape[-1] - inputs["input_ids"].shape[-1]
        tokens_per_second = tokens_generated / elapsed_time if elapsed_time > 0 else float("inf")

        print(f"\nChatbot: {bot_response}")
        print(f"\nResponse time: {elapsed_time:.2f} seconds")
        print(f"Tokens per second: {tokens_per_second:.2f} t/s")

        messages.append({"role": "assistant", "content": bot_response})

# Start chat
if __name__ == "__main__":
    print("Chatbot: I be ready to chat! Type exit, quit, or bye to end the conversation.")
    chat_with_bot()
