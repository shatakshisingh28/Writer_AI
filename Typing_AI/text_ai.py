from transformers import AutoModelForCausalLM, AutoTokenizer

# Load pretrained GPT-2 model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate text
def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,       # randomness
        temperature=0.7,      # creativity
        top_k=50,             # limit choices for faster generation
        top_p=0.95
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = input("Enter your prompt: ")
result = generate_text(prompt, max_length=300)
print("\nGenerated Text:\n")
print(result)
