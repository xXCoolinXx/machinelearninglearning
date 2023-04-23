import torch
from transformers import pipeline


def main():
    generate_text = pipeline(model="databricks/dolly-v2-12b", \
        torch_dtype=torch.bfloat16, \
        trust_remote_code=True, \
        device="cpu")

    #Prompt the user for input to the model and then give an output in a loop
    while True:
        text = input("Enter text: (Enter 'q' to quit)")
        
        if text == "q":
            break

        print(generate_text(text, max_length=100, num_return_sequences=1)[0]['generated_text'])

if __name__ == "__main__":
    main()
