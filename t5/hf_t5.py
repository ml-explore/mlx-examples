import argparse

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, T5EncoderModel


def embed(t5_model: str):
    batch = [
        "translate English to German: That is good.",
        "This is an example of T5 working on MLX.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(t5_model)
    torch_model = T5EncoderModel.from_pretrained(t5_model)
    torch_tokens = tokenizer(batch, return_tensors="pt", padding=True)
    torch_forward = torch_model(**torch_tokens, output_hidden_states=True)
    torch_output = torch_forward.last_hidden_state.detach().numpy()

    print("\n TF BERT:")
    for input_str, embedding in list(zip(batch, torch_output)):
        print("Input:", input_str)
        print(embedding)
        print()


def generate(t5_model: str):
    prompt = "translate English to German: As much as six inches of rain could fall in the New York City region through Monday morning, and officials warned of flooding along the coast."
    tokenizer = AutoTokenizer.from_pretrained(t5_model)
    torch_model = AutoModelForSeq2SeqLM.from_pretrained(t5_model)
    torch_tokens = tokenizer(prompt, return_tensors="pt", padding=True).input_ids
    outputs = torch_model.generate(torch_tokens, do_sample=False, max_length=512)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the T5 model using Hugging Face Transformers."
    )
    parser.add_argument(
        "--encode-only",
        action="store_true",
        help="Only run the encoder and print the embeddings.",
        default=False,
    )
    parser.add_argument(
        "--model",
        default="t5-small",
        help="The huggingface name of the T5 model to save.",
    )
    args = parser.parse_args()
    if args.encode_only:
        embed(args.model)
    else:
        generate(args.model)
