from transformers import T5EncoderModel, AutoTokenizer

import argparse


def run(t5_model: str):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the T5 model using Hugging Face Transformers."
    )
    parser.add_argument(
        "--model",
        choices=[
            "t5-small",
        ],
        default="t5-small",
        help="The huggingface name of the T5 model to save.",
    )
    args = parser.parse_args()

    run(args.model)
