import argparse

from transformers import AutoModel, AutoTokenizer


def run(bert_model: str):
    batch = [
        "This is an example of BERT working on MLX.",
        "A second string",
        "This is another string.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    torch_model = AutoModel.from_pretrained(bert_model)
    torch_tokens = tokenizer(batch, return_tensors="pt", padding=True)
    torch_forward = torch_model(**torch_tokens)
    torch_output = torch_forward.last_hidden_state.detach().numpy()
    torch_pooled = torch_forward.pooler_output.detach().numpy()

    print("\n HF BERT:")
    print(torch_output)
    print("\n\n HF Pooled:")
    print(torch_pooled[0, :20])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the BERT model using Hugging Face Transformers."
    )
    parser.add_argument(
        "--bert-model",
        choices=[
            "bert-base-uncased",
            "bert-base-cased",
            "bert-large-uncased",
            "bert-large-cased",
        ],
        default="bert-base-uncased",
        help="The huggingface name of the BERT model to save.",
    )
    args = parser.parse_args()

    run(args.bert_model)
