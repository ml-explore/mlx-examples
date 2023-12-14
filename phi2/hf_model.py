from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    inputs = tokenizer(
        '''def print_prime(n):
    """
    Print all primes between 1 and n
    """''',
        return_tensors="pt",
        return_attention_mask=False,
    )

    print(model(**inputs))

    # outputs = model.generate(**inputs, max_length=200)
    # text = tokenizer.batch_decode(outputs)[0]
    # print(text)
