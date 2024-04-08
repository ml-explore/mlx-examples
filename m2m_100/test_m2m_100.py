if __name__ == "__main__":
    from transformers import NllbTokenizer

    model = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-1.3B")
    tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B", src_lang="eng_Latn", tgt_lang="fra_Latn")

    inputs = tokenizer("what is your name?", return_tensors="pt")

    # generated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"])

    # print(generated_tokens)
    # print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
    # print(tokenizer.lang_code_to_id["fra_Latn"])

    print(inputs)

    encoder_outputs = model.model.encoder(**inputs)

    decoder_input_ids = torch.tensor([[2, tokenizer.lang_code_to_id["fra_Latn"]]])
    print("decoder_input_ids ======>", decoder_input_ids)
    decoder_input_mask = torch.tensor([[1, 1]])

    for i in range(10):
        outputs = model.model.decoder(input_ids=decoder_input_ids, attention_mask=decoder_input_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=inputs["attention_mask"])

        logits = model.lm_head(outputs[0])
        logits = logits[:, -1, :]

        next_token = torch.argmax(logits, dim=-1)

        decoder_input_ids = torch.cat([decoder_input_ids, next_token.unsqueeze(0)], dim=-1)
        decoder_input_mask = torch.cat([decoder_input_mask, torch.ones_like(next_token).unsqueeze(0)], dim=-1)

        break
    
    print(tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True))