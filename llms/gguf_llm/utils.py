import sentencepiece as spm
import sentencepiece.sentencepiece_model_pb2 as model


def spm_tokenizer(metadata):
    tokens = metadata["tokenizer.ggml.tokens"]
    bos = metadata["tokenizer.ggml.bos_token_id"].item()
    eos = metadata["tokenizer.ggml.eos_token_id"].item()
    unk = metadata["tokenizer.ggml.unknown_token_id"].item()

    normalizer_spec = model.NormalizerSpec(
        name="identity",
        precompiled_charsmap=b"",
        add_dummy_prefix=True,
        remove_extra_whitespaces=False,
        normalization_rule_tsv=b"",
    )
    trainer_spec = model.TrainerSpec(
        model_type="BPE",
        vocab_size=len(tokens),
        input_format="text",
        split_by_unicode_script=True,
        split_by_whitespace=True,
        split_by_number=True,
        treat_whitespace_as_suffix=False,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        vocabulary_output_piece_score=True,
        byte_fallback=True,
        unk_id=unk,
        bos_id=bos,
        eos_id=eos,
        pad_id=-1,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        pretokenization_delimiter="",
    )
    m = model.ModelProto(trainer_spec=trainer_spec, normalizer_spec=normalizer_spec)
    scores = metadata.get("tokenizer.ggml.scores", None)
    scores = scores.tolist() if scores is not None else None
    token_types = metadata.get("tokenizer.ggml.token_type", None)
    token_types = token_types.tolist() if token_types is not None else None

    for i, token in enumerate(tokens):
        score = scores[i] if scores else 0
        token_type = token_types[i] if token_types else 0
        m.pieces.append(
            model.ModelProto.SentencePiece(piece=token, score=score, type=token_type)
        )
    tokenizer = spm.SentencePieceProcessor(model_proto=m.SerializeToString())
    return tokenizer
