model_config = {
    'language_model': {
        'hidden_size': 4096,
        'num_hidden_layers': 32,
        'intermediate_size': 11008,
        'num_attention_heads': 32,
        'rms_norm_eps': 1e-5,
        'vocab_size': 32000,
        'num_key_value_heads': 32,
        'rope_theta': 0,
        'rope_traditional': False,
        'rope_scaling': None},

    'vision_tower': {
        'num_hidden_layers': 24,
        'hidden_size': 1024,
        'intermediate_size': 4096,
        'num_attention_heads': 16,
        'num_channels': 3,
        'image_size': 336,
        'patch_size': 14
    },

    'multi_modal_projector': {
        'in_features': 1024,
        'out_features': 4096
    },

    'vision_feature_layer': -2,
    'vision_feature_selection_strategy': 'default',
    'image_token_index': 32000,
    'pad_token_id': 32001,
    'tie_word_embeddings': False,
    'vocab_size': 32064,  # TODO: confirm this value


}
