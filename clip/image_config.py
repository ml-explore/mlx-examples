CLIP_IMAGE_CONFIG = {'openai/clip-vit-base-patch32': {
    "crop_size": {
        "height": 224,
        "width": 224
    },
    "do_center_crop": True,
    "do_convert_rgb": True,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "feature_extractor_type": "CLIPFeatureExtractor",
    "image_mean": [
        0.48145466,
        0.4578275,
        0.40821073
    ],
    "image_processor_type": "CLIPImageProcessor",
    "image_std": [
        0.26862954,
        0.26130258,
        0.27577711
    ],
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {
        "shortest_edge": 224
    }
}}
