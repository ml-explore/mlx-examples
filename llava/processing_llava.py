from image_processor import CLIPImageProcessor


class LlavaProcessor:
    def __init__(self, image_processor=None, tokenizer=None):
        self.image_processor = CLIPImageProcessor()
        self.tokenizer = tokenizer

    def __call__(
        self,
        text=None,
        images=None,
        padding=False,
        truncation=None,
        max_length=None,
        return_tensors=None,
    ):
        if images is not None:
            pixel_values = self.image_processor(images)
        else:
            pixel_values = None

        return {"pixel_values": pixel_values}
