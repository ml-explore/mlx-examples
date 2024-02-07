from abc import ABC

class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation."""

    def __call__(self, token_string: str, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")

class StopSequenceCriteria(StoppingCriteria):
    """Stopping criteria that compares the generated string for presence of a
    sequence of tokens."""

    def __init__(self, stop_sequence, prompt):
        self.stop_sequence = stop_sequence
        self.prompt = prompt

    def __call__(self, tokens_string: str, **kwargs):
        # Remove prompt to prevent wanting to stop on newline but the prompt contains some
        generated_text = tokens_string.replace(self.prompt,'')
        # Check if the target sequence appears in the generated text
        if self.stop_sequence in generated_text:
            # Stop generation
            return True
        # Continue generation
        return False

    def __len__(self):
        return 1

    def __iter__(self):
        yield self
