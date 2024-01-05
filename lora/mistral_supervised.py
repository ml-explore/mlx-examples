"""
For Mistral supervised fine-tuning

Assumes records are of the form:

{ "input": "..",
  "output": ".." }

And renders the input using the MISTRAL prompt format

"""
from .supervised_lora import TrainingRecordHandler, main

MISTRAL_INSTRUCTION_PROMPT = "[INST] {prompt_input}[/INST] "


class MistralTrainingRecordHandler(TrainingRecordHandler):
    def get_input(self, record) -> str:
        return MISTRAL_INSTRUCTION_PROMPT.format(prompt_input=record["input"])

    def get_output(self, record) -> str:
        return record["output"]


if __name__ == "__main__":
    main(MistralTrainingRecordHandler())
