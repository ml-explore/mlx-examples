import ast
import re

# constants copy-pasted from transformers.tokenization_utils_base
SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
ADDED_TOKENS_FILE = "added_tokens.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"

FULL_TOKENIZER_FILE = "tokenizer.json"
_re_tokenizer_file = re.compile(r"tokenizer\.(.*)\.json")


def get_class_attribute_value(filename, class_name, attribute_name):
    with open(filename, "r") as file:
        node = ast.parse(file.read())

    for n in node.body:
        if isinstance(n, ast.ClassDef) and n.name == class_name:
            for item in n.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == attribute_name:
                            return ast.literal_eval(item.value)
    return None
