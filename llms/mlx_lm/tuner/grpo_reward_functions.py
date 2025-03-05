import re
from typing import Callable, List, Optional

RewardFunctions = Callable[[List[str], List[str], List[str]], List[float]]


def r1_extract_xml_answer(text: str) -> str:
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except:
        print("r1_extract_xml_answer returned empty string")
        return ""


def r1_int_reward_func(
    prompts: list, completions: list, answer: list, **kwargs
) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [0.5 if r and r.isdigit() else 0.0 for r in extracted_responses]


def r1_accuracy_reward_func(
    prompts: list, completions: list, answer: list, **kwargs
) -> list[float]:
    if not completions or not answer:
        return [0.0] * len(prompts)
    extracted_responses = [r1_extract_xml_answer(r) for r in completions]
    return [
        2.0 if r and a and r == a else 0.0 for r, a in zip(extracted_responses, answer)
    ]


def r1_soft_format_reward_func(
    prompts: list, completions: list, answer: list, **kwargs
) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)

    scores = []
    for completion in completions:
        if not completion:
            scores.append(0.0)
            continue

        reason_start = completion.find("<think>")
        reason_end = completion.find("</think>")
        answer_start = completion.find("<answer>")
        answer_end = completion.find("</answer>")

        if (
            reason_start != -1
            and reason_end != -1
            and answer_start != -1
            and answer_end != -1
            and reason_start < reason_end < answer_start < answer_end
        ):
            reason_content = completion[reason_start + 13 : reason_end].strip()
            answer_content = completion[answer_start + 8 : answer_end].strip()
            if reason_content and answer_content:
                scores.append(0.5)
                continue
        scores.append(0.0)
    return scores


def r1_strict_format_reward_func(
    prompts: list, completions: list, answer: list, **kwargs
) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)
    pattern = r"<think> .*? </think><answer> .*? </answer>"
    matches = [bool(re.search(pattern, r)) if r else False for r in completions]
    return [0.5 if match else 0.0 for match in matches]


def r1_count_xml(
    prompts: list, completions: list, answer: list, **kwargs
) -> list[float]:
    if not completions:
        return [0.0] * len(prompts)
    scores = []
    for text in completions:
        if not text:
            scores.append(0.0)
            continue
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.125
        if text.count("</think>") == 1:
            count += 0.125
        if text.count("<answer>") == 1:
            count += 0.125
        if text.count("</answer>") == 1:
            count += 0.125
        end_text = text.split("</answer>")[-1]
        count -= len(end_text) * 0.001 if len(end_text) > 0 else 0
        scores.append(max(0.0, count))
    return scores
