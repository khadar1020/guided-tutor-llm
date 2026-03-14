import argparse
import os
import pathlib

import torch
from tqdm import tqdm
from tqdm.contrib import tzip
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import Dataset, Example, Scores
from tools import escape_template, safe_eval, JudgeLLM, ClientLLM


def main(
        dataset: pathlib.Path,
        inference_prompt: pathlib.Path,
        eval_prompt: pathlib.Path,
        judge_llm: ClientLLM,
        model_path: str,
        output_path: pathlib.Path
) -> None:
    with open(eval_prompt, "r", encoding="utf-8") as file:
        judge_llm_prompt = escape_template(file.read())

    with open(inference_prompt, "r", encoding="utf-8") as file:
        inference_prompt_template = escape_template(file.read())

    with open(dataset) as f:
        eval_prompts = Dataset.model_validate_json(f.read())

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    answers = []
    for prompt in tqdm(eval_prompts):
        _prompt = inference_prompt_template.format(input=prompt)
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": _prompt}, ], tokenize=False, add_generation_prompt=True
        )
        encoded_inputs = tokenizer([formatted, ], return_tensors="pt").to("cuda")

        generate_kwargs = dict(encoded_inputs, max_new_tokens=250)

        output = model.generate(**generate_kwargs)
        response = tokenizer.decode(output[0], skip_special_tokens=True)[len(_prompt) + 1:]
        answers.append(response)

    client: ClientLLM = judge_llm
    scores = Scores()
    for prompt, answer in tzip(eval_prompts, answers):
        student = answer.split("Student")[0] if "Student" in answer else answer

        raw_evaluation, error, evaluation = safe_eval(
            client, judge_llm_prompt.format(conversation=prompt, answer=student)
        )

        scores.root.append(
            Example(
                prompt=prompt,
                output=answer,
                raw_evaluation=raw_evaluation,
                evaluation_error=error,
                evaluation=evaluation
            )
        )

    with open(output_path, "w") as f:
        f.write(scores.model_dump_json(indent=2))

    os.chmod(output_path, 0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="EVAL-MODEL",
        description="Evaluate the model using the Judge LLM."
    )
    parser.add_argument("--input", required=True, type=pathlib.Path,
                        help="Path to evaluation datasets")
    parser.add_argument("--inference-prompt", required=True, type=pathlib.Path,
                        help="Path to the inference prompt")
    parser.add_argument("--eval-prompt", required=True, type=pathlib.Path,
                        help="Path to the judge evaluation prompt")
    parser.add_argument(
        "--judge-llm", required=True, nargs=3, action=JudgeLLM,
        help="Service to use of the judge LLM. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct)."
    )
    parser.add_argument("--model-path", required=True, type=str, help="HF model name or path to model weights")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Path to assessment")
    args = parser.parse_args()

    main(args.input, args.inference_prompt, args.eval_prompt, args.judge_llm, args.model_path, args.output)
