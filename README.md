<div align="center">
  <img alt="Socratic LLM banner" height="220px" src="./resources/banner.png">
</div>

# Socratic LLM

Train and run a “Socratic tutor” LLM: instead of giving the final answer, it asks short, targeted questions that help a student reason their way to the solution.

This repo includes:

- Prompt templates for inference and evaluation (`templates/inference.txt`, `templates/judge_llm.txt`)
- Scripts to generate a preference dataset and fine-tune an instruct model with Direct Preference Optimization (DPO)
- A small Gradio chatbot demo (`chatbot/socratic_ui.py`)

## How the model works (high level)

1. The base instruct model generates multiple candidate teacher replies for each student prompt.
2. A *judge LLM* (OpenAI or Ollama) scores each reply on a strict rubric:
   - asks questions (yes/no)
   - stays on-topic (1–5)
   - is helpful (1–5)
   - reveals the answer (yes/no)
3. The highest-scoring reply becomes **chosen** and the lowest-scoring reply becomes **rejected**.
4. DPO fine-tuning trains the model to prefer **chosen** over **rejected**.

## How we train (more detail)

The training code is fully scripted and lives in `src/`:

1. **Start from an instruct model** (default: `microsoft/Phi-3-mini-4k-instruct`).
2. **Generate candidate tutor replies** for each seed prompt (see `datasets/*_train.json`) using the Socratic inference template in `templates/inference.txt` (the generator samples 5 candidates per prompt).
3. **Score each candidate** with a judge LLM using `templates/judge_llm.txt`, then convert the rubric into a single summary score.
4. **Build DPO pairs** by taking the best-scoring candidate as **chosen** and the worst-scoring candidate as **rejected** (`src/gen_train_dataset.py` writes `train_dataset.json`).
5. **Fine-tune with DPO** using TRL’s `DPOTrainer` (`src/train.py`). It trains the model against a frozen reference model initialized from the same base checkpoint.

To reproduce end-to-end (dataset generation → DPO fine-tune → evaluation), run `python src/pipeline.py ...` as shown below.

## Quickstart

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the end-to-end pipeline

This generates DPO datasets, fine-tunes models, evaluates them, and writes artifacts under `--output-dir`.

```bash
python src/pipeline.py \
  --instruct-model microsoft/Phi-3-mini-4k-instruct \
  --output-dir ./runs/demo \
  --judge-llm openai "$OPENAI_API_KEY" gpt-4o
```

Using Ollama as the judge:

```bash
python src/pipeline.py \
  --output-dir ./runs/demo \
  --judge-llm ollama http://localhost:11434 llama3:70b-instruct
```

Note: dataset generation and training scripts assume a CUDA GPU (`device_map="cuda"`).

### Run the chatbot demo

```bash
python chatbot/socratic_ui.py --server-port 2121
```

Then open `http://localhost:2121`.

## Prompting / examples

The scripts and demo use a simple plain-text dialogue format:

```text
Student: What can stop a fire?
Teacher: What are the things a fire needs in order to keep burning?
Student: Heat, fuel, and oxygen?
```

Example (area of a circle):

```text
Student: What is the formula for the area of a circle?
Teacher: What measurement of a circle do you usually have—radius, diameter, or circumference?
Student: I have the radius.
Teacher: If you cut a circle into many thin slices and alternate them, it starts to look like a rectangle. What would the “height” be in terms of the radius, and what would the “base” be in terms of the circumference?
Student: Height is r, and the base is about half the circumference.
Teacher: Right. Since the circumference is 2πr, half is πr. What do you get when you multiply base × height?
Student: πr².
```

When the model behaves correctly, it should *not* reveal the final answer immediately. Instead, it should ask one short question at a time and give concise hints.

You can tweak the behavior without re-training by editing `templates/inference.txt`.

## Useful scripts

- `src/gen_train_dataset.py` – generate DPO training data (chosen/rejected pairs)
- `src/train.py` – fine-tune with DPO
- `src/eval_model.py` / `src/self_eval.py` – evaluate a model (or prompt-only baseline)
- `src/pipeline.py` – run everything end-to-end

## License

MIT — see `LICENSE`.
