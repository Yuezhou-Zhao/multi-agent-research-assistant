"""Writer-style calibration exemplars — pooled with raw arXiv abstracts
to bring the Gamma calibration reference distribution in line with what
the Critic actually scores.

Motivation (measured, not guessed): calibrated on raw abstracts alone,
Gamma's L1 rejected 79% of Writer sentences across 5 varied queries,
driving 100% force_finalized rate — the "approved" path never fired.
Root cause: Writer's synthesized-and-cited prose ("This method, known
as X, has been shown to..." [source_id]) sits in a different embedding
region than raw abstracts ("We introduce X. X achieves..."). Same
technique + honest evidence + honest response = keep the spec's cascade
thresholds at 0.05/0.25, widen the reference population instead.

Runs once. Result is cached to evaluation/writer_style_exemplars.json,
which is committed so anyone cloning the repo doesn't need to burn ~$0.03
of OpenAI budget to reproduce the calibrated guardrail. Re-generate by
deleting the JSON and re-running this file.
"""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # only affects a process that hasn't already loaded env

EXEMPLARS_PATH = Path(__file__).parent / "writer_style_exemplars.json"

# Deliberately broad topic set — the goal is coverage of "well-cited
# synthesized answers about CS/ML topics," not any one subfield.
TOPICS = [
    "transformer attention mechanisms",
    "chain-of-thought prompting",
    "retrieval-augmented generation architectures",
    "hallucination detection in language models",
    "dense passage retrieval",
    "instruction tuning of large language models",
    "reward modeling for RLHF",
    "sparse mixture-of-experts models",
    "long-context language modeling",
    "self-consistency decoding",
    "parameter-efficient fine-tuning methods",
    "in-context learning behavior",
    "chain-of-verification techniques",
    "constitutional AI training",
    "reasoning benchmarks for LLMs",
    "knowledge distillation for language models",
    "prompt engineering for zero-shot tasks",
    "vector database indexing strategies",
    "cross-encoder reranking",
    "multi-hop question answering",
    "tool use in language model agents",
    "self-refinement in LLM outputs",
    "grounded generation from documents",
    "contrastive learning for embeddings",
    "reasoning chain evaluation",
]

PROMPT = """Write a single, 4-sentence answer to the research question below. Every sentence must cite a plausible-looking arXiv-style paper id in [square brackets], e.g. [2308.12345v1]. Use ordinary academic-writer style with transition words like "Additionally," "Furthermore," "This approach" — DO NOT copy the style of a raw paper abstract. Do not add a title or bibliography.

Question: What does the recent literature say about {topic}?"""


async def generate_exemplars(topics: list[str] = TOPICS) -> list[str]:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Two exemplars per topic (different seeds via temperature) -> ~50 total
    async def one(topic: str) -> str:
        response = await llm.ainvoke(PROMPT.format(topic=topic))
        return response.content.strip()

    tasks = [one(t) for t in topics for _ in range(2)]
    return await asyncio.gather(*tasks)


def load_exemplars() -> list[str]:
    if EXEMPLARS_PATH.exists():
        with open(EXEMPLARS_PATH) as f:
            return json.load(f)
    exemplars = asyncio.run(generate_exemplars())
    with open(EXEMPLARS_PATH, "w") as f:
        json.dump(exemplars, f, indent=2)
    return exemplars


if __name__ == "__main__":
    exemplars = load_exemplars()
    print(f"Loaded {len(exemplars)} exemplars from {EXEMPLARS_PATH}")
    print()
    print("Sample:")
    print(exemplars[0])
