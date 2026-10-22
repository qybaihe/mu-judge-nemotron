"""How the student judge sees a question and how its answer is read: one prompt per question, the same STATE first
so that the questions of one request share a prefix, and the answer read off the first token, which is always one
of the question's labels (yes/no, A-I, 0-3). Training, evaluation and serving all go through this file.
"""

from __future__ import annotations

import json

SYSTEM = (
    "You are the judge model inside a coding agent. The agent sends you its STATE at a decision point and one "
    "QUESTION about it. Judge only from STATE. Reply with the answer label alone."
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def state_text(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, indent=1, ensure_ascii=False)


def labels(question: dict) -> list[str]:
    if question["type"] == "boolean":
        return ["yes", "no"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(LETTERS[: len(question["criteria"])])


def question_text(question: dict) -> str:
    kind = question["type"]
    lines = []
    if kind == "boolean":
        lines.append(f"QUESTION (yes or no): {question['instructions']}")
        if question.get("criteria"):
            lines.append(f"Meaning: {question['criteria']}")
        lines.append("Answer yes or no.")
    elif kind == "score":
        lines.append(f"QUESTION (rating): {question['instructions']}")
        lines.append("Levels:")
        lines += [f"{i}. {level}" for i, level in enumerate(question["criteria"])]
        lines.append("Answer with the number of one level.")
    else:
        lines.append(f"QUESTION (choice): {question['instructions']}")
        lines.append("Options:")
        lines += [f"{LETTERS[i]}. {name}: {desc}" for i, (name, desc) in enumerate(question["criteria"].items())]
        lines.append("Answer with the letter of one option.")
    return "\n".join(lines)


def messages(state, question: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"STATE:\n{state_text(state)}\n\n{question_text(question)}"},
    ]


def target(question: dict, answer: dict, floor: float = 0.01) -> list[float]:
    """The teacher's answer (mu's Answer object) as a distribution over labels(question), never exactly 0 or 1."""
    kind = question["type"]
    if kind == "boolean":
        p = answer["probability"]
        dist = [p, 1 - p]
    elif kind == "score":
        n = len(question["criteria"])
        s = min(max(answer["score"], 0), n - 1)
        lo = int(s)
        dist = [0.0] * n
        dist[lo] = 1 - (s - lo)
        if lo + 1 < n:
            dist[lo + 1] += s - lo
    else:
        probs = answer.get("probabilities") or {answer["choice"]: 1.0}
        dist = [float(probs.get(name, 0.0)) for name in question["criteria"]]
    dist = [max(x, floor) for x in dist]
    total = sum(dist)
    return [x / total for x in dist]


def to_system_one(question: dict, probs: list[float]) -> dict:
    """The student's label probabilities (renormalised over the labels) as a System One answer."""
    total = sum(probs) or 1.0
    probs = [p / total for p in probs]
    kind = question["type"]
    if kind == "boolean":
        return {"noul": probs[0]}
    if kind == "score":
        return {"score": sum(i * p for i, p in enumerate(probs))}
    names = list(question["criteria"])
    best = max(range(len(names)), key=lambda i: probs[i])
    return {"choice": names[best], "probabilities": dict(zip(names, probs))}


def from_wire(question: dict) -> dict:
    """System One calls a yes/no question `noul`; the kernel calls it `boolean`."""
    return {**question, "type": "boolean"} if question.get("type") == "noul" else question
