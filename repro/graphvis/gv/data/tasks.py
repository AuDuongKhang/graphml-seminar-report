"""Sáu tác vụ đọc ảnh đồ thị của giai đoạn 1 (D_g) và khuôn prompt QA của giai đoạn 2 (D_v).

Mẫu câu lấy nguyên văn Phụ lục A của bài; đã sửa hai lỗi của GraphVis-code/generate_data.py
(dòng 54 thiếu dấu phẩy, dòng 60 viết sai "{node\"}"). Bậc được tính trên đúng tập bộ ba dùng để
đếm cạnh, nên số cạnh, bậc và ảnh nhất quán với nhau.
"""
import random
from collections import Counter

TASKS = ("node_description", "node_degree", "highest_degree", "node_count", "edge_count", "triple_listing")
OCR_TASKS = ("node_description", "triple_listing")
GRAPH_TASKS = ("node_degree", "highest_degree", "node_count", "edge_count")

PROMPTS = {
    "node_description": [
        "List all nodes of the graph shown in the image.",
        "Provide the names of all nodes displayed in the graph image.",
        "Can you name all the nodes shown in the graph image?",
        "Identify all the vertices in the diagram of the graph provided.",
        "Detail all the vertices from the graph depicted in the image.",
    ],
    "highest_degree": [
        "Name one of the node with the highest degree in the graph. And what is its degree?",
        "Identify one of the node that has the most connections in the graph and specify its degree.",
        "Can you tell me which node (name one) has the highest degree in this graph and what that degree is?",
        "Provide the name and degree of the node with the most connections in the graph.",
        "Which node in the graph has the greatest number of connections, and what is that total?",
    ],
    "node_degree": [
        'What is the degree of the node with the name "{node}"?',
        'What is the degree of the node labeled "{node}"?',
        'Can you tell me the degree of the node named "{node}"?',
        'What is the total number of connections that the node "{node}" has?',
        'How many connections does the node "{node}" have?',
    ],
    "node_count": [
        "How many nodes are there in the graph?",
        "What is the total number of nodes in the graph?",
        "Can you tell me how many nodes are in the graph?",
        "What is the total number of vertices in the graph?",
        "How many vertices are there in the graph?",
    ],
    "edge_count": [
        "How many edges are there in the graph?",
        "What is the total number of edges in the graph?",
        "Can you tell me how many edges are in the graph?",
        "What is the total number of connections in the graph?",
        "How many connections are there in the graph?",
    ],
    "triple_listing": [
        "List all the triples in the graph.",
        "Provide all the triples in the graph.",
        "Can you list all the triples in the graph?",
        "Detail all the triples in the graph.",
        "Enumerate all the triples in the graph.",
    ],
}

QA_PROMPT = ("The image represents a knowledge graph relevant to the question, which may or may not be useful. "
             "Question: {question}\n{options}\nAnswer with the option's letter from the given choices directly.")
TEXT_PROMPT = "Question: {question}\n{options}\nAnswer with the option's letter from the given choices directly."


def degrees(sg) -> Counter:
    deg = Counter({n: 0 for n in sg.nodes})
    for h, _, t in sg.triples:
        deg[h] += 1
        deg[t] += 1
    return deg


def format_triples(triples) -> str:
    return ", ".join(f"({h}, {r}, {t})" for h, r, t in triples)


def make_task_sample(task: str, sg, rng: random.Random) -> tuple:
    template = rng.choice(PROMPTS[task])
    if task == "node_description":
        return template, f"The image depicts the following nodes: {', '.join(sg.nodes)}"
    if task == "node_count":
        return template, f"There are {len(sg.nodes)} nodes in the graph."
    if task == "edge_count":
        return template, f"There are {len(sg.triples)} edges in the graph."
    if task == "triple_listing":
        return template, f"The triples in the graph are listed as: {format_triples(sg.triples)}"
    deg = degrees(sg)
    if task == "highest_degree":
        best = max(sg.nodes, key=lambda n: (deg[n], -sg.nodes.index(n)))
        return template, f'One node with the highest degree is "{best}" with a degree of {deg[best]}.'
    if task == "node_degree":
        node = rng.choice(sg.nodes)
        return template.replace("{node}", node), f'The degree of the node "{node}" is {deg[node]}.'
    raise ValueError(f"unknown task {task}")


def format_options(item) -> str:
    return "\n".join(f"{l}. {c}" for l, c in zip(item.labels, item.choices))
