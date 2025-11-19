from collections import defaultdict

def zoek_eulerpad(graaf):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    # Calculate in-degrees and out-degrees
    for node in graaf:
        for neighbor in graaf[node]:
            out_degree[node] += 1
            in_degree[neighbor] += 1

    # Find start and end nodes for Eulerian path
    start, end = None, None
    for node in set(in_degree.keys()).union(out_degree.keys()):
        if out_degree[node] > in_degree[node]:
            if start is not None:
                raise ValueError("De graaf heeft meer dan één mogelijke startknoop.")
            start = node
        elif in_degree[node] > out_degree[node]:
            if end is not None:
                raise ValueError("De graaf heeft meer dan één mogelijk eindknoop.")
            end = node

    # If no start node is found, pick any node with outgoing edges
    if start is None:
        start = next(iter(graaf))

    # Eulerian path traversal (Hierholzer's algorithm)
    path = []
    stack = [start]
    while stack:
        current = stack[-1]
        if graaf[current]:
            next_node = graaf[current].pop()
            stack.append(next_node)
        else:
            path.append(stack.pop())

    return path[::-1]  # Reverse the path to get the correct order