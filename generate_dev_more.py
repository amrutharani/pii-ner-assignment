import json
import random

# Load files
def load(path):
    out = []
    with open(path, "r") as f:
        for l in f:
            if l.strip():
                out.append(json.loads(l))
    return out

train = load("data/train.jsonl")
test = load("data/test.jsonl")
stress = load("data/stress.jsonl")

# Sample
new_dev = []
new_dev.extend(random.sample(train, 100))
new_dev.extend(random.sample(test, 40))
new_dev.extend(random.sample(stress, 35))

print("Created dev size:", len(new_dev))  # should be 175

# Write new dev
with open("data/dev.jsonl", "w") as f:
    for ex in new_dev:
        f.write(json.dumps(ex) + "\n")

print("Wrote dev.jsonl successfully!")
