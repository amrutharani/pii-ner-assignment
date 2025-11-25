import json
from typing import List, Dict, Any
from torch.utils.data import Dataset


class PIIDataset(Dataset):
    def __init__(self, path: str, tokenizer, label_list: List[str],
                 max_length: int = 256, is_train: bool = True):
        self.items = []
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.max_length = max_length

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj["text"]
                entities = obj.get("entities", [])

                # char-level tagging
                char_tags = ["O"] * len(text)
                for e in entities:
                    s, e_idx, lab = e["start"], e["end"], e["label"]
                    char_tags[s] = f"B-{lab}"
                    for c in range(s + 1, e_idx):
                        char_tags[c] = f"I-{lab}"

                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=self.max_length,
                    add_special_tokens=True,
                )

                input_ids = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                offsets = enc["offset_mapping"]

                # fixed logic: use full char span
                bio_tags = []
                for (start, end) in offsets:
                    if start == end:
                        bio_tags.append("O")
                        continue

                    # If the token overlaps entity chars
                    token_tag = "O"
                    if start < len(char_tags) and end <= len(char_tags):
                        span_tags = char_tags[start:end]
                        # any entity?
                        ent_tags = [t for t in span_tags if t != "O"]
                        if ent_tags:
                            # ensure B-/I- consistency
                            first = ent_tags[0]
                            if first.startswith("I-"):
                                token_tag = first  # keep I
                            else:
                                token_tag = first  # B
                    bio_tags.append(token_tag)

                label_ids = [self.label2id.get(t, self.label2id["O"]) for t in bio_tags]

                self.items.append({
                    "id": obj["id"],
                    "text": text,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_ids,
                    "offset_mapping": offsets,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch, pad_token_id: int, label_pad_id: int = -100):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(arr, pad_value):
        return arr + [pad_value] * (max_len - len(arr))

    input_ids = [pad(x["input_ids"], pad_token_id) for x in batch]
    attention_mask = [pad(x["attention_mask"], 0) for x in batch]
    labels = [pad(x["labels"], label_pad_id) for x in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "offset_mapping": [x["offset_mapping"] for x in batch],
    }
