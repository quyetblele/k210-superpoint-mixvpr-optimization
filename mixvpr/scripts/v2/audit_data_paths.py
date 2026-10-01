from pathlib import Path
from collections import Counter, defaultdict
import json

DATA = Path("/mnt/d/k210_official_ab/data.json")

def main():
    data = json.loads(DATA.read_text())
    by_domain = defaultdict(Counter)
    missing_examples = defaultdict(list)
    for row in data["rows"]:
        domain = row["domain"]
        exists = Path(row["path"]).is_file()
        by_domain[domain]["total"] += 1
        by_domain[domain]["exists" if exists else "missing"] += 1
        if not exists and len(missing_examples[domain]) < 5:
            missing_examples[domain].append(row["path"])
    result = {
        "by_domain": {k: dict(v) for k, v in by_domain.items()},
        "missing_examples": dict(missing_examples),
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
