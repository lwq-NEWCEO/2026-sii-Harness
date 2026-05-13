import json
import collections
from pathlib import Path

from solution import MyHarness
from llm_client import call_llm, count_tokens, count_messages_tokens


MAX_PROMPT_TOKENS = 2048


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    train = load_jsonl("data/train_dev.jsonl")
    dev = load_jsonl("data/test_dev.jsonl")

    h = MyHarness(call_llm, count_tokens, count_messages_tokens, MAX_PROMPT_TOKENS)

    for row in train:
        h.update(row["text"], row["label"])

    errors = []
    confusion = collections.Counter()
    gold_total = collections.Counter()
    gold_error = collections.Counter()
    retrieval_coverage = collections.Counter()

    for i, row in enumerate(dev):
        text = row["text"]
        gold = row["label"]

        pred = h.predict(text)
        dbg = getattr(h, "_last_debug", {})

        top10 = dbg.get("top10", [])
        selected = dbg.get("selected", [])

        top_labels = [lab for _sc, lab in top10]
        selected_labels = [x["label"] for x in selected]

        gold_total[gold] += 1

        if gold in top_labels[:1]:
            retrieval_coverage["gold_in_top1"] += 1
        if gold in top_labels[:3]:
            retrieval_coverage["gold_in_top3"] += 1
        if gold in top_labels[:5]:
            retrieval_coverage["gold_in_top5"] += 1
        if gold in top_labels[:10]:
            retrieval_coverage["gold_in_top10"] += 1
        if gold in selected_labels:
            retrieval_coverage["gold_in_selected"] += 1

        if pred != gold:
            confusion[(gold, pred)] += 1
            gold_error[gold] += 1

            err = {
                "idx": i,
                "text": text,
                "gold": gold,
                "pred": pred,
                "raw_response": dbg.get("response", ""),
                "fallback": dbg.get("fallback", ""),
                "top10": top10,
                "selected": selected,
                "gold_in_top10": gold in top_labels,
                "gold_in_selected": gold in selected_labels,
            }
            errors.append(err)

    acc = 1.0 - len(errors) / max(1, len(dev))

    print("=" * 60)
    print(f"Accuracy: {acc:.4%}")
    print(f"Errors: {len(errors)} / {len(dev)}")
    print("=" * 60)

    print("\nRetrieval coverage over all dev samples:")
    for k in ["gold_in_top1", "gold_in_top3", "gold_in_top5", "gold_in_top10", "gold_in_selected"]:
        print(f"{k}: {retrieval_coverage[k]} / {len(dev)} = {retrieval_coverage[k] / len(dev):.2%}")

    print("\nTop confusion pairs:")
    for (gold, pred), cnt in confusion.most_common(30):
        print(f"{cnt:3d} | gold={gold} -> pred={pred}")

    print("\nWorst gold labels by error count:")
    for gold, cnt in gold_error.most_common(30):
        total = gold_total[gold]
        print(f"{cnt:3d}/{total:3d} | {gold} | error_rate={cnt / total:.1%}")

    out_path = Path("debug_errors.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")

    print(f"\nSaved detailed errors to {out_path}")


if __name__ == "__main__":
    main()
