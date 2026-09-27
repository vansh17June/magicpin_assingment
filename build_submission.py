"""Generate the 30-line challenge submission from the deterministic seed dataset."""

from __future__ import annotations

import json
import argparse
import random
import tempfile
from pathlib import Path

from bot import _ai_settings, compose
from dataset import generate_dataset


ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT / "dataset"
OUTPUT = ROOT / "submission.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the canonical Vera challenge submission")
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Use the configured Gemini/OpenAI provider for each message (may incur provider usage).",
    )
    args = parser.parse_args()
    if args.use_ai and _ai_settings() is None:
        parser.error("--use-ai requires VERA_GEMINI_API_KEY or VERA_OPENAI_API_KEY in the environment")
    categories, merchant_seeds, customer_seeds, trigger_seeds = generate_dataset.load_seeds(SEED_DIR)
    rng = random.Random(generate_dataset.SEED)
    merchants = generate_dataset.expand_merchants(merchant_seeds, rng)
    customers = generate_dataset.expand_customers(customer_seeds, merchants, rng)
    triggers = generate_dataset.expand_triggers(trigger_seeds, merchants, customers, rng)
    merchant_by_id = {item["merchant_id"]: item for item in merchants}
    customer_by_id = {item["customer_id"]: item for item in customers}
    trigger_by_id = {item["id"]: item for item in triggers}

    with tempfile.TemporaryDirectory(prefix="vera-submission-") as temp_dir:
        pair_path = Path(temp_dir)
        generate_dataset.write_test_pairs(pair_path, triggers, rng)
        pairs = json.loads((pair_path / "test_pairs.json").read_text(encoding="utf-8"))["pairs"]

    with OUTPUT.open("w", encoding="utf-8", newline="\n") as output:
        for pair in pairs:
            merchant = merchant_by_id[pair["merchant_id"]]
            trigger = trigger_by_id[pair["trigger_id"]]
            category = categories[merchant["category_slug"]]
            customer = customer_by_id.get(pair.get("customer_id"))
            composed = compose(category, merchant, trigger, customer, use_ai=args.use_ai)
            output.write(json.dumps({"test_id": pair["test_id"], **composed}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} composed messages to {OUTPUT.name}")


if __name__ == "__main__":
    main()
