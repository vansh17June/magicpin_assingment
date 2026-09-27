"""Focused contract tests for the Vera challenge composer and API logic."""

from __future__ import annotations

import copy
import json
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import bot
from dataset import generate_dataset


ROOT = Path(__file__).resolve().parent


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class VeraBotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        categories, merchant_seeds, customer_seeds, trigger_seeds = generate_dataset.load_seeds(ROOT / "dataset")
        rng = random.Random(generate_dataset.SEED)
        merchants = generate_dataset.expand_merchants(merchant_seeds, rng)
        customers = generate_dataset.expand_customers(customer_seeds, merchants, rng)
        triggers = generate_dataset.expand_triggers(trigger_seeds, merchants, customers, rng)
        cls.categories = categories
        cls.merchants = {item["merchant_id"]: item for item in merchants}
        cls.customers = {item["customer_id"]: item for item in customers}
        cls.triggers = {item["id"]: item for item in triggers}

    def test_all_seed_triggers_produce_a_valid_composition(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            for trigger in self.triggers.values():
                merchant = self.merchants[trigger["merchant_id"]]
                category = self.categories[merchant["category_slug"]]
                customer = self.customers.get(trigger.get("customer_id"))
                composed = bot.compose(category, merchant, trigger, customer, use_ai=False)
                self.assertTrue(composed["body"], trigger["kind"])
                self.assertIn(composed["send_as"], {"vera", "merchant_on_behalf"})
                self.assertIn(composed["cta"], {"yes_stop", "open_ended", "none"})
                self.assertTrue(composed["suppression_key"])
                self.assertTrue(composed["rationale"])

    def test_customer_outreach_requires_scope_and_reminder_preference(self) -> None:
        trigger = self.triggers["trg_003_recall_due_priya"]
        merchant = self.merchants[trigger["merchant_id"]]
        customer = copy.deepcopy(self.customers[trigger["customer_id"]])
        category = self.categories[merchant["category_slug"]]
        permitted = bot.compose(category, merchant, trigger, customer, use_ai=False)
        self.assertEqual(permitted["send_as"], "merchant_on_behalf")
        customer["preferences"]["reminder_opt_in"] = False
        blocked = bot.compose(category, merchant, trigger, customer, use_ai=False)
        self.assertEqual(blocked["send_as"], "vera")
        self.assertEqual(blocked["cta"], "none")

    def test_customer_refill_uses_trigger_medicines_and_cta(self) -> None:
        trigger = self.triggers["trg_019_chronic_refill_grandfather"]
        merchant = self.merchants[trigger["merchant_id"]]
        customer = self.customers[trigger["customer_id"]]
        message = bot.compose(
            self.categories[merchant["category_slug"]],
            merchant,
            trigger,
            customer,
            use_ai=False,
        )
        self.assertIn("metformin", message["body"])
        self.assertIn("Reply YES", message["body"])
        self.assertIn("STOP", message["body"])

    def test_gemini_rewrites_only_body_and_rejects_number_changes(self) -> None:
        category = self.categories["dentists"]
        merchant = self.merchants["m_001_drmeera_dentist_delhi"]
        trigger = self.triggers["trg_001_research_digest_dentists"]
        baseline = bot.compose(category, merchant, trigger, use_ai=False)

        def response_for(transform):
            def fake_urlopen(request, timeout):
                self.assertIn("generateContent", request.full_url)
                self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
                sent = json.loads(request.data)
                self.assertEqual(sent["generationConfig"]["temperature"], 0)
                prompt = json.loads(sent["contents"][0]["parts"][0]["text"])
                body = transform(prompt["draft"])
                return FakeResponse({
                    "candidates": [{
                        "content": {"parts": [{"text": json.dumps({"body": body})}]}
                    }]
                })
            return fake_urlopen

        rewrite = lambda body: body.replace("For your Dentists:", "For your dental practice:")
        with patch.dict(os.environ, {"VERA_GEMINI_API_KEY": "test-key"}, clear=True):
            with patch("urllib.request.urlopen", response_for(rewrite)):
                ai_message = bot.compose(category, merchant, trigger)
            self.assertNotEqual(ai_message["body"], baseline["body"])
            self.assertEqual(ai_message["cta"], baseline["cta"])
            with patch("urllib.request.urlopen", response_for(lambda body: body.replace("38%", "39%"))):
                with self.assertRaisesRegex(RuntimeError, "number"):
                    bot.compose(category, merchant, trigger)

    def test_tick_uses_approved_initial_message_template_and_suppresses_repeat(self) -> None:
        category = self.categories["dentists"]
        merchant = self.merchants["m_001_drmeera_dentist_delhi"]
        trigger = copy.deepcopy(self.triggers["trg_001_research_digest_dentists"])
        trigger.pop("expires_at", None)
        trigger_id = "template-contract-test"
        with patch.dict(os.environ, {}, clear=True):
            with bot._lock:
                bot._contexts.clear()
                bot._sent_suppressions.clear()
                bot._conversations.clear()
                bot._contexts[("category", "dentists")] = {"version": 1, "payload": category}
                bot._contexts[("merchant", merchant["merchant_id"])] = {"version": 1, "payload": merchant}
                bot._contexts[("trigger", trigger_id)] = {"version": 1, "payload": trigger}
            first = bot._make_actions("2026-04-27T00:00:00Z", [trigger_id])
            second = bot._make_actions("2026-04-27T00:00:00Z", [trigger_id])
            self.assertEqual(len(first), 1)
            self.assertEqual(second, [])
            action = first[0]
            self.assertEqual(action["template_name"], "vera_merchant_message_v1")
            self.assertEqual(action["template_body"], "{{1}}")
            self.assertEqual(action["template_params"], [action["body"]])
        with bot._lock:
            bot._contexts.clear()
            bot._sent_suppressions.clear()
            bot._conversations.clear()

    def test_reply_routes_commitment_and_opt_out(self) -> None:
        with bot._lock:
            bot._conversations.clear()
            bot._merchant_replies.clear()
        positive = bot._reply({
            "conversation_id": "test-yes",
            "merchant_id": "merchant-test",
            "from_role": "merchant",
            "message": "Ok lets do it. Whats next?",
        })
        self.assertEqual(positive["action"], "send")
        self.assertIn("next step", positive["rationale"])
        negative = bot._reply({
            "conversation_id": "test-stop",
            "merchant_id": "merchant-test",
            "from_role": "merchant",
            "message": "Stop messaging me.",
        })
        self.assertEqual(negative["action"], "end")
        unrelated = bot._reply({
            "conversation_id": "test-gst",
            "merchant_id": "merchant-test",
            "from_role": "merchant",
            "message": "Can you help me file my GST?",
        })
        self.assertEqual(unrelated["action"], "end")

    def test_trigger_fact_additions_are_used(self) -> None:
        merchant = self.merchants["m_009_apollo_pharmacy_jaipur"]
        category = self.categories[merchant["category_slug"]]
        supply = self.triggers["trg_018_supply_atorvastatin_recall"]
        supply_message = bot.compose(category, merchant, supply, use_ai=False)
        self.assertIn("AT2024-1102", supply_message["body"])
        self.assertIn("official notice", supply_message["body"])
        cde_merchant = self.merchants["m_001_drmeera_dentist_delhi"]
        cde = {
            "kind": "cde_opportunity",
            "payload": {
                "digest_item_id": "d_2026W17_ida_webinar",
                "credits": 2,
                "fee": "free_for_members",
            },
            "suppression_key": "cde-test",
        }
        cde_message = bot.compose(
            self.categories["dentists"], cde_merchant, cde, use_ai=False
        )
        self.assertIn("2 CDE credits", cde_message["body"])
        self.assertIn("free for members", cde_message["body"])


if __name__ == "__main__":
    unittest.main()
