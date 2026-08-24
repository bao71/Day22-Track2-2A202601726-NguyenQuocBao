"""
Step 4 - Guardrails AI Validators.

Implements two custom validators:
- PIIDetector: redacts email, phone, SSN, and credit-card-like numbers.
- JSONFormatter: validates and repairs common malformed JSON from LLM output.
"""
import argparse
import json
import os
import re

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["GUARDRAILS_DISABLE_TELEMETRY"] = "true"

from guardrails import Guard
from guardrails.validators import FailResult, PassResult, Validator, register_validator

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "PHONE": r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, redacted_text)
            if matches:
                found_pii.extend((pii_type, match) for match in matches)
                redacted_text = re.sub(pattern, f"[{pii_type}_REDACTED]", redacted_text)

        if found_pii:
            print(f"  Redacted {len(found_pii)} PII item(s): {[item[0] for item in found_pii]}")
            return FailResult(
                error_message=f"Detected PII: {', '.join(sorted({item[0] for item in found_pii}))}",
                fix_value=redacted_text,
            )

        return PassResult(value_override=value)


@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        text = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', text)
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    def validate(self, value: str, metadata: dict):
        try:
            parsed = json.loads(value)
            return PassResult(value_override=json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            print("  JSON repaired successfully")
            return FailResult(
                error_message="JSON was repaired",
                fix_value=json.dumps(parsed, indent=2, ensure_ascii=False),
            )
        except json.JSONDecodeError as exc:
            fallback = json.dumps(
                {
                    "error": "Could not parse JSON",
                    "raw": value[:200],
                },
                ensure_ascii=False,
                indent=2,
            )
            return FailResult(error_message=f"Could not repair JSON: {exc}", fix_value=fallback)


def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Email", "Contact John at john.doe@example.com for details."),
        ("Phone", "Call our support line at (555) 867-5309."),
        ("SSN", "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII", "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean", "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "FIXED" if result.validated_output != text else "PASS"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


def demo_json_guard():
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))
    test_cases = [
        ("Valid JSON", '{"name": "Alice", "age": 30}'),
        ("Markdown fences", '```json\n{"name": "Bob", "score": 95}\n```'),
        ("Single quotes", "{'name': 'Charlie', 'active': true}"),
        ("Trailing comma", '{"items": ["a", "b",], "total": 2,}'),
        ("Truly invalid", "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "PASS" if result.validation_passed else "FIXED/FALLBACK"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:80]}")
        print(f"  Output: {result.validated_output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=["all", "pii", "json"], default="all")
    args = parser.parse_args()

    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    if args.demo in ("all", "pii"):
        demo_pii_guard()
    if args.demo in ("all", "json"):
        demo_json_guard()

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()
