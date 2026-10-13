import json
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVIDENCE_FILE = (
    PROJECT_ROOT
    / "reports"
    / "credit_risk"
    / "ai_copilot"
    / "evidence_packet.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "credit_risk"
    / "ai_copilot"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "openai/gpt-oss-20b"


# --------------------------------------------------
# 2. Load and validate evidence
# --------------------------------------------------

def load_evidence():
    """Load the verified model score and SHAP evidence."""

    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"Evidence packet not found: {EVIDENCE_FILE}\n"
            "Generate evidence_packet.json in your "
            "SHAP notebook first."
        )

    with open(EVIDENCE_FILE, "r", encoding="utf-8") as file:
        evidence = json.load(file)

    required_fields = {
        "model",
        "model_score",
        "feature_evidence",
        "limitations",
    }

    missing_fields = required_fields - set(evidence)

    if missing_fields:
        raise ValueError(
            f"Evidence packet missing fields: {missing_fields}"
        )

    if not isinstance(evidence["feature_evidence"], list):
        raise ValueError("feature_evidence must be a list.")

    return evidence


# --------------------------------------------------
# 3. Generate the AI explanation
# --------------------------------------------------

def generate_risk_explanation():
    """Explain verified model and SHAP evidence for an analyst."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Check the .env file in the NovaBank root folder."
        )

    evidence = load_evidence()

# Give the LLM only the qualitative evidence it needs.
# Python will display exact scores and numerical values separately.
    llm_evidence = {
        "model": evidence["model"],
        "purpose": (
            "Explain which verified features push the model score "
            "higher or lower. Do not make a lending decision."
    ),
        "feature_evidence": [
        {
                "feature": item["feature"],
                "direction": item["direction"],
        }
            for item in evidence["feature_evidence"]
    ],
        "limitations": [
            "The data represents historical US lending.",
            "SHAP contributions describe model behaviour, not causation.",
            "The model score has not been adequately calibrated.",
            "The results do not establish performance on all future applicants.",
            "A human analyst must review the evidence."
    ]
}
    # Allowed feature identifiers from the verified evidence.
    allowed_feature_names = sorted({
        item["feature"]
        for item in llm_evidence["feature_evidence"]
})

    print("Allowed feature identifiers:", allowed_feature_names)

    client = Groq(api_key=api_key)

    system_prompt = """
You are NovaBank's AI Risk Explanation Copilot.

Explain the supplied credit-risk model evidence to a human analyst.

Strict rules:
1. Use only facts and values present in the evidence packet.
2. Never invent borrower information, numbers, or model results.
3. A positive SHAP contribution pushes the model score higher.
4. A negative SHAP contribution pushes the model score lower.
5. SHAP contributions do not prove causation.
6. Call the score a model-predicted score, not a validated,
   calibrated individual default probability.
7. Do not approve or reject a loan.
8. Do not accuse anyone of fraud or misconduct.
9. Clearly communicate limitations and the need for human review.
10. Return a JSON object with exactly these keys:
    - summary: a concise evidence-based explanation
    - higher_score_contributors: list of supported feature names
    - lower_score_contributors: list of supported feature names
    - analyst_checks: list of verification questions
    - caveats: list of limitations
11. Do not include digits, dates, percentages, currency amounts,
    numerical scores, or numerical quantities in free-text fields.
12. Do not repeat exact numerical feature values or SHAP contributions.
13. Explain features in plain English. For example, describe
    "inq_last_6mths" as "recent credit inquiries".
14. Python will display exact figures separately from the explanation.
15. Do not introduce facts that are absent from the supplied evidence.
"""
    response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": json.dumps(
            llm_evidence,
            allow_nan=False,
    ),
},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "risk_explanation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string"
                    },
                    "higher_score_contributors": {
                        "type": "array",
                        "items": {"type": "string",
                        'enum':allowed_feature_names
                        }

                    },
                    "lower_score_contributors": {
                        "type": "array",
                        "items": {"type": "string",
                        'enum':allowed_feature_names
                        }
                    },
                    "analyst_checks": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "caveats": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                },
                "required": [
                    "summary",
                    "higher_score_contributors",
                    "lower_score_contributors",
                    "analyst_checks",
                    "caveats",
                ],
                "additionalProperties": False,
            },
        },
    },
    reasoning_effort="low",
    reasoning_format="hidden",
    max_completion_tokens=2000,
)

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "The AI response was not valid JSON."
        ) from exc

    # Check the response structure.
    required_output_fields = {
        "summary",
        "higher_score_contributors",
        "lower_score_contributors",
        "analyst_checks",
        "caveats",
    }

    missing_output_fields = (
        required_output_fields - set(result)
    )

    if missing_output_fields:
        raise ValueError(
            f"AI response missing fields: {missing_output_fields}"
        )

    import re

    # Create a lookup table from the verified SHAP evidence.
    evidence_by_feature = {
        item["feature"]: item
        for item in evidence["feature_evidence"]
}

    allowed_features = set(evidence_by_feature)

# Ensure contributor lists contain only supported feature names.
    for key in [
        "higher_score_contributors",
        "lower_score_contributors",
]:
        contributors = result[key]

        if not isinstance(contributors, list):
            raise ValueError(f"{key} must be a list.")

        if len(contributors) != len(set(contributors)):
            raise ValueError(f"{key} contains duplicate features.")

        unsupported = set(contributors) - allowed_features

        if unsupported:
            raise ValueError(
                f"Unsupported feature names in {key}: {unsupported}"
        )

    # Ensure contribution directions match the actual SHAP values.
    for feature in result["higher_score_contributors"]:
        if evidence_by_feature[feature]["shap_contribution"] <= 0:
            raise ValueError(
                f"{feature} does not have a positive SHAP contribution."
        )

    for feature in result["lower_score_contributors"]:
        if evidence_by_feature[feature]["shap_contribution"] >= 0:
            raise ValueError(
                f"{feature} does not have a negative SHAP contribution."
        )

# Free-text fields must not introduce numerical claims.
    free_text = [result["summary"]]

    free_text.extend(result["analyst_checks"])
    free_text.extend(result["caveats"])

    for text in free_text:
        if not isinstance(text, str):
            raise ValueError("AI free-text fields must contain strings.")

        if re.search(r"\d", text):
            raise ValueError(
                "AI explanation contains a numerical claim. "
                "Exact values must be rendered from the evidence packet."
        )
    # --------------------------------------------------
# Make the summary evidence-grounded
# --------------------------------------------------

    FEATURE_LABELS = {
        "annual_inc": "annual income",
        "emp_length": "employment length",
        "inq_last_6mths": "recent credit inquiries",
        "mths_since_last_delinq": "time since last recorded delinquency",
        "pub_rec": "public record count",
        "purpose": "loan purpose",
        "revol_util": "revolving-credit utilization",
        "verification_status": "income verification status",
}

    higher_features = result["higher_score_contributors"]
    lower_features = result["lower_score_contributors"]

    higher_labels = [
        FEATURE_LABELS.get(feature, feature)
        for feature in higher_features
]

    lower_labels = [
        FEATURE_LABELS.get(feature, feature)
        for feature in lower_features
]


    def format_feature_list(features):
        if not features:
            return "none identified"

        if len(features) == 1:
            return features[0]

        if len(features) == 2:
            return f"{features[0]} and {features[1]}"

        return (
            ", ".join(features[:-1])
            + ", and "
            + features[-1]
    )


    higher_text = format_feature_list(higher_labels)
    lower_text = format_feature_list(lower_labels)

# Generate the factual summary deterministically.
# Do not let the LLM infer whether a higher raw value is risky.
    result["summary"] = (
        "For this specific case, the model's positive SHAP "
        f"contributions were associated with {higher_text}. "
        f"The negative contribution was associated with {lower_text}. "
        "These contributions describe how the recorded feature values "
        "and encoded categories influenced this model's score relative "
        "to its reference value. A positive contribution does not mean "
        "that increasing the feature would necessarily increase risk. "
        "These are model attributions, not causal explanations."
)


    print("AI explanation passed structural and evidence-direction checks.")
    return result



# --------------------------------------------------
# 4. Run the copilot and save its output
# --------------------------------------------------

def main():
    result = generate_risk_explanation()

    output_file = OUTPUT_DIR / "risk_explanation.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\n===== NOVABANK AI RISK EXPLANATION =====\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved explanation to: {output_file}")


if __name__ == "__main__":
    main()
