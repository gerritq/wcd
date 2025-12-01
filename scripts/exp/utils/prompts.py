VANILLA_PROMPTS = {
    "system": (
        "You are a multilingual classifier. "
        "Decide whether the given {lang} claim requires a citation. "
        "Use 1 if the claim needs a citation. Use 0 if the claim does not need a citation.\n\n"
        "Return only JSON in the format: {{\"label\": 0}} or {{\"label\": 1}}. "
        "No explanations or extra text."
    ),
    "user_claim": "Claim: {claim}",
    "user_context": (
        "Section: {section}\n"
        "Previous Sentence: {previous_sentence}\n"
        "Claim: {claim}\n"
        "Subsequent Sentence: {subsequent_sentence}"
    ),
    "assistant": '{{"label": {label}}}'
}

RATIONALE_PROMPTS = {
    
    "system": (
        "You are a multilingual classifier. "
        "Read the {lang} claim and provide a short rationale explaining why it would require a citation or not. "
        "Return only JSON in the format: {{\"rationale\": \"...\"}}. "
        "No extra text."
    ),
    "user_claim": "Claim: {claim}",
    "user_context": (
        "Section: {section}\n"
        "Previous Sentence: {previous_sentence}\n"
        "Claim: {claim}\n"
        "Subsequent Sentence: {subsequent_sentence}"
    ),
    "assistant": "{{\"rationale\": \"{rationale}\"}}",
}
RATIONALE_LABEL_PROMPTS = {
    "system": (
        "You are a multilingual classifier. "
        "Decide whether the given {lang} claim requires a citation. "
        "Use 1 if the claim requires a citation and 0 if it does not. "
        "Provide a concise rationale in English in no more than one sentences. "
        "Return only JSON in the format: {{\"rationale\": \"...\", \"label\": 0/1}}. "
        "No extra text."
    ),
    "user_claim": "Claim: {claim}",
    "user_context": (
        "Section: {section}\n"
        "Previous Sentence: {previous_sentence}\n"
        "Claim: {claim}\n"
        "Subsequent Sentence: {subsequent_sentence}"
    ),
    "assistant": "{{\"rationale\": \"{rationale}\", \"label\": {label}}}",
}
