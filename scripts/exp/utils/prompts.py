VANILLA_PROMPTS = {
    "en": {
        "system": (
            "You are a multilingual classifier. "
            "Decide whether the given English claim requires a citation. "
            "Use 1 if the claim needs a citation. Use 0 if the claim does not need a citation.\n\n"
            "Return only JSON in the format: {\"label\": 0} or {\"label\": 1}. "
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
    },
    "nl": {
        "system": (
            "You are a multilingual classifier. "
            "Decide whether the given Dutch claim requires a citation. "
            "Use 1 if the claim needs a citation. Use 0 if the claim does not need a citation.\n\n"
            "Return only JSON in the format: {\"label\": 0} or {\"label\": 1}. "
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
    },
    "it": {
        "system": (
            "You are a multilingual classifier. "
            "Decide whether the given Italian claim requires a citation. "
            "Use 1 if the claim needs a citation. Use 0 if the claim does not need a citation.\n\n"
            "Return only JSON in the format: {\"label\": 0} or {\"label\": 1}. "
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
    },
    "nl_nl": {
        "system": (
            "Je bent een classifier. Bepaal of de gegeven bewering een bronvermelding nodig heeft. "
            "Gebruik 1 als de bewering een bronvermelding nodig heeft. Gebruik 0 als dat niet nodig is.\n"
            "Geef uitsluitend JSON terug in het formaat: {\"label\": 0} of {\"label\": 1}. "
            "Geen uitleg of extra tekst."
        ),
        "user_claim": "Bewering: {claim}",
        "user_context": (
            "Sectie: {section}\n"
            "Vorige zin: {previous_sentence}\n"
            "Bewering: {claim}\n"
            "Volgende zin: {subsequent_sentence}"
        ),
        "assistant": '{{"label": {label}}}'
    }
}

RATIONALE_PROMPTS = {
    "en": {
        "system": (
            "You are a classifier. Read the claim and provide a short rationale explaining "
            "why it would require a citation or not. "
            "Return only JSON in the format: {\"rationale\": \"...\"}. "
            "No explanations or extra text."
        )
    },
    "nl": {
        "system": (
            "You are a multilingual classifier. "
            "Read the Dutch claim and provide a short rationale explaining why it would require a citation or not. "
            "Return only JSON in the format: {\"rationale\": \"...\"}. "
            "No explanations or extra text."
        ),
        "user_claim": "Claim: {claim}",
        "user_context": (
            "Section: {section}\n"
            "Previous Sentence: {previous_sentence}\n"
            "Claim: {claim}\n"
            "Subsequent Sentence: {subsequent_sentence}"
        ),
        "assistant": "{{\"rationale\": \"{rationale}\"}}",
    },
    # "nl": {
    #     "system": (
    #         "Je bent een classifier. Lees de bewering en geef een korte redenering die uitlegt "
    #         "of deze wel of geen bronvermelding nodig heeft. "
    #         "Geef uitsluitend JSON terug in het formaat: {\"rationale\": \"...\"}. "
    #         "Geen verdere uitleg of extra tekst."
    #     ),
    #     "user_claim": "Bewering: {claim}",
    #     "user_context": (
    #         "Sectie: {section}\n"
    #         "Vorige zin: {previous_sentence}\n"
    #         "Bewering: {claim}\n"
    #         "Volgende zin: {subsequent_sentence}"
    #     ),
    #     "assistant": "{{\"rationale\": \"{rationale}\"}}",
    # },
}

RATIONALE_LABEL_PROMPTS = {
    # "nl": {
    #     "system": (
    #         "Je bent een classifier. Bepaal of de gegeven bewering een bronvermelding nodig heeft. "
    #         "Gebruik 1 als de bewering een bronvermelding nodig heeft en 0 als dat niet zo is. "
    #         "Geef een beknopte redenering in maximaal drie zinnen met stapsgewijze uitleg. "
    #         "Geef uitsluitend JSON terug in het formaat: {\"rationale\": \"...\", \"label\": 0/1}. "
    #         "Geen verdere uitleg of extra tekst."
    #     ),
    #     "user_claim": "Bewering: {claim}",
    #     "user_context": (
    #         "Sectie: {section}\n"
    #         "Vorige zin: {previous_sentence}\n"
    #         "Bewering: {claim}\n"
    #         "Volgende zin: {subsequent_sentence}"
    #     ),
    #     "assistant": "{{\"rationale\": \"{rationale}\", \"label\": {label}}}",
    # },
    "nl": {
        "system": (
            "You are a multilingual classifier. "
            "Decide whether the given Dutch claim requires a citation. "
            "Use 1 if the claim requires a citation and 0 if it does not. "
            "Provide a concise rationale in English in no more than one sentences. "
            "Return only JSON in the format: {\"rationale\": \"...\", \"label\": 0/1}. "
            "No explanations or extra text."
        ),
        "user_claim": "Claim (Dutch): {claim}",
        "user_context": (
            "Section: {section}\n"
            "Previous Sentence: {previous_sentence}\n"
            "Claim: {claim}\n"
            "Subsequent Sentence: {subsequent_sentence}"
        ),
        "assistant": "{{\"rationale\": \"{rationale}\", \"label\": {label}}}",
    },
}
