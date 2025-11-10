VANILLA_PROMPTS = {
    "en": {
        "system": (
            "You are a classifier. Decide whether the given claim requires a citation. "
            "Use 1 if the claim needs a citation. Use 0 if the claim does not need a citation.\n"
            "Return only JSON in the format: {\"label\": 0} or {\"label\": 1}. "
            "No explanations or extra text."
        ),
        "user": "Claim: {claim}",
        "assistant": '{{"label": {label}}}'
    },
    "nl": {
        "system": (
            "Je bent een classifier. Bepaal of de gegeven bewering een bronvermelding nodig heeft. "
            "Gebruik 1 als de bewering een bronvermelding nodig heeft. Gebruik 0 als dat niet nodig is.\n"
            "Geef uitsluitend JSON terug in het formaat: {\"label\": 0} of {\"label\": 1}. "
            "Geen uitleg of extra tekst."
        ),
        "user": "Bewering: {claim}",
        "assistant": '{{"label": {label}}}'
    }
}
    
EXPLANATION_PROMPTS = {
    "en": {
        "system": (
            "You are a classifier. Decide whether the given claim requires a citation. "
            "Use 1 if the claim needs a citation. Use 0 if it does not.\n"
            "Return only JSON in the format: {\"label\": 0, \"explanation\": \"...\"}. "
            "The explanation must be short (1-3 sentences). No extra text."
        ),
        "user": "Claim: {claim}",
        "assistant": '{{"label": {label}, "explanation": "{explanation}"}}'
    },
    "nl": {
        "system": (
            "Je bent een classifier. Bepaal of de gegeven claim een bronvermelding nodig heeft. "
            "Gebruik 1 als de claim een bron nodig heeft. Gebruik 0 als dat niet nodig is.\n"
            "Geef uitsluitend JSON terug in het formaat: {\"label\": 0, \"explanation\": \"...\"}. "
            "De uitleg moet kort zijn (1 zin). Geen extra tekst."
        ),
        "user": "Claim: {claim}",
        "assistant": '{{"label": {label}, "explanation": "{explanation}"}}'
    }
}
