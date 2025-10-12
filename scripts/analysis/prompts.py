PROMPTS = {
    "en": """Your task is to determine whether a claim needs a citation (label=1) or no citation (label=0).

Claim: {{claim}}

Output the label in the following format:
{"label": <label>}
""",

    "en_system": """Your task is to determine whether a claim needs a citation (label=1) or no citation (label=0).

Output the label in the following format:
{"label": <label>}
""",

    "pt": """Sua tarefa é determinar se uma afirmação precisa de uma citação (label=1) ou não precisa de uma citação (label=0).

Afirmação: {{claim}}

Apresente o rótulo no seguinte formato:
{"label": <label>}
""",

    "nl": """Je taak is om te bepalen of een bewering een bronvermelding nodig heeft (label=1) of geen bronvermelding nodig heeft (label=0).

Bewerking: {{claim}}

Geef het label weer in het volgende formaat:
{"label": <label>}
"""
}