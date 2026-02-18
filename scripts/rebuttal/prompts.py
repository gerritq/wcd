MINIMAL_PROMPT = {
    "system": (
        "You are a multilingual classifier. "
        "Respond with {{\"label\": 1}} if the claim requires a citation. "
        "Respond with {{\"label\": 0}} otherwise."
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

INSTRUCT_PROMPT = {
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

VERBOSE_PROMPT = {
    "system": (
        "You are a multilingual Wikipedia citation classifier. "
        "You are provided with a {lang} claim and its context. "
        "Your task is to analyze the claim and the context to decide whether the claim needs a citation. "
        "On Wikipedia, there are various reasons why a claim may or may not require a citation. The reasons are listed below:\n\n"
        "# Reasons why citations are needed (Label 1)\n"
        "• Quotation – The statement is a direct quotation or close paraphrase of a source.\n"
        "• Statistics – The statement contains statistics or quantitative data.\n"
        "• Controversial – The statement makes surprising or potentially controversial claims.\n"
        "• Opinion – The statement expresses a person’s subjective opinion or belief.\n"
        "• Private Life – The statement contains claims about a person’s private life (e.g., date of birth, relationship status).\n"
        "• Scientific – The statement includes technical or scientific claims.\n"
        "• Historical – The statement makes general or historical claims that are not common knowledge.\n"
        "• Other (Needs Citation) – The statement requires a citation for other reasons (briefly describe why).\n\n"
        "# Reasons why citations are not needed (Label 0)\n"
        "• Common Knowledge – The statement contains well-known or widely established facts.\n"
        "• Plot – The statement describes the plot or characters of a book, film, or similar work that is the subject of the article.\n"
        "• Other (No Citation Needed) – The statement does not require a citation for other reasons (briefly describe why).\n\n"
        "Based on these reasons, think step-by-step to decide in which category the claim falls. "
        "Return only JSON in the format: {{\"label\": 0}} or {{\"label\": 1}}. "
        "No extra text."
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
