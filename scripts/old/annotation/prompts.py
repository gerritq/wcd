"""
COT_SHORT_PROMPT = cot but one sentence onlu
COT_PROMPT = cot without sentence restriction
COT_WIKI_SHORT_PROMPT = ..
COT_WIKI_PROMPT = ..

Note: the one sentence version asks twice for a sinlge sentence, otherwise it is the same
 Search for single sentence in these prompts
"""

COT_SHORT_PROMPT = {
    "nl": (
                    "You are a multilingual Wikipedia citation reasoning assistant. "
                    "You are provided with a Dutch claim, its context, and a citation-needed label. "
                    "A label of 1 means the claim requires a citation. "
                    "A label of 0 means the claim does not require a citation.\n\n"
                    "Your task is to provide a concice, one-sentence, step-by-step rationale in English explaining why the label applies to the given claim. "
                    "Return only your answer as valid JSON in a json block in the format:\n"
                    "{{\"response\": \"your_rationale\"}}\n\n"
                    "Analyze the following claim in Dutch:\n"
                    "Section: {section}\n"
                    "Previous sentence: {previous_sentence}\n"
                    "Claim: {claim}\n"
                    "Subsequent sentence: {subsequent_sentence}\n"
                    "Label: {label}\n\n"
                    "Now think step by step and then provide a single-sentence answer as JSON only:"
    ),

    # this is the original prompt that we used for nl
    # "nl": (
    #     "Je bent een Wikipedia-assistent voor citatiebeoordeling. "
    #     "Je krijgt een bewering, de context ervan en een label dat aangeeft of er een bron nodig is. "
    #     "Een label van 1 betekent dat de bewering een bron nodig heeft. "
    #     "Een label van 0 betekent dat de bewering geen bron nodig heeft.\n\n"
    #     "Je taak is om een rationale te geven waarom het label van toepassing is op de gegeven bewering. "
    #     "Geef een beknopte uitleg met stapsgewijze redenering in maximaal één zin.\n\n"
    #     "Geef je antwoord terug als geldige JSON in een json-blok in het formaat:\n"
    #     "{{\"rationale\": your_rationale}}\n\n"
    #     "Analyseer de volgende bewering:\n"
    #     "Sectie: {section}\n"
    #     "Vorige zin: {previous_sentence}\n"
    #     "Beweringszin: {claim}\n"
    #     "Volgende zin: {subsequent_sentence}\n"
    #     "Label: {label}\n\n"
    # )
}

COT_PROMPT = { "nl": (
                    "You are a multilingual Wikipedia citation reasoning assistant. "
                    "You are provided with a Dutch claim, its context, and a citation-needed label. "
                    "A label of 1 means the claim requires a citation. "
                    "A label of 0 means the claim does not require a citation.\n\n"
                    "Your task is to provide a clear step-by-step rationale in English explaining why the label applies to the given claim. "
                    "Return only your answer as valid JSON in a json block in the format:\n"
                    "{{\"response\": \"your_rationale\"}}\n\n"
                    "Analyze the following claim in Dutch:\n"
                    "Section: {section}\n"
                    "Previous sentence: {previous_sentence}\n"
                    "Claim: {claim}\n"
                    "Subsequent sentence: {subsequent_sentence}\n"
                    "Label: {label}\n\n"
                    "Now think step-by-step and then provide your answer as JSON only:"
    ),
}

COT_WIKI_SHORT_PROMPT = {
    "nl": (
        "You are a multilingual Wikipedia citation reasoning assistant. "
        "You are provided with a Dutch claim, its context, and a citation-needed label. "
        "A label of 1 means the claim requires a citation. "
        "A label of 0 means the claim does not require a citation.\n\n"
        "Your task is twofold. First, analyze the claim and select the appropriate citation reason based on the provided label. "
        "Second, provide a concise, step-by-step explanation in English explaining why this citation reason applies to the claim. "
        "The explanation should be one sentence only, and focus on general patterns between the claim and the reason. "
        "Select from the citation reasons below:\n"
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
        "• Common Knowledge – The statement only contains well-known or established facts.\n"
        "• Plot – The statement describes the plot or characters of a book, film, or similar work that is the article’s subject.\n"
        "• Other (No Citation Needed) – The statement does not need a citation for other reasons (briefly describe why).\n\n"
        "Return only your answer as valid JSON in a json block in the format:\n"
        "{{\"reason\": \"selected_reason\", \"explanation\": \"your_explanation\"}}\n\n"
        "Analyze the following claim in Dutch:\n"
        "Section: {section}\n"
        "Previous sentence: {previous_sentence}\n"
        "Claim: {claim}\n"
        "Subsequent sentence: {subsequent_sentence}\n"
        "Label: {label}\n\n"
        "Think step by step and then provide the reason and explanation."
    ),
}

COT_WIKI_PROMPT = { "nl": (
                "You are a multilingual Wikipedia citation reasoning assistant. "
                "You are provided with a Dutch claim, its context, and a citation-needed label. "
                "A label of 1 means the claim requires a citation. "
                "A label of 0 means the claim does not require a citation.\n\n"
                "Your task is to provide a clear step-by-step rationale in English explaining why the label applies to the given claim. "
                "Your rationale must be based on one or more of the following Wikipedia citation reason categories, and you should explicitly mention the relevant category or categories in your explanation:\n"
                "# Reasons why citations are needed\n"
                "• Quotation – The statement is a direct quotation or close paraphrase of a source.\n"
                "• Statistics – The statement contains statistics or quantitative data.\n"
                "• Controversial – The statement makes surprising or potentially controversial claims (e.g., conspiracy theories).\n"
                "• Opinion – The statement expresses a person’s subjective opinion or belief.\n"
                "• Private Life – The statement contains claims about a person’s private life (e.g., date of birth, relationship status).\n"
                "• Scientific – The statement includes technical or scientific claims.\n"
                "• Historical – The statement makes general or historical claims that are not common knowledge.\n"
                "• Other (Needs Citation) – The statement requires a citation for other reasons (briefly describe why).\n\n"
                "# Reasons why citations are not needed\n"
                "• Common Knowledge – The statement only contains well-known or established facts.\n"
                "• Plot – The statement describes the plot or characters of a book, film, or similar work that is the article’s subject.\n"
                "• Other (No Citation Needed) – The statement does not need a citation for other reasons (briefly describe why).\n\n"
                "Return only your answer as valid JSON in a json block in the format:\n"
                "{{\"response\": \"your_rationale\"}}\n\n"
                "Analyze the following claim in Dutch:\n"
                "Section: {section}\n"
                "Previous sentence: {previous_sentence}\n"
                "Claim: {claim}\n"
                "Subsequent sentence: {subsequent_sentence}\n"
                "Label: {label}\n\n"
                "Now think step-by-step and then provide your answer as JSON only:"
    ),
}


# PROMPT_CLAIM = (
#     "You are a Wikipedia citation reasoning assistant. "
#     "Your task is to explain *why* a given claim should or should not require a citation. "
#     "A label of 1 means the claim *requires* a citation. "
#     "A label of 0 means the claim *does not require* a citation.\n\n"
#     "Provide a concise rationale through reasoning step by step with no more than 3 sentences.\n\n"
#     "Return your answer as valid JSON in a json block in the format:\n"
#     "{{\"rationale\": your_rationale}}\n\n"

#     "Analyze the following claim:\n"
#     "Claim: {claim}\n"
#     "Label: {label}\n\n"
# )