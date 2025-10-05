PROMPT = """Your task is to determine whether a claim needs a citation (label=1) or no citation (label=0).

Claim: "{{claim}}"

Output the label in the following format:
{"label": <label>}
"""