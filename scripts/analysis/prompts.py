PROMPT = """Your task is to determine whether a claim needs a citation (label=1) or no citation (label=0).

Claim: {{claim}}

Output the label in the following format:
{"label": <label>}
"""

# PROMPT = """Do you think the following claim needs a citation (YES) or no citation (NO)?

# Claim: ""{{claim}}""

# Only answer with YES or NO. Not other text.
# """