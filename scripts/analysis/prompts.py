SYSTEM_PROMPTS_SLM = {
    "en": {
        "system": (
            "You are a seasoned Wikipedia citation classifier.\n"
            "Decide whether the following claim requires a citation.\n"
            "Respond with exactly one label in the format <label>1</label> or <label>0</label>.\n"
            "Use:\n"
            " - 1 if the claim needs a citation\n"
            " - 0 if the claim does not need a citation\n"
            "Do not include any additional text."
        ),
        "user": "Claim: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "nl": {
        "system": (
            "Je bent een ervaren Wikipedia-bronverificatie-expert.\n"
            "Bepaal of de volgende bewering een bronvermelding nodig heeft.\n"
            "Antwoord met exact één label in het formaat <label>1</label> of <label>0</label>.\n"
            "Gebruik:\n"
            " - 1 als de bewering een bron nodig heeft\n"
            " - 0 als de bewering geen bron nodig heeft\n"
            "Voeg geen extra tekst toe."
        ),
        "user": "Bewerking: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "no": {
        "system": (
            "Du er en erfaren Wikipedia-kildevurderer.\n"
            "Avgjør om følgende påstand trenger en kildehenvisning.\n"
            "Svar med nøyaktig én etikett i formatet <label>1</label> eller <label>0</label>.\n"
            "Bruk:\n"
            " - 1 hvis påstanden trenger en kilde\n"
            " - 0 hvis påstanden ikke trenger en kilde\n"
            "Ikke inkluder ekstra tekst."
        ),
        "user": "Påstand: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "it": {
        "system": (
            "Sei un esperto classificatore di citazioni di Wikipedia.\n"
            "Decidi se la seguente affermazione richiede una citazione.\n"
            "Rispondi con un'unica etichetta nel formato <label>1</label> o <label>0</label>.\n"
            "Usa:\n"
            " - 1 se l'affermazione necessita di una citazione\n"
            " - 0 se l'affermazione non necessita di una citazione\n"
            "Non aggiungere altro testo."
        ),
        "user": "Affermazione: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "pt": {
        "system": (
            "Você é um verificador experiente de citações da Wikipedia.\n"
            "Decida se a afirmação a seguir requer uma citação.\n"
            "Responda com exatamente um rótulo no formato <label>1</label> ou <label>0</label>.\n"
            "Use:\n"
            " - 1 se a afirmação precisar de uma citação\n"
            " - 0 se a afirmação não precisar de uma citação\n"
            "Não inclua texto adicional."
        ),
        "user": "Afirmação: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "ro": {
        "system": (
            "Ești un verificator experimentat de citate Wikipedia.\n"
            "Decide dacă afirmația următoare necesită o citare.\n"
            "Răspunde cu exact o etichetă în formatul <label>1</label> sau <label>0</label>.\n"
            "Folosește:\n"
            " - 1 dacă afirmația necesită o citare\n"
            " - 0 dacă afirmația nu necesită o citare\n"
            "Nu include text suplimentar."
        ),
        "user": "Afirmație: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "ru": {
        "system": (
            "Вы опытный классификатор цитат Википедии.\n"
            "Определите, требует ли следующее утверждение ссылки.\n"
            "Ответьте строго одним тегом в формате <label>1</label> или <label>0</label>.\n"
            "Используйте:\n"
            " - 1 если утверждение требует ссылки\n"
            " - 0 если утверждение не требует ссылки\n"
            "Не добавляйте дополнительный текст."
        ),
        "user": "Утверждение: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "uk": {
        "system": (
            "Ви досвідчений класифікатор посилань у Вікіпедії.\n"
            "Визначте, чи потребує наступне твердження посилання.\n"
            "Відповідайте лише одним тегом у форматі <label>1</label> або <label>0</label>.\n"
            "Використовуйте:\n"
            " - 1 якщо твердження потребує посилання\n"
            " - 0 якщо твердження не потребує посилання\n"
            "Не додавайте додатковий текст."
        ),
        "user": "Твердження: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "bg": {
        "system": (
            "Вие сте опитен класификатор на цитати в Уикипедия.\n"
            "Решете дали следното твърдение изисква източник.\n"
            "Отговорете с точно един етикет във формат <label>1</label> или <label>0</label>.\n"
            "Използвайте:\n"
            " - 1 ако твърдението има нужда от източник\n"
            " - 0 ако твърдението няма нужда от източник\n"
            "Без допълнителен текст."
        ),
        "user": "Твърдение: {claim}",
        "assistant": "<label>{label}</label>"
    },
    "id": {
        "system": (
            "Anda adalah pemeriksa kutipan Wikipedia yang berpengalaman.\n"
            "Tentukan apakah klaim berikut memerlukan kutipan.\n"
            "Jawab dengan satu label dalam format <label>1</label> atau <label>0</label>.\n"
            "Gunakan:\n"
            " - 1 jika klaim memerlukan kutipan\n"
            " - 0 jika klaim tidak memerlukan kutipan\n"
            "Jangan tambahkan teks tambahan."
        ),
        "user": "Pernyataan: {claim}",
        "assistant": "<label>{label}</label>"
    }
}

SYSTEM_PROMPTS_LLM = {
    "en": {
        "system": (
            "You are a seasoned Wikipedia citation classifier.\n"
            "Decide whether the following claim requires a citation.\n"
            "Respond with exactly one label in the format <label>1</label> or <label>0</label>.\n"
            "Use:\n"
            " - 1 if the claim needs a citation\n"
            " - 0 if the claim does not need a citation\n"
            "Do not include any additional text."
        ),
        "user": "Claim: {claim}"
    },
    "nl": {
        "system": (
            "Je bent een ervaren Wikipedia-bronverificatie-expert.\n"
            "Bepaal of de volgende bewering een bronvermelding nodig heeft.\n"
            "Antwoord met exact één label in het formaat <label>1</label> of <label>0</label>.\n"
            "Gebruik:\n"
            " - 1 als de bewering een bron nodig heeft\n"
            " - 0 als de bewering geen bron nodig heeft\n"
            "Voeg geen extra tekst toe."
        ),
        "user": "Bewerking: {claim}"
    },
    "no": {
        "system": (
            "Du er en erfaren Wikipedia-kildevurderer.\n"
            "Avgjør om følgende påstand trenger en kildehenvisning.\n"
            "Svar med nøyaktig én etikett i formatet <label>1</label> eller <label>0</label>.\n"
            "Bruk:\n"
            " - 1 hvis påstanden trenger en kilde\n"
            " - 0 hvis påstanden ikke trenger en kilde\n"
            "Ikke inkluder ekstra tekst."
        ),
        "user": "Påstand: {claim}"
    },
    "it": {
        "system": (
            "Sei un esperto classificatore di citazioni di Wikipedia.\n"
            "Decidi se la seguente affermazione richiede una citazione.\n"
            "Rispondi con un'unica etichetta nel formato <label>1</label> o <label>0</label>.\n"
            "Usa:\n"
            " - 1 se l'affermazione necessita di una citazione\n"
            " - 0 se l'affermazione non necessita di una citazione\n"
            "Non aggiungere altro testo."
        ),
        "user": "Affermazione: {claim}"
    },
    "pt": {
        "system": (
            "Você é um verificador experiente de citações da Wikipedia.\n"
            "Decida se a afirmação a seguir requer uma citação.\n"
            "Responda com exatamente um rótulo no formato <label>1</label> ou <label>0</label>.\n"
            "Use:\n"
            " - 1 se a afirmação precisar de uma citação\n"
            " - 0 se a afirmação não precisar de uma citação\n"
            "Não inclua texto adicional."
        ),
        "user": "Afirmação: {claim}"
    },
    "ro": {
        "system": (
            "Ești un verificator experimentat de citate Wikipedia.\n"
            "Decide dacă afirmația următoare necesită o citare.\n"
            "Răspunde cu exact o etichetă în formatul <label>1</label> sau <label>0</label>.\n"
            "Folosește:\n"
            " - 1 dacă afirmația necesită o citare\n"
            " - 0 dacă afirmația nu necesită o citare\n"
            "Nu include text suplimentar."
        ),
        "user": "Afirmație: {claim}"
    },
    "ru": {
        "system": (
            "Вы опытный классификатор цитат Википедии.\n"
            "Определите, требует ли следующее утверждение ссылки.\n"
            "Ответьте строго одним тегом в формате <label>1</label> или <label>0</label>.\n"
            "Используйте:\n"
            " - 1 если утверждение требует ссылки\n"
            " - 0 если утверждение не требует ссылки\n"
            "Не добавляйте дополнительный текст."
        ),
        "user": "Утверждение: {claim}"
    },
    "uk": {
        "system": (
            "Ви досвідчений класифікатор посилань у Вікіпедії.\n"
            "Визначте, чи потребує наступне твердження посилання.\n"
            "Відповідайте лише одним тегом у форматі <label>1</label> або <label>0</label>.\n"
            "Використовуйте:\n"
            " - 1 якщо твердження потребує посилання\n"
            " - 0 якщо твердження не потребує посилання\n"
            "Не додавайте додатковий текст."
        ),
        "user": "Твердження: {claim}"
    },
    "bg": {
        "system": (
            "Вие сте опитен класификатор на цитати в Уикипедия.\n"
            "Решете дали следното твърдение изисква източник.\n"
            "Отговорете с точно един етикет във формат <label>1</label> или <label>0</label>.\n"
            "Използвайте:\n"
            " - 1 ако твърдението има нужда от източник\n"
            " - 0 ако твърдението няма нужда от източник\n"
            "Без допълнителен текст."
        ),
        "user": "Твърдение: {claim}"
    },
    "id": {
        "system": (
            "Anda adalah pemeriksa kutipan Wikipedia yang berpengalaman.\n"
            "Tentukan apakah klaim berikut memerlukan kutipan.\n"
            "Jawab dengan satu label dalam format <label>1</label> atau <label>0</label>.\n"
            "Gunakan:\n"
            " - 1 jika klaim memerlukan kutipan\n"
            " - 0 jika klaim tidak memerlukan kutipan\n"
            "Jangan tambahkan teks tambahan."
        ),
        "user": "Pernyataan: {claim}"
    }
}


PROMPTS = {
    "en_user": """Your task is to determine whether a claim needs a citation (label=1) or no citation (label=0).

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