VANILLA_PROMPTS = {
    "en": {
        "system": (
            "You are a classifier. Decide whether the given claim requires a citation.\n"
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
    },
    # "no": {
    #     "system": (
    #         "Du er en klassifiserer. Bestem om påstanden trenger en kildehenvisning. "
    #         "Bruk 1 hvis påstanden trenger en kilde. Bruk 0 hvis den ikke gjør det.\n"
    #         "Returner kun JSON i formatet: {\"label\": 0} eller {\"label\": 1}. "
    #         "Ingen forklaringer eller ekstra tekst."
    #     ),
    #     "user_claim": "Påstand: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "it": {
    #     "system": (
    #         "Sei un classificatore. Decidi se l'affermazione data richiede una citazione. "
    #         "Usa 1 se l'affermazione necessita di una citazione. Usa 0 se non la necessita.\n"
    #         "Restituisci solo JSON nel formato: {\"label\": 0} oppure {\"label\": 1}. "
    #         "Nessuna spiegazione o testo aggiuntivo."
    #     ),
    #     "user_claim": "Affermazione: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "pt": {
    #     "system": (
    #         "Você é um classificador. Decida se a afirmação dada precisa de uma citação. "
    #         "Use 1 se a afirmação precisar de citação. Use 0 se não precisar.\n"
    #         "Retorne apenas JSON no formato: {\"label\": 0} ou {\"label\": 1}. "
    #         "Sem explicações ou texto extra."
    #     ),
    #     "user_claim": "Afirmação: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "ro": {
    #     "system": (
    #         "Ești un clasificator. Decide dacă afirmația dată necesită o sursă. "
    #         "Folosește 1 dacă afirmația necesită o sursă. Folosește 0 dacă nu necesită.\n"
    #         "Returnează doar JSON în formatul: {\"label\": 0} sau {\"label\": 1}. "
    #         "Fără explicații sau text suplimentar."
    #     ),
    #     "user_claim": "Afirmație: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "ru": {
    #     "system": (
    #         "Вы классификатор. Определите, требует ли утверждение ссылки. "
    #         "Используйте 1, если утверждение требует ссылки. Используйте 0, если не требует.\n"
    #         "Возвращайте только JSON в формате: {\"label\": 0} или {\"label\": 1}. "
    #         "Без объяснений и лишнего текста."
    #     ),
    #     "user_claim": "Утверждение: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "uk": {
    #     "system": (
    #         "Ви є класифікатором. Визначте, чи потребує твердження посилання. "
    #         "Використовуйте 1, якщо твердження потребує посилання. Використовуйте 0, якщо не потребує.\n"
    #         "Поверніть лише JSON у форматі: {\"label\": 0} або {\"label\": 1}. "
    #         "Без пояснень чи додаткового тексту."
    #     ),
    #     "user_claim": "Твердження: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "bg": {
    #     "system": (
    #         "Вие сте класификатор. Определете дали твърдението се нуждае от източник. "
    #         "Използвайте 1, ако твърдението се нуждае от цитиране. Използвайте 0, ако не се нуждае.\n"
    #         "Върнете само JSON във формат: {\"label\": 0} или {\"label\": 1}. "
    #         "Без обяснения или допълнителен текст."
    #     ),
    #     "user_claim": "Твърдение: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "id": {
    #     "system": (
    #         "Anda adalah pengklasifikasi. Tentukan apakah klaim tersebut memerlukan kutipan. "
    #         "Gunakan 1 jika klaim memerlukan kutipan. Gunakan 0 jika tidak.\n"
    #         "Kembalikan hanya JSON dalam format: {\"label\": 0} atau {\"label\": 1}. "
    #         "Tanpa penjelasan atau teks tambahan."
    #     ),
    #     "user_claim": "Klaim: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "vi": {
    #     "system": (
    #         "Bạn là bộ phân loại. Hãy xác định liệu tuyên bố có cần trích dẫn hay không. "
    #         "Dùng 1 nếu tuyên bố cần trích dẫn. Dùng 0 nếu không cần.\n"
    #         "Chỉ trả về JSON theo định dạng: {\"label\": 0} hoặc {\"label\": 1}. "
    #         "Không giải thích hoặc thêm văn bản."
    #     ),
    #     "user_claim": "Tuyên bố: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # },
    # "tr": {
    #     "system": (
    #         "Bir sınıflandırıcısınız. Verilen iddianın kaynak gerektirip gerektirmediğine karar verin. "
    #         "İddia kaynak gerektiriyorsa 1'i, gerektirmiyorsa 0'ı kullanın.\n"
    #         "Yalnızca şu formatta JSON döndürün: {\"label\": 0} veya {\"label\": 1}. "
    #         "Açıklama veya ek metin yok."
    #     ),
    #     "user_claim": "İddia: {claim}",
    #     "assistant": '{{"label": {label}}}'
    # }
}