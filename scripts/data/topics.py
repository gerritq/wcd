import json
import requests


def get_topic(article: str, lang: str):
    inference_url = 'https://api.wikimedia.org/service/lw/inference/v1/models/outlink-topic-model:predict'
    headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    data = {'page_title': article, "lang": lang}
    try:
        response = requests.post(inference_url, headers=headers, data=json.dumps(data))
    except Exception as e:
        print(f"API Error for article {article} in language {lang}: {e}")
        return None
    
    try:
        r = response.json()
        print(r)
        sorted_results = sorted(r['prediction']['results'], key=lambda x: x['score'], reverse=True)
        topic = sorted_results[0]['topic']
        return topic
    except Exception as e:
        print(f"Parsing Error for article {article} in language {lang}: {e}")
        return None

if __name__ == "__main__":
    t = get_topic("Donald_Trump", "en")
    print("Topic", t)