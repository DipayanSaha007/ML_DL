from transformers import pipeline
import requests

# Set up pipelines
qa_pipeline = pipeline("question-answering")

def search_web(query):
    api_url = f"https://api.bing.microsoft.com/v7.0/search?q={query}"
    headers = {"Ocp-Apim-Subscription-Key": "YOUR_BING_API_KEY"}
    response = requests.get(api_url, headers=headers)
    results = response.json()
    return results['webPages']['value'][0]['snippet']  # First snippet as context

def main():
    while True:
        user_query = input("Ask me anything (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break

        # Step 1: Search web
        context = search_web(user_query)

        # Step 2: Extract answer
        answer = qa_pipeline({"question": user_query, "context": context})
        print("Answer:", answer['answer'])

if __name__ == "__main__":
    main()