import requests
from bs4 import BeautifulSoup
import nltk
from sentence_transformers import SentenceTransformer, util
import torch

nltk.download('punkt')
nltk.download('punkt_tab')

URL = 'https://en.wikipedia.org/wiki/Artificial_intelligence'
MIN_TEXT_LEN = 100
SIMILARITY_THRESHOLD = 0.4

print("🤖 Loading language model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_page_text(url):
    print(f"🌐 Crawling: {url}")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['table', 'style', 'script', 'sup', 'img']):
            tag.decompose()
        content = soup.find('div', {'id': 'bodyContent'})
        page_text = content.get_text(separator=' ', strip=True) if content else ""
        return page_text if len(page_text) > MIN_TEXT_LEN else ""
    except Exception as e:
        print(f"❌ Error fetching the page: {e}")
        return ""

def split_text(text):
    return [s for s in nltk.sent_tokenize(text) if len(s) > 30]

def create_knowledge_base(text):
    sentences = split_text(text)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return sentences, embeddings

def get_response(query, base_sentences, base_embeddings):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, base_embeddings)[0]
    best_score, best_index = torch.max(similarities, dim=0)

    if best_score >= SIMILARITY_THRESHOLD:
        return base_sentences[best_index]
    else:
        return "❗ Sorry, I couldn’t find anything relevant in the Wikipedia article. Try rephrasing."

if __name__ == "__main__":
    print("🔍 Crawling the Wikipedia article on Artificial Intelligence...")
    full_text = get_page_text(URL)

    if not full_text:
        print("❌ Failed to retrieve article content. Exiting.")
        exit()

    print("✅ Article text collected. Building knowledge base...")
    base_sentences, base_embeddings = create_knowledge_base(full_text)

    print("\n💬 AIBot is ready! Ask anything about Artificial Intelligence.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("You: ")
        if query.lower().strip() == "quit":
            break
        response = get_response(query, base_sentences, base_embeddings)
        print(f"AIBot: {response}\n")
