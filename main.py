import os
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatPerplexity
from dotenv import load_dotenv

import random as rm

from agent.analyse_news import analyze_news
from agent.research import conduct_research
from agent.judge import make_verdict
from dataloader import fake_news_dataset, dataset_loader

GOSSIPCOP_MAPPER = {
        'id': 'ids',
        'title': 'titles',
        'news_url': 'news_urls',
        'tweet_ids': 'tweet_ids',
        'label': 'labels',
    }

FRN_MAPPER = {
    'title': 'titles',
    'text': 'texts',
    'subject': 'subjects',
    'label': 'labels'
}

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY")



# 1. Setup
open_ai_llm = ChatOpenAI(model="gpt-5-nano-2025-08-07", temperature=0 , api_key=OPENAI_API_KEY) # Low temp for factual accuracy

# tavily_search_tool = TavilySearchResults(max_results=3)

pplx_llm = ChatPerplexity(
        temperature=0,
        model="sonar-pro",
        pplx_api_key=PPLX_API_KEY
    )






# --- MAIN AGENT LOOP ---
def fake_news_agent(news_item : dict):
    print("--- Step 1: Breaking down news... ---")
    analysis = analyze_news(article_text=news_item.get("text",None) , news_url=news_item.get("new_url",None) , llm=pplx_llm)
    print(analysis)
    
    print("\n--- Step 2: Conducting Research... ---")
    evidence = conduct_research(analysis , llm=pplx_llm)
    print(f"Found {len(evidence)} sources.")
    
    print("\n--- Step 3: Final Decision... ---")
    verdict = make_verdict(analysis, evidence , llm=open_ai_llm)
    print(verdict)

# # Example Usage
# fake_news = "Breaking: The moon has officially been purchased by a tech billionaire for $50."
# fake_news_agent(fake_news)



if __name__ == "__main__":
    dataset = dataset_loader("data/FTDS.csv" , mapper=FRN_MAPPER)
    print(f"Dataset size: {len(dataset)}")
    idxs = [rm.randint(0, len(dataset)-1) for _ in range(3)]
    for idx in idxs:
        item = dataset.__getitem__(0)
        print(item)
        print(f"--- New Article 0:{0} ---".center(50,'-'))
        print("Title:", item['title'])
        fake_news_agent(item)
    # n=1
    # dataset = fake_news_dataset("data/fake_or_real_news.csv")
    # fake_news = [dataset[i] for i in range(n)]  
    # for i in range(len(fake_news)):
    #     print(f"--- New Article idx:{i} ---".center(50,'-'))
    #     print("Title:", fake_news[i]['title'])
    #     fake_news_agent(fake_news[i]['text'])

    # print("openai_api_hey:", OPENAI_API_KEY)
    # print("tavily_api_key:", TAVILY_API_KEY)