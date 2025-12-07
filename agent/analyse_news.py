from langchain_core.prompts import ChatPromptTemplate


def analyze_article(article_text , llm , news_url=None):
    prompt = ChatPromptTemplate.from_template(
        """
        Analyze this news text:
        {text}
        
        Output:
        1. Summary: [2 sentences]
        2. Key Claims: [List specific factual claims to check]
        """
    )
    chain = prompt | llm
    return chain.invoke({"text": article_text}).content



def analyze_news_with_url(news_url, llm):
    prompt = ChatPromptTemplate.from_template(
        """
        The article can be found at: {url}
        
        Output:
        1. Summary: [2 sentences]
        2. Key Claims: [List specific factual claims to check]
        """
    )
    chain = prompt | llm
    return chain.invoke({"url": news_url}).content


def analyze_news(article_text:None, news_url=None , llm=None):
    if news_url:
        return analyze_news_with_url(news_url, llm)
    elif article_text:
        return analyze_article(article_text , llm)
    else:
        raise ValueError("Either article_text or news_url must be provided.")