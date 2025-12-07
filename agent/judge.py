from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# --- TASK 3: JUDGE ---
def make_verdict(original_text, research_data , llm):
    final_prompt = ChatPromptTemplate.from_template(
        """
        Original Article: {article}
        
        External Evidence Found: {evidence}
        
        Task: Decide if the article is Fake or Real.
        Output Format:
        - VERDICT: [Real/Fake]
        - CONFIDENCE: [0-100%]
        - REASONING: [Explain why, citing the evidence]
        """
    )
    chain = final_prompt | llm
    return chain.invoke({"article": original_text, "evidence": research_data}).content
