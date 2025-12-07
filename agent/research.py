from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# --- TASK 2: RESEARCHER ---
def conduct_research(analysis_result , llm):
    research_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a thorough researcher. Verify the user's claims using your online search capabilities. Provide citations."),
        ("human", "Analyze and verify the following claims: {claims}")
    ])

    # 3. Create the chain
    chain = research_prompt | llm
    
    print(f"Conducting research on: {analysis_result[:50]}...")
    
    # 4. Invoke the chain
    # The response.content will contain the synthesized research.
    # The response.additional_kwargs may contain citations depending on the API version.
    response = chain.invoke({"claims": analysis_result}).content
    
    return response