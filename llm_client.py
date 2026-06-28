from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt & set context in messages 

    SYSTEM_PROMPT = f""" 
    
    You are a helpful assistant that is an expert in NASA mission histories. 
    You are to respond to user questions about NASA missions utilizing the context provided below and following the rules established below as well. 

    CONTEXT: {context}

    RULES 
    1. All parts of your answer regarding a NASA mission upon user query are to be directly grounded in the context provided with citation in the form [SOURCE: (document source name/reference)] constructed from the context. 
    2. If you are not sure about the answer for a user query due to insufficient context information do not try to fill in the gap, simply state that you do not have sufficient information to answer the question. 
    3. If parts of the context conflict each other, state the conflict with each separate statement and its citation within the context. 
    4. Address the question directly and do not provide information not relevant to the user query.
    5. Maintain a professional and informational tone. 

    USER QUERY  
    {user_message}


    """

   
    # Add chat history

    history = []
    history.append({"role": "system", "content": SYSTEM_PROMPT})
    if conversation_history: 
        history.extend(conversation_history)

    history.append({"role": "user", "content": user_message})

    print(f"history being used for response generation: {history}")

    # Create OpenAI Client
    client = OpenAI(api_key=openai_key)

    # Send request to OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=history
    )
    
    # Return response
    return response.choices[0].message.content
