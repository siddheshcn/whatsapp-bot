
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def generate_langchain_response(prompt_text, template=None):
    if template is None:
        template = """You are a helpful assistant.
        
        Question: {question}
        
        Answer:"""
        
    llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = PromptTemplate(
        input_variables=["question"],
        template=template
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=prompt_text)
    
    return response.strip()
