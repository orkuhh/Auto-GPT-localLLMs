from llama_cpp import Llama

llm = Llama(model_path="path/to/llm", n_ctx=2048, embedding=True)

def create_chat_completion(messages, model=llm, temperature=0.36, max_tokens=0)->str:
    response = llm(messages[0]["content"], stop=["Q:", "### Human:"], echo=False, temperature=temperature, max_tokens=max_tokens)
    return response["choices"][0]["text"]