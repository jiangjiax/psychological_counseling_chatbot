import gradio as gr
from langchain_zhipu.embeddings import ZhipuAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_zhipu import ChatZhipuAI
from utils import get_ak
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate


def initialize_chatbot(vector_store_dir: str="faiss_index"):
    db = FAISS.load_local(vector_store_dir, ZhipuAIEmbeddings(api_key=get_ak()), allow_dangerous_deserialization=True)
    llm = ChatZhipuAI(model_name="glm-4", temperature=0.8, api_key=get_ak())
    
    template = """
        Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
        ------
        <ctx>
        {context}
        </ctx>
        ------
        <hs>
        {history}
        </hs>
        ------
        {question}
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    global BOT    
    BOT = RetrievalQA.from_chain_type(
        llm, 
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        chain_type='stuff',
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"
            ),
        }
    )
    # 返回向量数据库的检索结果
    BOT.return_source_documents = True

    return BOT

def chat(message, history):
    ans = BOT({"query": message})
    print(ans)
    return ans["result"]

def launch_gradio():
    demo = gr.ChatInterface(
        fn=chat,
        title="心理咨询",
        chatbot=gr.Chatbot(height=600),
    )
    demo.launch(share=True, server_name="127.0.0.1")

if __name__ == "__main__":
    initialize_chatbot()
    launch_gradio()
