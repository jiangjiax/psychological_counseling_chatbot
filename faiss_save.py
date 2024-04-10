from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_zhipu.embeddings import ZhipuAIEmbeddings
from utils import get_ak
import asyncio
import time

with open("data.txt") as f:
    data = f.read()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    length_function=len,
    is_separator_regex=True,
)
docs = text_splitter.create_documents([data])

startTimer = time.time()
db = FAISS.from_documents(docs, ZhipuAIEmbeddings(
    api_key=get_ak(),
))
print("Time taken:", time.time() - startTimer)

async def asimilarity():
    results = await db.asimilarity_search("我的同事批评我怎么办？", k=3)
    print(results)

asyncio.run(asimilarity())

db.save_local("faiss_index")