import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==========================================
# 1. 初始化 FastAPI 應用程式
# ==========================================
app = FastAPI(
    title="TAIDE RAG API",
    description="基於 Ollama (TAIDE) 與 Qdrant 的 RAG (檢索增強生成) 系統"
)

# ==========================================
# 2. 環境變數設定
# ==========================================
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
LLM_MODEL = os.getenv("LLM_MODEL", "cwchang/llama3-taide-lx-8b-chat-alpha1:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")

LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))

# ==========================================
# 3. 初始化 LangChain 元件 (Embeddings 與 LLM)
# ==========================================
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)

llm = Ollama(
    base_url=OLLAMA_BASE_URL,
    model=LLM_MODEL,
    num_ctx=LLM_NUM_CTX,
    temperature=0.2,
)

# ==========================================
# 4. 初始化 Qdrant 向量資料庫
# ==========================================
client = QdrantClient(url=QDRANT_URL)
collection_name = os.getenv("COLLECTION_NAME", "taide_rag_collection")

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

# ==========================================
# 5. 文件切塊設定
# ==========================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
)


def split_and_store(text: str, base_metadata: dict):
    """
    將文字切塊後寫入 Qdrant。
    """
    chunks = text_splitter.split_text(text)
    if not chunks:
        return 0

    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = dict(base_metadata or {})
        meta["chunk"] = i
        meta["chunk_chars"] = len(chunk)
        metadatas.append(meta)

    vector_store.add_texts(texts=chunks, metadatas=metadatas)
    return len(chunks)


# ==========================================
# 5.1 刪除相關 helper
# ==========================================
def make_source_filter(source: str) -> Filter:
    """
    建立 Qdrant filter，用來找出某個 metadata.source 的所有 chunks。
    """
    return Filter(
        must=[
            FieldCondition(
                key="metadata.source",
                match=MatchValue(value=source),
            )
        ]
    )


# ==========================================
# 6. 定義 API 請求的資料結構 (Pydantic Models)
# ==========================================
class DocumentInput(BaseModel):
    text: str
    metadata: dict = {}


class QueryInput(BaseModel):
    question: str


# ==========================================
# 7. API 路由設定
# ==========================================
@app.get("/")
async def root():
    return {
        "service": "TAIDE RAG API",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "ingest": "POST /ingest",
            "upload": "POST /upload",
            "query": "POST /query",
            "delete_collection": "DELETE /documents/collection",
            "delete_by_source": "DELETE /documents/source/{source}",
        },
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_document(doc: DocumentInput):
    try:
        n = split_and_store(doc.text, doc.metadata or {"source": "inline"})
        return {
            "status": "success",
            "message": f"已將文件切分為 {n} 個 chunk 並加入向量資料庫",
            "chunks": n,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="未提供檔案名稱")

    allowed_extensions = (".md", ".txt")
    file_ext = os.path.splitext(filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的檔案格式：{file_ext}，僅支援 {allowed_extensions}",
        )

    try:
        content_bytes = await file.read()
        text = content_bytes.decode("utf-8")

        base_metadata = {
            "source": filename,
            "chars": len(text),
        }

        n = split_and_store(text, base_metadata)

        return {
            "status": "success",
            "message": f"檔案 '{filename}' 已切分為 {n} 個 chunk 並加入向量資料庫",
            "chunks": n,
            "chars": len(text),
        }
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="檔案編碼不是 UTF-8，請轉換後再上傳")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_rag(query: QueryInput):
    try:
        system_prompt = (
            "你是一個有用的繁體中文 AI 助手。請根據以下提供的上下文來回答問題。\n"
            "如果你不知道答案，請直接說不知道，不要編造答案。\n\n"
            "上下文：\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(
            vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K}),
            question_answer_chain
        )

        response = rag_chain.invoke({"input": query.question})

        return {
            "question": query.question,
            "answer": response["answer"],
            "source_documents": [doc.page_content for doc in response["context"]],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/collection")
async def delete_collection(recreate: bool = True):
    """
    刪除整個 Qdrant collection。
    若 recreate=true，會自動建立一個空的同名稱 collection。
    """
    try:
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        if recreate:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

        return {
            "status": "success",
            "message": "已刪除整個向量資料庫 collection"
                       + ("，並已重新建立" if recreate else ""),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/source/{source}")
async def delete_documents_by_source(source: str):
    """
    依照 metadata.source 刪除某個來源的所有 chunks。
    例如：DELETE /documents/source/myfile.md
    """
    try:
        if not client.collection_exists(collection_name):
            return {
                "status": "success",
                "message": "collection 不存在，沒有可刪除的資料",
                "deleted": 0,
            }

        source_filter = make_source_filter(source)

        # 先計算符合條件的 points 數量
        count_result = client.count(
            collection_name=collection_name,
            count_filter=source_filter,
        )
        deleted_count = count_result.count

        # 依 filter 刪除 points
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(filter=source_filter),
        )

        return {
            "status": "success",
            "message": f"已刪除來源 '{source}' 的所有 chunks",
            "deleted": deleted_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
