"""
茶語時光 RAG：索引與查詢
用法：
    python rag.py          # 重建索引並測試一題
    import rag; rag.ask("孕婦可以喝什麼？")
"""
from pathlib import Path

import anthropic
import chromadb
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

# ===== 設定 =====
KB_FOLDER = "data"                      # 知識庫資料夾
DB_PATH = "./db"                        # Chroma 資料存放位置
COLLECTION = "tea_shop"
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
LLM_MODEL = "claude-sonnet-5-5"

# ===== 只載入一次的元件 =====
model = SentenceTransformer(EMBED_MODEL)
db = chromadb.PersistentClient(path=DB_PATH)
col = db.get_or_create_collection(COLLECTION)
llm = anthropic.Anthropic()


# ===== 索引流水線 =====
def load(folder: str) -> list[dict]:
    """讀取 folder 內所有 .md 檔，回傳 [{'text': ..., 'source': 檔名}, ...]"""
    docs = []
    for p in Path(folder).glob("*.md"):
        docs.append({"text": p.read_text(encoding="utf-8"), "source": p.name})
    return docs


def split(docs: list[dict]) -> list[dict]:
    """把每份文件依行切塊，每塊前面加上所屬標題"""
    chunks = []
    for doc in docs:
        current = ""
        for line in doc["text"].splitlines():
            line = line.strip()
            if line.startswith("## "):
                current = line[3:]
            elif line == "" or line.startswith("#") or line.startswith(">"):
                continue
            else:
                chunks.append({
                    "text": f"【{current}】{line}",
                    "source": doc["source"],
                    "section": current,
                })
    return chunks


def embed(texts: list[str]) -> np.ndarray:
    """把一串文字轉成正規化後的向量矩陣"""
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def store(chunks: list[dict], vectors: np.ndarray) -> None:
    """把 chunks 和對應的向量存進 Chroma"""
    col.upsert(
        ids=[f"chunk-{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=vectors.tolist(),
        metadatas=[{"source": c["source"], "section": c["section"]} for c in chunks],
    )


def build_index(folder: str = KB_FOLDER) -> int:
    """重建整個索引，回傳總塊數"""
    global col
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    col = db.get_or_create_collection(COLLECTION)

    chunks = split(load(folder))
    store(chunks, embed([c["text"] for c in chunks]))
    return col.count()


# ===== 查詢流水線 =====
def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """找出和問題最相關的 top_k 塊"""
    q = embed([question])
    res = col.query(query_embeddings=q.tolist(), n_results=top_k)

    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        results.append({
            "text": doc,
            "source": meta["source"],
            "section": meta["section"],
            "score": 1 - dist / 2,
        })
    return results


def retrieve_with_sections(question: str, top_k: int = 3) -> list[dict]:
    """先找出最相關的句子，再把它們所屬 section 的全部句子取出來"""
    hits = retrieve(question, top_k)
    sections = list(dict.fromkeys(h["section"] for h in hits))

    results = []
    for sec in sections:
        res = col.get(where={"section": sec}, include=["documents", "metadatas"])
        rows = sorted(
            zip(res["ids"], res["documents"], res["metadatas"]),
            key=lambda r: int(r[0].split("-")[1]),
        )
        for _id, doc, meta in rows:
            results.append({"text": doc, "source": meta["source"], "section": meta["section"]})
    return results


def build_prompt(question: str, chunks: list[dict]) -> str:
    """把問題和檢索到的資料組成提示詞"""
    lines = [f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1)]
    context = "\n".join(lines)

    return f"""你是「茶語時光」手搖飲店的客服助理。

規則：
1. 只能根據 <資料> 中的內容回答，不要自己推測或補充。
2. 如果資料中沒有答案，請回答「抱歉，這個問題我無法回答，請洽門市 02-2700-1234。」
3. 回答要親切，使用繁體中文。
4. 如果資料中有多個符合條件的選項，請全部列出，不要只挑其中幾個。
5. 在回答最後標出參考的資料編號，例如：（參考：[1][3]）

<資料>
{context}
</資料>

<問題>
{question}
</問題>"""


def generate(prompt: str) -> str:
    """把提示詞交給 Claude，回傳回答文字"""
    msg = llm.messages.create(
        model=LLM_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def ask(question: str, top_k: int = 3) -> str:
    chunks = retrieve_with_sections(question, top_k)
    return generate(build_prompt(question, chunks))


# ===== Agentic RAG =====
AGENT_SYSTEM = """你是「茶語時光」手搖飲店的客服助理。

規則：
1. 回答前一定要用工具查詢資料，只能根據查到的內容回答，不要自己推測。
2. 如果查不到答案，請回答「抱歉，這個問題我無法回答，請洽門市 02-2700-1234。」
3. 回答要親切，使用繁體中文。
4. 如果有多個符合條件的選項，請全部列出。"""


def section_names() -> list[str]:
    """取得目前索引中所有段落名稱（每次即時查詢，重建索引後也會是最新的）"""
    metas = col.get(include=["metadatas"])["metadatas"]
    return list(dict.fromkeys(m["section"] for m in metas))


def search_kb(query: str) -> str:
    """工具：語意搜尋"""
    chunks = retrieve_with_sections(query, top_k=3)
    return "\n".join(c["text"] for c in chunks)


def get_section(name: str) -> str:
    """工具：取出某個段落的全部內容，依原文順序排列"""
    res = col.get(where={"section": name}, include=["documents"])
    if not res["documents"]:
        return f"找不到段落「{name}」。可用的段落有：{'、'.join(section_names())}"
    rows = sorted(zip(res["ids"], res["documents"]), key=lambda r: int(r[0].split("-")[1]))
    return "\n".join(doc for _id, doc in rows)


TOOL_FUNCS = {"search_kb": search_kb, "get_section": get_section}


def get_tools() -> list[dict]:
    """工具說明書；段落清單即時產生"""
    return [
        {
            "name": "search_kb",
            "description": "用語意搜尋茶語時光的知識庫。適合查詢具體問題，例如某個飲料能不能做熱的、刷卡規定、營業時間。",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "要搜尋的問題或關鍵字"}},
                "required": ["query"],
            },
        },
        {
            "name": "get_section",
            "description": "取出知識庫中某個段落的完整內容。適合需要完整清單的問題，例如全部菜單、全部規定。"
                           f"可用的段落：{'、'.join(section_names())}",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "段落名稱，必須是可用段落之一"}},
                "required": ["name"],
            },
        },
    ]


def agent_ask(question: str, max_steps: int = 6) -> tuple[str, list[dict]]:
    """Agentic RAG：讓 Claude 自己決定用哪些工具。回傳 (回答, 查詢路徑)"""
    messages = [{"role": "user", "content": question}]
    tools = get_tools()
    trace = []

    for _ in range(max_steps):
        msg = llm.messages.create(
            model=LLM_MODEL,
            max_tokens=2000,
            system=AGENT_SYSTEM,
            tools=tools,
            messages=messages,
        )
        if msg.stop_reason != "tool_use":
            answer = "".join(b.text for b in msg.content if b.type == "text")
            return answer, trace

        messages.append({"role": "assistant", "content": msg.content})
        results = []
        for b in msg.content:
            if b.type == "tool_use":
                output = TOOL_FUNCS[b.name](**b.input)
                trace.append({"tool": b.name, "input": b.input, "output": output})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": output})
        messages.append({"role": "user", "content": results})

    return "抱歉，這個問題查詢太多次了，請洽門市 02-2700-1234。", trace


if __name__ == "__main__":
    print(f"索引完成，共 {build_index()} 塊")