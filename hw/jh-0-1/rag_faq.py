"""
RAG 과제: 웹 크롤링 기반 FAQ 생성

설치: pip install openai chromadb requests beautifulsoup4 python-dotenv pandas numpy
실행: python rag_faq.py
"""

import os, re, json, time
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "환경변수 OPENAI_API_KEY 가 필요합니다."
client = OpenAI(api_key=OPENAI_API_KEY)

CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
embed_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name=EMBED_MODEL
)

# ============================================================
# 1. 웹 크롤링
# ============================================================

URLS = [
    "https://ko.wikipedia.org/wiki/플래시_메모리",
    "https://ko.wikipedia.org/wiki/솔리드_스테이트_드라이브",
    "https://ko.wikipedia.org/wiki/반도체",
    "https://ko.wikipedia.org/wiki/트랜지스터",
    "https://ko.wikipedia.org/wiki/집적_회로",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (RAG-study-bot/0.1)"}

def crawl_wiki(url):
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.find("h1").get_text(strip=True)
    body = soup.select_one("#mw-content-text")
    # 표, 각주 번호, 편집 링크 등 노이즈 제거
    for tag in body.select("table, sup.reference, .mw-editsection, style, script"):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    paras = [re.sub(r"\[\d+\]", "", p) for p in paras if len(p) > 40]
    return {"url": url, "title": title, "text": "\n".join(paras)}

docs = []
for url in URLS:
    try:
        d = crawl_wiki(url)
        docs.append(d)
        print(f"[OK] {d['title']}: {len(d['text'])}자")
    except Exception as e:
        print(f"[FAIL] {url}: {e}")
    time.sleep(1)   # 서버 부담 줄이기

pd.DataFrame(docs).to_csv("crawled_raw.csv", index=False, encoding="utf-8-sig")

# ============================================================
# 2. 청킹
# ============================================================

def chunk_paragraphs(text, max_chars=1000):
    chunks, buf = [], ""
    for p in text.split("\n"):
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf.strip())
            buf = ""
        buf += p + "\n"
    if buf.strip():
        chunks.append(buf.strip())
    return chunks

ids, texts, metas = [], [], []
for d_i, d in enumerate(docs):
    for c_i, c in enumerate(chunk_paragraphs(d["text"])):
        ids.append(f"d{d_i}_c{c_i}")
        texts.append(c)
        metas.append({"title": d["title"], "url": d["url"], "chunk": c_i})

print("총 청크 수:", len(ids))
print("평균 길이:", np.mean([len(t) for t in texts]).round(1))

# ============================================================
# 3. Embedding + Chroma DB 저장
# ============================================================

chroma = chromadb.PersistentClient(path="./chroma_db")

# 재실행 시 중복 방지
for name in ["docs", "faq"]:
    try:
        chroma.delete_collection(name)
    except Exception:
        pass

doc_col = chroma.create_collection(
    name="docs", embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
)

BATCH = 100
for i in range(0, len(ids), BATCH):
    doc_col.add(ids=ids[i:i+BATCH], documents=texts[i:i+BATCH], metadatas=metas[i:i+BATCH])

print("저장된 청크:", doc_col.count())

# ============================================================
# 검색 테스트
# ============================================================

res = doc_col.query(query_texts=["낸드 플래시와 노어 플래시의 차이는?"], n_results=3)
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"--- {meta['title']} (chunk {meta['chunk']}) | distance={dist:.3f}")
    print(doc[:200], "...\n")

# ============================================================
# 4. FAQ 생성
# ============================================================

FAQ_PROMPT = """아래 문서 조각을 바탕으로, 일반 독자가 궁금해할 만한 FAQ 1개를 한국어로 만들어라.
규칙:
- 답변은 문서 조각 내용에만 근거하고 2~3문장으로 작성
- 문서에 없는 내용은 지어내지 말 것
- 질문은 문서를 안 본 사람이 자연스럽게 물을 법한 형태로
JSON 형식으로만 답하라: {{"question": "...", "answer": "..."}}

문서 제목: {title}
문서 조각:
{chunk}"""

def generate_faq(chunk, title):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": FAQ_PROMPT.format(title=title, chunk=chunk)}],
    )
    return json.loads(resp.choices[0].message.content)

FAQ_PER_DOC = 5
faq_rows = []
for d_i, d in enumerate(docs):
    cand = [(cid, t) for cid, t in zip(ids, texts) if cid.startswith(f"d{d_i}_") and len(t) > 300]
    step = max(1, len(cand) // FAQ_PER_DOC)
    for cid, chunk in cand[::step][:FAQ_PER_DOC]:
        try:
            qa = generate_faq(chunk, d["title"])
            faq_rows.append({"question": qa["question"], "answer": qa["answer"],
                             "title": d["title"], "url": d["url"], "source_chunk": cid})
            print(f"[{d['title']}] Q: {qa['question']}")
        except Exception as e:
            print(f"[FAIL] {cid}: {e}")

faq_df = pd.DataFrame(faq_rows)
faq_df.to_csv("faq.csv", index=False, encoding="utf-8-sig")
print(faq_df.head(10).to_string())

# ============================================================
# FAQ를 Chroma에 저장 (질문을 임베딩, 답변은 metadata)
# ============================================================

faq_col = chroma.create_collection(
    name="faq", embedding_function=embed_fn, metadata={"hnsw:space": "cosine"}
)
faq_col.add(
    ids=[f"faq{i}" for i in range(len(faq_df))],
    documents=faq_df["question"].tolist(),
    metadatas=faq_df[["answer", "title", "url", "source_chunk"]].to_dict("records"),
)
print("저장된 FAQ:", faq_col.count())

# ============================================================
# 5. 검색 성능 평가 (Hit@k)
# ============================================================

def hit_at_k(k):
    res = doc_col.query(query_texts=faq_df["question"].tolist(), n_results=k)
    return np.mean([src in got for src, got in zip(faq_df["source_chunk"], res["ids"])])

print(pd.DataFrame({"k": [1, 3, 5], "hit_rate": [hit_at_k(k) for k in [1, 3, 5]]}).to_string(index=False))

# ============================================================
# 6. RAG 질의응답
# ============================================================

def ask(query, faq_threshold=0.2, k=4):
    # 1) FAQ 매칭
    f = faq_col.query(query_texts=[query], n_results=1)
    dist = f["distances"][0][0]
    if dist < faq_threshold:
        m = f["metadatas"][0][0]
        return f"[FAQ 매칭 d={dist:.3f}] Q: {f['documents'][0][0]}\nA: {m['answer']}\n출처: {m['url']}"

    # 2) 문서 검색 + 생성
    r = doc_col.query(query_texts=[query], n_results=k)
    context = "\n\n".join(
        f"[{i+1}] ({m['title']}) {d}"
        for i, (d, m) in enumerate(zip(r["documents"][0], r["metadatas"][0]))
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "주어진 문맥에만 근거해 한국어로 답하라. 근거로 쓴 문맥 번호를 [1]처럼 표시하라. 문맥에 답이 없으면 '제공된 자료에서 찾을 수 없습니다'라고 답하라."},
            {"role": "user", "content": f"문맥:\n{context}\n\n질문: {query}"},
        ],
    )
    sources = "\n".join(f"[{i+1}] {m['title']} - {m['url']}" for i, m in enumerate(r["metadatas"][0]))
    return f"[RAG 생성]\n{resp.choices[0].message.content}\n\n출처:\n{sources}"

TEST_QUERIES = [
    faq_df["question"].iloc[0],          # FAQ와 동일한 질문 → FAQ 매칭 기대
    "SSD가 HDD보다 빠른 이유가 뭐야?",    # 표현이 다른 질문 → RAG
    "트랜지스터는 누가 발명했어?",
    "오늘 서울 날씨 어때?",               # 범위 밖 → '찾을 수 없음' 기대
]
for q in TEST_QUERIES:
    print(f"### {q}\n{ask(q)}\n")
