# -*- coding: utf-8 -*-
"""
rag.py —— 课程答疑机器人核心引擎（RAG 检索增强问答）
仅使用 Python 标准库，无需安装任何第三方依赖。

工作流程（RAG）：
  1. 文档解析：支持 .txt / .md / .docx（docx 用 zipfile + ElementTree 直接解析）
  2. 文本分块：按段落聚合成 200~400 字的语义块，相邻块带少量重叠
  3. 中文切词：面向中文课程资料，采用“单字 + 二字组（bigram）”切词方案
  4. 索引构建：BM25 倒排索引（TF、DF、IDF、平均文档长度）
  5. 检索：对用户问题进行同款切词，计算 BM25 得分，返回 Top-K 相关块
  6. 答案组织：从得分最高的块中抽取与问题最相关的句子作为答案，
     并给出“相关知识点 + 来源引用”（无大模型模式）；
     若配置了大模型 API（见 llm_api.py），则自动切换为生成式回答。
"""

import json
import math
import os
import re
import time
import zipfile
from collections import Counter
from xml.etree import ElementTree as ET

# ---------------------------------------------------------------------------
# 基础配置
# ---------------------------------------------------------------------------
TOP_K = 3                 # 检索返回的相关块数量
CHUNK_SIZE = 400          # 分块目标字数
CHUNK_OVERLAP = 60        # 相邻块重叠字数
SENT_LIMIT = 280          # 直接回答的字数上限
ANSWER_SENT_COUNT = 4     # 直接回答最多抽取的句子数

STOPWORDS = set("""
的了和与及或在是有被把对对于中上下面一个一种这个那个什么如何怎么哪些吗呢啊
请帮我告诉一下可以哪些什么为什么是什么叫做称为所谓其中通过进行以及或者但是因为
所以如果那么这些那些我们你们他们它们自己大家一般通常主要其他另外例如比如
""".split())

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


# ---------------------------------------------------------------------------
# 中文切词：单字 + 二字组（bigram），英文/数字按词切分
# ---------------------------------------------------------------------------
def tokenize(text):
    """把一段文本切成词元序列（小写化、去停用词）。"""
    text = text.lower()
    tokens = []
    for m in re.finditer(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text):
        tok = m.group()
        if re.match(r"^[a-z0-9_]+$", tok):
            if len(tok) >= 2 and tok not in STOPWORDS:
                tokens.append(tok)
        else:
            if len(tok) == 1:
                if tok not in STOPWORDS:
                    tokens.append(tok)
            else:
                for i in range(len(tok) - 1):
                    big = tok[i:i + 2]
                    if big not in STOPWORDS:
                        tokens.append(big)
    return tokens


def split_sentences(text):
    """按中英文标点切句。"""
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# 文档解析：txt / md / docx
# ---------------------------------------------------------------------------
def read_docx_text(path):
    """用标准库直接解析 docx（本质是 zip 包里的 XML），提取段落文本。"""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    lines = []
    for p in root.iter(W_NS + "p"):
        buf = []
        for node in p.iter():
            if node.tag == W_NS + "t" and node.text:
                buf.append(node.text)
        line = "".join(buf).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def load_document(path):
    """按扩展名选择解析方式，返回纯文本。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return read_docx_text(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 分块：按段落聚合，带重叠
# ---------------------------------------------------------------------------
HEADING_RE = re.compile(r"^#{1,6}\s+")
NUM_HEADING_RE = re.compile(r"^(\d+[\.、]\d*|[一二三四五六七八九十]+[、.．])\s*")


def is_heading(line):
    line = line.strip()
    if not line:
        return False
    if HEADING_RE.match(line):
        return True
    if NUM_HEADING_RE.match(line) and len(line) <= 40:
        return True
    if len(line) <= 25 and (line.endswith("：") or line.endswith(":")):
        return True
    return False


def chunk_document(source, text):
    """把整篇文档切成 (章节标题, 正文块) 列表。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks = []
    cur_title = "正文"
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append({"title": cur_title, "text": buf.strip()})
            buf = ""

    for p in paragraphs:
        if is_heading(p):
            flush()
            cur_title = re.sub(r"^#{1,6}\s+", "", p).strip()[:50]
            continue
        if len(buf) + len(p) > CHUNK_SIZE and buf:
            flush()
            # 保留上一块结尾作重叠
            buf = buf[-CHUNK_OVERLAP:] if len(buf) > CHUNK_OVERLAP else buf
        buf += p + "\n"
    flush()
    return chunks


# ---------------------------------------------------------------------------
# BM25 索引
# ---------------------------------------------------------------------------
class BM25Index:
    """极简 BM25 实现：倒排统计 + 打分。"""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.n = len(chunks)
        self.avgdl = (sum(len(c["tokens"]) for c in chunks) / self.n) if self.n else 0.0
        self.k1, self.b = k1, b
        self.df = Counter()
        self.freqs = []
        for c in chunks:
            f = Counter(c["tokens"])
            self.freqs.append(f)
            for term in f:
                self.df[term] += 1
        self.idf = {}
        for t, n in self.df.items():
            self.idf[t] = math.log((self.n - n + 0.5) / (n + 0.5) + 1.0)

    def score(self, q_tokens, idx):
        f = self.freqs[idx]
        dl = len(self.chunks[idx]["tokens"])
        score = 0.0
        for t in set(q_tokens):
            if t not in f:
                continue
            tf = f[t]
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1.0, self.avgdl))
            score += self.idf.get(t, 0.0) * tf * (self.k1 + 1) / denom
        return score

    def search(self, q_tokens, top_k=TOP_K):
        scored = []
        for i in range(self.n):
            s = self.score(q_tokens, i)
            if s > 0:
                scored.append((s, i))
        scored.sort(key=lambda x: -x[0])
        results = []
        for s, i in scored[:top_k]:
            c = dict(self.chunks[i])
            c["score"] = round(s, 4)
            results.append(c)
        return results


# ---------------------------------------------------------------------------
# 知识库（文档注册表 + 索引缓存）
# ---------------------------------------------------------------------------
class KnowledgeBase:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.meta_path = os.path.join(data_dir, ".meta.json")
        os.makedirs(data_dir, exist_ok=True)

    def list_docs(self):
        """扫描知识库目录，返回全部支持的文档（.txt/.md/.docx）。"""
        meta = self._load_meta()
        files_meta = meta.get("_files", {})
        items = []
        supported = (".txt", ".md", ".docx")
        try:
            names = os.listdir(self.data_dir)
        except OSError:
            names = []
        for name in sorted(names):
            if not name.lower().endswith(supported):
                continue
            path = os.path.join(self.data_dir, name)
            if not os.path.isfile(path):
                continue
            info = files_meta.get(name, {})
            items.append({
                "name": name,
                "size": os.path.getsize(path),
                "chunks": info.get("chunks", 0),
                "mtime": os.path.getmtime(path),
            })
        return items

    def add_doc(self, name, raw_bytes):
        safe = re.sub(r"[^\w\u4e00-\u9fff.\-]+", "_", name)
        path = os.path.join(self.data_dir, safe)
        with open(path, "wb") as f:
            f.write(raw_bytes)
        self._invalidate()
        return safe

    def delete_doc(self, name):
        path = os.path.join(self.data_dir, os.path.basename(name))
        if os.path.isfile(path):
            os.remove(path)
        self._invalidate()

    def _load_meta(self):
        if os.path.isfile(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_meta(self, meta):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

    def _invalidate(self):
        meta = self._load_meta()
        meta.pop("_index", None)
        self._save_meta(meta)

    def build_index(self, force=False):
        """解析全部文档 -> 分块 -> 切词 -> BM25 索引，带缓存。"""
        docs = self.list_docs()
        sig = [(d["name"], d["size"], d["mtime"]) for d in docs]
        meta = self._load_meta()
        if not force and meta.get("_index", {}).get("sig") == sig:
            cached = meta["_index"].get("chunks")
            if cached is not None:
                idx = BM25Index(cached)
                return idx, docs
        chunks = []
        docinfo = {}
        for d in docs:
            path = os.path.join(self.data_dir, d["name"])
            try:
                text = load_document(path)
            except Exception:
                continue
            for c in chunk_document(d["name"], text):
                c["source"] = d["name"]
                c["tokens"] = tokenize(c["title"] + " " + c["text"])
                chunks.append(c)
            docinfo[d["name"]] = {
                "chunks": len([x for x in chunks if x["source"] == d["name"]]),
                "mtime": d["mtime"],
            }
        meta["_files"] = docinfo
        meta["_index"] = {"sig": sig, "chunks": chunks}
        self._save_meta(meta)
        idx = BM25Index(chunks)
        return idx, docs

    def stats(self):
        idx, docs = self.build_index()
        token_count = sum(len(c["tokens"]) for c in idx.chunks)
        return {
            "docs": len(docs),
            "chunks": len(idx.chunks),
            "tokens": token_count,
            "avgdl": round(idx.avgdl, 2),
        }


# ---------------------------------------------------------------------------
# 答案组织（无大模型时的抽取式回答；有 API key 时由 llm_api 生成）
# ---------------------------------------------------------------------------
def extractive_answer(question, top_chunks):
    """从 Top-1 块中抽取与问题最相关的句子，拼成直接回答。"""
    q_tokens = set(tokenize(question))
    primary = top_chunks[0]
    sentences = split_sentences(primary["text"])

    def rel(sent):
        toks = tokenize(sent)
        if not toks:
            return 0
        return sum(1 for t in toks if t in q_tokens) / len(toks)

    scored = sorted(sentences, key=rel, reverse=True)
    picked, length = [], 0
    for s in scored:
        if length + len(s) > SENT_LIMIT or len(picked) >= ANSWER_SENT_COUNT:
            break
        if rel(s) == 0 and picked:
            continue
        picked.append(s)
        length += len(s)
    if not picked:
        picked = sentences[:2]
    # 按原文顺序输出
    picked = [s for s in sentences if s in picked][:ANSWER_SENT_COUNT]
    direct = "".join(picked)
    return direct


def retrieve(question, kb, top_k=TOP_K):
    """仅执行检索：返回 Top-K 相关片段与耗时。"""
    t0 = time.time()
    idx, docs = kb.build_index()
    q_tokens = tokenize(question)
    top_chunks = idx.search(q_tokens, top_k)
    elapsed_ms = int((time.time() - t0) * 1000)
    return top_chunks, elapsed_ms


def build_related(top_chunks):
    """把检索结果组装为前端展示用的“相关知识点”列表。"""
    related = []
    for c in top_chunks:
        snippet = re.sub(r"\s+", " ", c["text"])[:110]
        related.append({"source": c["source"], "title": c["title"], "snippet": snippet, "score": c["score"]})
    return related


def answer_question(question, kb, llm_generate=None):
    """完整问答：检索 -> （可选大模型生成）-> 抽取式兜底 -> 组装结果。"""
    t0 = time.time()
    top_chunks, _ = retrieve(question, kb)
    if not top_chunks:
        return {
            "ok": True,
            "mode": "notfound",
            "answer": "抱歉，在已上传的课程资料中没有检索到与“%s”相关的内容。"
                      "请尝试换一种问法，或先在左侧“知识库管理”中上传更多课程资料。" % question,
            "related": [],
            "top_chunks": [],
            "elapsed_ms": int((time.time() - t0) * 1000),
        }
    # 优先使用大模型生成（若已配置），否则抽取式回答
    answer_text, mode = None, "extractive"
    if llm_generate is not None:
        try:
            answer_text = llm_generate(question, top_chunks)
            if answer_text:
                mode = "llm"
        except Exception:
            answer_text = None
    if not answer_text:
        answer_text = extractive_answer(question, top_chunks)
    return {
        "ok": True,
        "mode": mode,
        "answer": answer_text,
        "primary_source": "%s · %s" % (top_chunks[0]["source"], top_chunks[0]["title"]),
        "related": build_related(top_chunks),
        "top_chunks": top_chunks,
        "elapsed_ms": int((time.time() - t0) * 1000),
    }
