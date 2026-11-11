# -*- coding: utf-8 -*-
"""
llm_api.py —— 大模型 API 接入层（DeepSeek 等 OpenAI 兼容接口）

功能：
    1. 配置文件 config.json 保存 API 设置（服务商 / Base URL / 模型 / Key / 是否启用）；
    2. generate_answer()：一次性生成式回答（RAG：检索片段作为上下文）；
    3. stream_answer()：流式生成（SSE 增量输出，前端逐字显示）；
    4. 未配置 API Key 或调用失败时返回 None / 抛异常，由调用方自动回退到
       本地抽取式回答（见 rag.py），保证系统在离线环境下永远可用。

支持的 OpenAI 兼容服务商（前端有预设，也可自定义）：
    DeepSeek      https://api.deepseek.com                       deepseek-chat
    通义千问      https://dashscope.aliyuncs.com/compatible-mode/v1   qwen-plus
    智谱 GLM      https://open.bigmodel.cn/api/paas/v4            glm-4-flash
    Moonshot      https://api.moonshot.cn/v1                     moonshot-v1-8k
"""

import json
import os
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

DEFAULTS = {
    "enabled": False,
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "api_key": "",
    "temperature": 0.3,
    "max_tokens": 600,
}

PROMPT_TEMPLATE = (
    "你是一名《人工智能导论》课程助教。请仅根据下面提供的课程资料回答学生的问题，"
    "回答要准确、简洁、分点清晰；如果资料中没有答案，请明确说“资料中未找到”。\n\n"
    "课程资料：\n%s\n\n学生问题：%s"
)


# ---------------------------------------------------------------------------
# 配置读写
# ---------------------------------------------------------------------------
def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
    merged = dict(DEFAULTS)
    merged.update(cfg or {})
    return merged


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def is_enabled():
    cfg = load_config()
    return bool(cfg.get("enabled") and cfg.get("api_key"))


def public_settings():
    """返回给前端的设置（Key 打码，不泄露明文）。"""
    cfg = load_config()
    key = cfg.get("api_key") or ""
    masked = ""
    if key:
        masked = (key[:4] + "****" + key[-4:]) if len(key) > 8 else "****"
    return {
        "enabled": bool(cfg.get("enabled")),
        "provider": cfg.get("provider"),
        "base_url": cfg.get("base_url"),
        "model": cfg.get("model"),
        "api_key_masked": masked,
        "has_key": bool(key),
    }


def update_settings(data):
    """更新设置：Key 留空表示保持不变。"""
    cfg = load_config()
    if "enabled" in data:
        cfg["enabled"] = bool(data["enabled"])
    if data.get("provider"):
        cfg["provider"] = str(data["provider"]).strip()
    if data.get("base_url"):
        cfg["base_url"] = str(data["base_url"]).strip().rstrip("/")
    if data.get("model"):
        cfg["model"] = str(data["model"]).strip()
    if data.get("api_key"):
        cfg["api_key"] = str(data["api_key"]).strip()
    save_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# 上下文与请求构造
# ---------------------------------------------------------------------------
def _build_context(top_chunks):
    parts = []
    for i, c in enumerate(top_chunks, 1):
        parts.append("[资料%d]《%s》%s：\n%s" % (i, c["source"], c["title"], c["text"]))
    return "\n\n".join(parts)


def _build_request(question, top_chunks, stream):
    cfg = load_config()
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": "你是严谨的《人工智能导论》课程答疑助手。"},
            {"role": "user", "content": PROMPT_TEMPLATE % (_build_context(top_chunks), question)},
        ],
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
        "stream": stream,
    }
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + cfg["api_key"],
        },
    )
    return req


# ---------------------------------------------------------------------------
# 生成接口
# ---------------------------------------------------------------------------
def friendly_error(e):
    """把异常转成用户可读的中文提示。"""
    if isinstance(e, urllib.error.HTTPError):
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        msg = ""
        try:
            msg = json.loads(body).get("error", {}).get("message", "")
        except Exception:
            pass
        if msg:
            return "HTTP %d：%s" % (e.code, msg)
        if e.code == 401:
            return "API Key 无效（401）：请检查 Key 是否完整、正确"
        if e.code == 402:
            return "账户余额不足（402）：请前往平台充值"
        if e.code == 429:
            return "请求过于频繁或额度受限（429）：请稍后再试"
        return "HTTP 错误 %d" % e.code
    if isinstance(e, urllib.error.URLError):
        return "网络连接失败：无法访问 %s（请检查网络/代理）" % getattr(e, "reason", e)
    return "%s：%s" % (type(e).__name__, str(e)[:200])


def generate_answer(question, top_chunks):
    """一次性生成答案；未配置 Key 时返回 None（调用方回退抽取式）。"""
    if not is_enabled():
        return None
    req = _build_request(question, top_chunks, stream=False)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"].strip()
    return content or None


def stream_answer(question, top_chunks):
    """流式生成：逐段 yield 增量文本；未配置 Key 时抛出异常。"""
    if not is_enabled():
        raise RuntimeError("未启用大模型 API（请先在设置面板中配置 API Key）")
    req = _build_request(question, top_chunks, stream=True)
    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except Exception:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if text:
                yield text


def test_connection():
    """用当前配置发一次最小请求，返回 (是否成功, 中文说明)。"""
    if not is_enabled():
        return False, "未启用或未填写 API Key"
    payload = {
        "model": load_config()["model"],
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 8,
        "stream": False,
    }
    cfg = load_config()
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + cfg["api_key"]},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                return True, "连接成功，模型可用"
            return False, "HTTP %d" % resp.status
    except urllib.error.HTTPError as e:
        return False, friendly_error(e)
    except Exception as e:
        return False, friendly_error(e)
