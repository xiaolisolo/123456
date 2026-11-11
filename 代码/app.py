# -*- coding: utf-8 -*-
"""
app.py —— 课程答疑机器人 Web 应用（单文件、零依赖）

运行方式：
    python app.py [端口]        # 默认端口 8000
    run.bat                     # Windows 双击运行

浏览器打开 http://127.0.0.1:8000 即可使用。
系统仅依赖 Python 标准库（http.server + 自带 RAG 引擎 rag.py），
不需要 pip 安装任何包，也不需要联网。

两种回答模式（可无缝切换）：
    1. 本地抽取式（默认）：完全离线，从检索片段中抽取答案；
    2. 大模型生成式（可选）：在左侧“大模型 API 设置”中填入 DeepSeek 等
       OpenAI 兼容服务的 API Key，保存后自动切换为生成式回答（流式输出），
       调用失败时自动回退到本地抽取式。
"""

import json
import os
import re
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import llm_api
from rag import (KnowledgeBase, answer_question, build_related,
                 extractive_answer, retrieve)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
KB = KnowledgeBase(DATA_DIR)

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>课程答疑机器人 · RAG</title>
<style>
:root{
  --bg:#f3f5fb; --card:#ffffff; --primary:#4f46e5; --primary2:#7c3aed;
  --ink:#1e2433; --sub:#6b7280; --line:#e5e7eb; --ok:#059669; --warn:#d97706;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"Segoe UI","Microsoft YaHei","PingFang SC",sans-serif;background:var(--bg);color:var(--ink);height:100vh;display:flex;flex-direction:column}
header{background:linear-gradient(90deg,var(--primary),var(--primary2));color:#fff;padding:12px 24px;display:flex;align-items:center;gap:12px;box-shadow:0 2px 8px rgba(79,70,229,.25)}
header h1{font-size:19px;font-weight:600}
header .tag{font-size:12px;background:rgba(255,255,255,.2);border-radius:999px;padding:3px 10px}
main{flex:1;display:grid;grid-template-columns:320px 1fr;gap:16px;padding:16px;min-height:0}
.panel{background:var(--card);border-radius:14px;box-shadow:0 1px 4px rgba(30,36,51,.07);display:flex;flex-direction:column;min-height:0;overflow:hidden}
.panel h2{font-size:14px;padding:14px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:8px}
.kb-inner{padding:14px 16px;overflow:auto}
.stats{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px}
.stat{background:#f8f9fe;border:1px solid var(--line);border-radius:10px;text-align:center;padding:10px 4px}
.stat b{font-size:18px;color:var(--primary);display:block}
.stat span{font-size:11px;color:var(--sub)}
.upload-box{border:2px dashed #c7cbe0;border-radius:10px;padding:16px;text-align:center;color:var(--sub);cursor:pointer;transition:.2s;margin-bottom:14px;font-size:13px}
.upload-box:hover{border-color:var(--primary);color:var(--primary);background:#f8f9ff}
.file{display:flex;justify-content:space-between;align-items:center;padding:8px 10px;border:1px solid var(--line);border-radius:8px;margin-bottom:8px;font-size:12.5px}
.file .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:180px}
.file .del{cursor:pointer;color:#ef4444;font-size:12px}
.settings{margin-top:14px;border:1px solid var(--line);border-radius:10px;background:#fbfbfe}
.settings summary{cursor:pointer;padding:10px 12px;font-size:13px;font-weight:600;color:var(--primary);user-select:none}
.settings[open] summary{border-bottom:1px solid var(--line)}
.set{padding:12px}
.set label{display:block;font-size:12px;color:var(--sub);margin:8px 0 4px}
.set input,.set select{width:100%;border:1px solid var(--line);border-radius:8px;padding:7px 10px;font-size:13px;outline:none}
.set input:focus,.set select:focus{border-color:var(--primary)}
.set .row{display:flex;align-items:center;gap:6px;margin-top:10px;font-size:13px;color:var(--ink);cursor:pointer}
.set .row input{width:auto}
.set button{width:100%;margin-top:12px;border:none;background:linear-gradient(90deg,var(--primary),var(--primary2));color:#fff;border-radius:8px;padding:9px;font-size:13.5px;cursor:pointer}
.set .st{margin-top:10px;font-size:12px;color:var(--ok);text-align:center;white-space:pre-wrap}
.set .st.err{color:#ef4444}
.chat{display:flex;flex-direction:column}
.msgs{flex:1;overflow:auto;padding:18px}
.msg{max-width:82%;margin-bottom:14px;display:flex;flex-direction:column}
.msg .bubble{padding:10px 14px;border-radius:12px;line-height:1.65;font-size:14.5px;white-space:pre-wrap}
.msg.user{align-self:flex-end;align-items:flex-end}
.msg.user .bubble{background:linear-gradient(90deg,var(--primary),var(--primary2));color:#fff;border-bottom-right-radius:4px}
.msg.bot{align-self:flex-start;align-items:flex-start}
.msg.bot .bubble{background:#f4f5fa;border:1px solid var(--line);border-bottom-left-radius:4px}
.msg.bot .bubble .cursor{display:inline-block;width:8px;height:16px;background:var(--primary);vertical-align:-2px;animation:blink 1s steps(1) infinite}
@keyframes blink{50%{opacity:0}}
.mode{font-size:11px;color:var(--sub);margin-top:5px}
.mode.llm{color:var(--ok)}
.mode.fallback{color:var(--warn)}
.src{margin-top:6px;font-size:12px;color:var(--ok);background:#ecfdf5;border:1px solid #a7f3d0;border-radius:8px;padding:4px 10px}
.detail{margin-top:6px;font-size:12px;color:var(--sub)}
.detail summary{cursor:pointer;color:var(--primary);user-select:none}
.chunk{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin-top:8px;background:#fff}
.chunk .ci{display:flex;justify-content:space-between;font-size:12px;color:var(--sub);margin-bottom:4px}
.chunk p{font-size:12.5px;color:#374151;max-height:64px;overflow:hidden}
.bar{height:5px;background:#eceef5;border-radius:3px;margin-top:6px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--primary),var(--primary2))}
.inputbar{border-top:1px solid var(--line);padding:12px;display:flex;gap:10px;background:#fff}
.inputbar input{flex:1;border:1px solid var(--line);border-radius:999px;padding:10px 16px;font-size:14px;outline:none}
.inputbar input:focus{border-color:var(--primary)}
.inputbar button{border:none;background:linear-gradient(90deg,var(--primary),var(--primary2));color:#fff;border-radius:999px;padding:10px 22px;font-size:14px;cursor:pointer}
.inputbar button:disabled{opacity:.5;cursor:default}
.hint{font-size:12px;color:var(--sub);padding:0 2px 6px}
@media (max-width:860px){main{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
  <h1>🎓 课程答疑机器人</h1>
  <span class="tag">RAG 检索增强 · 本地运行 · 零依赖</span>
  <span class="tag" id="modeTag">…</span>
</header>
<main>
  <aside class="panel">
    <h2>📚 知识库管理</h2>
    <div class="kb-inner">
      <div class="stats" id="stats"></div>
      <div class="upload-box" id="uploadBox">⬆️ 点击上传课程资料<br><span style="font-size:11px">支持 .txt / .md / .docx</span></div>
      <input type="file" id="fileInput" style="display:none" multiple>
      <div id="fileList"></div>
      <details class="settings" id="settingsBox" __OPEN__>
        <summary>⚙️ 大模型 API 设置</summary>
        <div class="set">
          <label>服务商</label>
          <select id="provider">
            <option value="deepseek">DeepSeek</option>
            <option value="qwen">通义千问（OpenAI 兼容）</option>
            <option value="zhipu">智谱 GLM</option>
            <option value="moonshot">Moonshot（Kimi）</option>
            <option value="custom">自定义</option>
          </select>
          <label>Base URL</label>
          <input id="baseUrl" placeholder="https://api.deepseek.com">
          <label>模型名称</label>
          <input id="model" placeholder="deepseek-chat">
          <label>API Key</label>
          <input id="apiKey" type="password" placeholder="sk-...（留空表示不修改）" autocomplete="off">
          <label class="row"><input type="checkbox" id="enabled"> 启用大模型生成式回答</label>
          <button id="saveSettings">保存并测试连接</button>
          <div class="st" id="settingsStatus"></div>
        </div>
      </details>
    </div>
  </aside>
  <section class="panel chat">
    <div class="msgs" id="msgs"></div>
    <div class="inputbar">
      <div style="flex:1">
        <div class="hint">试试：什么是机器学习？/ 监督学习与无监督学习的区别？/ 什么是过拟合？</div>
        <input id="q" placeholder="输入你的问题，回车发送" autocomplete="off">
      </div>
      <button id="send">发送</button>
    </div>
  </section>
</main>
<script>
var INITIAL = __INITIAL__;
var LLM_ENABLED = false;
var PRESETS = {
  deepseek: {base:'https://api.deepseek.com', model:'deepseek-chat'},
  qwen:     {base:'https://dashscope.aliyuncs.com/compatible-mode/v1', model:'qwen-plus'},
  zhipu:    {base:'https://open.bigmodel.cn/api/paas/v4', model:'glm-4-flash'},
  moonshot: {base:'https://api.moonshot.cn/v1', model:'moonshot-v1-8k'}
};
var elMsgs = document.getElementById('msgs');
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function addUser(t){elMsgs.insertAdjacentHTML('beforeend','<div class="msg user"><div class="bubble">'+esc(t)+'</div></div>');scrollDown()}
function makeBotBubble(){
  var wrap=document.createElement('div');wrap.className='msg bot';
  var b=document.createElement('div');b.className='bubble';
  var cur=document.createElement('span');cur.className='cursor';
  b.appendChild(cur);
  wrap.appendChild(b);
  elMsgs.appendChild(wrap);
  return wrap;
}
function fillBotMeta(wrap,res){
  var html='';
  if(res.mode==='llm'){html+='<div class="mode llm">🤖 大模型生成 · 基于检索资料 · 耗时 '+res.elapsed_ms+' ms</div>'}
  if(res.mode==='extractive'){html+='<div class="mode">📄 本地抽取式回答 · 耗时 '+res.elapsed_ms+' ms</div>'}
  if(res.note){html+='<div class="mode fallback">⚠️ '+esc(res.note)+'</div>'}
  if(res.primary_source){html+='<div class="src">📍 答案来源：'+esc(res.primary_source)+'</div>'}
  if(res.related && res.related.length){
    var max=Math.max.apply(null,res.related.map(function(x){return x.score}))||1;
    html+='<details class="detail"><summary>🔍 查看检索过程（Top-'+res.related.length+' 相关片段）</summary>';
    res.related.forEach(function(c){
      html+='<div class="chunk"><div class="ci"><b>《'+esc(c.source)+'》'+esc(c.title)+'</b><span>BM25 得分 '+c.score+'</span></div>'
          +'<p>'+esc(c.snippet)+'…</p><div class="bar"><i style="width:'+Math.round(c.score/max*100)+'%"></i></div></div>';
    });
    html+='</details>';
  }
  wrap.insertAdjacentHTML('beforeend',html);
}
function addBot(res){
  var wrap=makeBotBubble();
  var b=wrap.querySelector('.bubble');
  b.removeChild(b.querySelector('.cursor'));
  b.textContent=res.answer;
  fillBotMeta(wrap,res);scrollDown();
}
function scrollDown(){elMsgs.scrollTop=elMsgs.scrollHeight}
function plainAsk(t){
  fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:t})})
   .then(function(r){return r.json()}).then(addBot)
   .catch(function(e){addBot({answer:'请求失败：'+e.message})})
   .finally(function(){enableSend(true)});
}
function streamAsk(t){
  var wrap=makeBotBubble();
  var b=wrap.querySelector('.bubble');
  var cur=b.querySelector('.cursor');
  fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:t,stream:true})})
   .then(function(resp){
     var reader=resp.body.getReader();
     var dec=new TextDecoder();
     var buf='',full='';
     function pump(){
       reader.read().then(function(r){
         if(r.done){return}
         buf+=dec.decode(r.value,{stream:true});
         var idx;
         while((idx=buf.indexOf('\n\n'))>=0){
           var chunk=buf.slice(0,idx);buf=buf.slice(idx+2);
           chunk.split('\n').forEach(function(line){
             if(line.indexOf('data: ')===0){
               var ev=JSON.parse(line.slice(6));
               if(ev.type==='delta'){full+=ev.text;b.textContent=full;b.appendChild(cur);scrollDown()}
               if(ev.type==='done'){
                 b.removeChild(cur);b.textContent=ev.answer;
                 fillBotMeta(wrap,ev);scrollDown();enableSend(true);
               }
             }
           });
         }
         pump();
       }).catch(function(e){b.removeChild(cur);b.textContent='请求中断：'+e.message;enableSend(true)});
     }
     pump();
   })
   .catch(function(e){b.removeChild(cur);b.textContent='请求失败：'+e.message;enableSend(true)});
}
function enableSend(on){var btn=document.getElementById('send');btn.disabled=!on;if(on)document.getElementById('q').focus()}
function ask(t){
  if(!t)return;
  addUser(t);
  enableSend(false);
  if(LLM_ENABLED){streamAsk(t)}else{plainAsk(t)}
}
function loadKb(){
  fetch('/api/kb').then(function(r){return r.json()}).then(function(d){
    document.getElementById('stats').innerHTML=
      '<div class="stat"><b>'+d.stats.docs+'</b><span>文档数</span></div>'
     +'<div class="stat"><b>'+d.stats.chunks+'</b><span>知识片段</span></div>'
     +'<div class="stat"><b>'+d.stats.tokens+'</b><span>词元数</span></div>';
    LLM_ENABLED=!!d.llm_enabled;
    document.getElementById('modeTag').textContent=d.llm_enabled?('🤖 大模型模式 · '+d.llm_model):'📄 本地抽取式模式';
    var list='';
    (d.docs||[]).forEach(function(f){
      list+='<div class="file"><span class="nm" title="'+esc(f.name)+'">📄 '+esc(f.name)+'（'+f.chunks+' 块）</span>'
           +'<span class="del" onclick="delFile(\''+encodeURIComponent(f.name)+'\')">✕ 删除</span></div>';
    });
    if(!list)list='<div class="hint">知识库为空，请先上传课程资料。</div>';
    document.getElementById('fileList').innerHTML=list;
  });
}
function delFile(name){
  if(!confirm('确定删除该资料？'))return;
  fetch('/api/delete?name='+name,{method:'POST'}).then(loadKb);
}
function upload(files){
  var jobs=[].slice.call(files);
  function next(){
    if(!jobs.length){loadKb();return}
    var f=jobs.shift();
    fetch('/api/upload?filename='+encodeURIComponent(f.name),{method:'POST',body:f})
      .then(function(){next()}).catch(function(){next()});
  }
  next();
}
/* ---- 大模型设置 ---- */
function loadSettings(){
  fetch('/api/settings').then(function(r){return r.json()}).then(function(s){
    document.getElementById('provider').value=s.provider||'deepseek';
    document.getElementById('baseUrl').value=s.base_url||'';
    document.getElementById('model').value=s.model||'';
    document.getElementById('enabled').checked=!!s.enabled;
    if(s.has_key){
      var k=document.getElementById('apiKey');
      k.placeholder='已保存：'+s.api_key_masked+'（留空不修改）';
    }
  });
}
document.getElementById('provider').onchange=function(){
  var p=PRESETS[this.value];
  if(p){document.getElementById('baseUrl').value=p.base;document.getElementById('model').value=p.model}
};
document.getElementById('saveSettings').onclick=function(){
  var st=document.getElementById('settingsStatus');
  st.textContent='正在保存并测试连接…';st.classList.remove('err');
  var body={
    provider:document.getElementById('provider').value,
    base_url:document.getElementById('baseUrl').value.trim(),
    model:document.getElementById('model').value.trim(),
    enabled:document.getElementById('enabled').checked,
    api_key:document.getElementById('apiKey').value.trim(),
    test:true
  };
  fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
   .then(function(r){return r.json()})
   .then(function(s){
     document.getElementById('apiKey').value='';
     var t=s.test||{};
     if(s.enabled&&s.has_key){
       if(t.ok){st.textContent='✅ 已保存，连接测试通过（'+s.model+'）';LLM_ENABLED=true}
       else{st.textContent='❌ 已保存，但连接测试失败：'+t.message;st.classList.add('err');LLM_ENABLED=false}
     }else if(!s.has_key){
       st.textContent='⚠️ 未检测到 API Key，仍为本地抽取式模式';st.classList.add('err');LLM_ENABLED=false
     }else{
       st.textContent='✅ 已保存（未启用，当前为本地抽取式模式）';LLM_ENABLED=false
     }
     loadKb();
   })
   .catch(function(){st.textContent='❌ 保存失败，请重试';st.classList.add('err')});
};
document.getElementById('uploadBox').onclick=function(){document.getElementById('fileInput').click()};
document.getElementById('fileInput').onchange=function(e){upload(e.target.files);e.target.value=''};
document.getElementById('send').onclick=function(){ask(document.getElementById('q').value.trim());document.getElementById('q').value=''};
document.getElementById('q').addEventListener('keydown',function(e){if(e.key==='Enter'){ask(this.value.trim());this.value=''}});
loadKb();
loadSettings();
if(INITIAL){addUser(INITIAL.question);addBot(INITIAL);}
else{addBot({answer:'👋 你好！我是《人工智能导论》课程答疑机器人。\n\n我可以基于左侧知识库中的课程资料回答你的问题，并给出答案来源。\n\n默认使用完全本地的抽取式回答；在左侧“⚙️ 大模型 API 设置”中填入 DeepSeek 等 API Key 后，可切换为大模型生成式回答（流式输出）。\n\n试试问我：\n· 什么是机器学习？\n· 监督学习和无监督学习有什么区别？\n· 什么是卷积神经网络？\n· RAG是什么？'});}
</script>
</body>
</html>
"""

DEBUG_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><title>检索过程可视化 · 课程答疑机器人</title>
<style>
body{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#f3f5fb;color:#1e2433;margin:0;padding:28px}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:20px}
.qbox{background:#fff;border-left:4px solid #4f46e5;border-radius:10px;padding:14px 18px;margin:14px 0;box-shadow:0 1px 4px rgba(30,36,51,.07)}
.qbox b{color:#4f46e5}
.chunk{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 18px;margin-bottom:14px;box-shadow:0 1px 4px rgba(30,36,51,.07)}
.ci{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:#6b7280}
.score{font-weight:700;color:#4f46e5}
p.ct{font-size:14.5px;line-height:1.7;margin:10px 0 8px;color:#374151}
.bar{height:8px;background:#eceef5;border-radius:4px;overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,#4f46e5,#7c3aed)}
.meta{font-size:12px;color:#9ca3af;margin-top:6px}
.empty{background:#fff;border:1px dashed #d1d5db;border-radius:10px;padding:26px;text-align:center;color:#6b7280}
</style>
</head>
<body>
<div class="wrap">
<h1>🔍 检索过程可视化（RAG · Retrieve）</h1>
<div class="qbox">问题：<b>__QUESTION__</b>　|　切词：<span class="meta">__TOKENS__</span>　|　耗时：<b>__ELAPSED__ ms</b></div>
__BODY__
</div>
</body>
</html>
"""


def esc_html(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Handler(BaseHTTPRequestHandler):
    server_version = "CourseQABot/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    # ---------- helpers ----------
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _sse_send(self, obj):
        data = json.dumps(obj, ensure_ascii=False)
        self.wfile.write(("data: " + data + "\n\n").encode("utf-8"))
        self.wfile.flush()

    # ---------- routes ----------
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            initial = None
            q = (qs.get("q") or [""])[0].strip()
            if q:
                res = answer_question(q, KB, llm_api.generate_answer)
                res["question"] = q
                initial = json.dumps(res, ensure_ascii=False)
            open_attr = "open" if (qs.get("panel") or [""])[0] == "settings" else ""
            html = PAGE.replace("__INITIAL__", initial if initial else "null").replace("__OPEN__", open_attr)
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/debug":
            q = (qs.get("q") or [""])[0].strip()
            if not q:
                self._send(200, DEBUG_PAGE.replace("__QUESTION__", "(请在地址栏加 ?q=问题)")
                           .replace("__TOKENS__", "-").replace("__ELAPSED__", "0")
                           .replace("__BODY__", '<div class="empty">请输入问题，例如：/debug?q=什么是监督学习</div>'),
                           "text/html; charset=utf-8")
                return
            top_chunks, elapsed_ms = retrieve(q, KB, 5)
            if top_chunks:
                maxs = top_chunks[0]["score"] or 1
                rows = []
                for i, c in enumerate(top_chunks, 1):
                    rows.append(
                        '<div class="chunk"><div class="ci"><span>Top-%d　《%s》 · %s</span>'
                        '<span class="score">BM25 = %.4f</span></div>'
                        '<p class="ct">%s</p><div class="bar"><i style="width:%d%%"></i></div>'
                        '<div class="meta">片段长度：%d 字　|　词元：%d 个</div></div>'
                        % (i, esc_html(c["source"]), esc_html(c["title"]), c["score"],
                           esc_html(re.sub(r"\s+", " ", c["text"]))[:500],
                           int(c["score"] / maxs * 100), len(c["text"]), len(c["tokens"])))
                body = "".join(rows)
            else:
                body = '<div class="empty">未检索到相关片段（BM25 得分均为 0）</div>'
            from rag import tokenize
            q_tokens = tokenize(q)
            html = (DEBUG_PAGE.replace("__QUESTION__", esc_html(q))
                    .replace("__TOKENS__", esc_html(" ".join(q_tokens[:30]) or "（无有效词元）"))
                    .replace("__ELAPSED__", str(elapsed_ms))
                    .replace("__BODY__", body))
            self._send(200, html, "text/html; charset=utf-8")
        elif path == "/api/settings":
            self._json(llm_api.public_settings())
        elif path == "/api/kb":
            self._json({"docs": KB.list_docs(), "stats": KB.stats(),
                        "llm_enabled": llm_api.is_enabled(),
                        "llm_model": llm_api.load_config().get("model", "")})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"
        if path == "/api/settings":
            try:
                data = json.loads(self._read_body().decode("utf-8"))
                llm_api.update_settings(data)
                result = llm_api.public_settings()
                if data.get("test"):
                    ok, msg = llm_api.test_connection()
                    result["test"] = {"ok": ok, "message": msg}
                self._json(result)
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
        elif path == "/api/ask":
            try:
                data = json.loads(self._read_body().decode("utf-8"))
                question = (data.get("question") or "").strip()
                if not question:
                    return self._json({"ok": False, "error": "问题不能为空"}, 400)
                want_stream = bool(data.get("stream"))
                if want_stream and llm_api.is_enabled():
                    self._stream_ask(question)
                else:
                    self._json(answer_question(question, KB, llm_api.generate_answer))
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
        elif path == "/api/upload":
            name = urllib.parse.unquote((qs.get("filename") or ["document.txt"])[0])
            raw = self._read_body()
            if not raw:
                return self._json({"ok": False, "error": "空文件"}, 400)
            try:
                saved = KB.add_doc(name, raw)
                self._json({"ok": True, "name": saved})
            except Exception as e:
                self._json({"ok": False, "error": str(e)}, 500)
        elif path == "/api/delete":
            name = urllib.parse.unquote((qs.get("name") or [""])[0])
            KB.delete_doc(name)
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def _stream_ask(self, question):
        """SSE 流式问答：检索 -> 大模型流式生成 -> 失败自动回退抽取式。"""
        t0 = time.time()
        top_chunks, _ = retrieve(question, KB)
        self._sse_start()
        if not top_chunks:
            self._sse_send({"type": "done", "mode": "notfound", "answer":
                            "抱歉，在已上传的课程资料中没有检索到与“%s”相关的内容。"
                            "请尝试换一种问法，或先在左侧“知识库管理”中上传更多课程资料。" % question,
                            "related": [], "elapsed_ms": int((time.time() - t0) * 1000)})
            return
        self._sse_send({"type": "start"})
        parts, note, mode = [], None, "llm"
        try:
            for delta in llm_api.stream_answer(question, top_chunks):
                parts.append(delta)
                self._sse_send({"type": "delta", "text": delta})
        except Exception as e:
            note = "大模型调用失败（%s），已自动回退本地抽取式回答。" % llm_api.friendly_error(e)
            mode = "extractive"
            parts = [extractive_answer(question, top_chunks)]
        self._sse_send({
            "type": "done",
            "mode": mode,
            "answer": "".join(parts),
            "note": note,
            "primary_source": "%s · %s" % (top_chunks[0]["source"], top_chunks[0]["title"]),
            "related": build_related(top_chunks),
            "elapsed_ms": int((time.time() - t0) * 1000),
        })


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    # 启动时预热知识库索引（首次问答更快，文件列表即时显示片段数）
    idx, docs = KB.build_index()
    cfg = llm_api.load_config()
    print("=" * 56)
    print("  课程答疑机器人（RAG）已启动")
    print("  请在浏览器打开： http://127.0.0.1:%d" % port)
    print("  知识库目录： %s" % DATA_DIR)
    print("  已加载 %d 份资料，共 %d 个知识片段" % (len(docs), len(idx.chunks)))
    if llm_api.is_enabled():
        print("  大模型模式：已启用（%s · %s）" % (cfg.get("provider"), cfg.get("model")))
    else:
        print("  大模型模式：未启用（本地抽取式）")
    print("  按 Ctrl+C 停止服务")
    print("=" * 56)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
        server.server_close()


if __name__ == "__main__":
    main()
