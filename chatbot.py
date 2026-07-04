import os
import json
import uuid
import httpx
import voyageai
from groq import Groq
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import Optional
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

voyage  = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
groq    = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Lovable Supabase (read-only — no service key, use anon key for public bot config)
# Bot config is fetched via Lovable's own API (passed from frontend) or
# stored in our personal Supabase mirror. For now we rely on the frontend
# passing bot config in the request body, keeping Railway fully stateless.

# ── Request shapes ─────────────────────────────────────────────────────────────
class WidgetQueryRequest(BaseModel):
    bot_id:        str
    public_token:  str
    question:      str
    session_id:    str          # random UUID from browser localStorage
    # Bot config passed from frontend (Railway has no access to Lovable's DB)
    bot_config: dict = {}       # {system_prompt, name, primary_color, greeting_message}
    # Document IDs to search (passed from frontend after resolving folder→docs)
    document_ids:  Optional[list[str]] = None
    # OR pass folder_ids and Railway resolves to document_ids via personal Supabase
    folder_context: Optional[str] = None  # department/folder name for context


class InternalQueryRequest(BaseModel):
    bot_id:       str
    question:     str
    user_id:      str
    conversation_id: Optional[str] = None
    bot_config:   dict = {}
    document_ids: Optional[list[str]] = None


# ── Core query logic (shared by internal + external) ───────────────────────────
async def run_bot_query(
    question: str,
    bot_config: dict,
    document_ids: Optional[list[str]] = None,
    folder_context: str = ""
) -> dict:
    """
    Core RAG query for a chatbot.
    Searches personal Supabase vector store, optionally filtered by document_ids.
    Returns {answer, sources}.
    """
    # Step 1: Embed the question
    result = voyage.embed([question], model="voyage-3", input_type="query")
    question_embedding = result.embeddings[0]

    # Step 2: Search vector store
    # If document_ids provided, filter to only those documents
    # Otherwise search all documents (bot has access to everything)
    search_params = {
        "query_embedding": question_embedding,
        "match_count": 8,
        "filter_asset_id": None  # we filter by document_id post-retrieval if needed
    }

    search_result = supabase.rpc("match_chunks", search_params).execute()
    chunks = search_result.data or []

    # Filter by document_ids if specified
    if document_ids:
        chunks = [c for c in chunks if c.get("document_id") in document_ids]

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the connected knowledge base. Please try rephrasing your question.",
            "sources": []
        }

    # Step 3: Build context
    context_parts = []
    for chunk in chunks:
        file_name = chunk.get("metadata", {}).get("file_name", "Knowledge Base")
        context_parts.append(f"[Source: {file_name}]\n{chunk['content']}")
    context = "\n\n---\n\n".join(context_parts)

    # Step 4: Build system prompt
    default_system = """You are a helpful assistant. Answer questions based ONLY on the provided documents.
Rules:
1. Answer ONLY from provided documents. Never use outside knowledge.
2. Be direct, clear, and conversational — this is a chat interface.
3. If the answer isn't in the documents, say: "I don't have that information in my knowledge base."
4. Keep answers concise — 2-4 sentences for simple questions, structured lists for complex ones.
5. Mention the source document when relevant."""

    # Merge with admin's custom system prompt
    custom_prompt = bot_config.get("system_prompt", "")
    full_system = f"{default_system}\n\nAdditional instructions: {custom_prompt}" if custom_prompt else default_system

    # Add bot name/personality context
    bot_name = bot_config.get("name", "Assistant")
    full_system = f"You are {bot_name}. {full_system}"

    # Step 5: Generate answer
    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=600,
        messages=[
            {"role": "system", "content": full_system},
            {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
    )

    answer = response.choices[0].message.content

    sources = list(set([
        chunk.get("metadata", {}).get("file_name", "Knowledge Base")
        for chunk in chunks
    ]))

    return {"answer": answer, "sources": sources}


# ── Route 1: External widget query (public token auth) ─────────────────────────
@router.post("/widget-query")
async def widget_query(request: WidgetQueryRequest):
    """
    Called by the embeddable widget on external websites.
    Auth: public_token (not user JWT — anyone can query if they have the token).
    Railway is stateless — conversation saving is handled by Lovable frontend.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        if not request.public_token:
            raise HTTPException(status_code=401, detail="Missing public token")

        # Run the query
        result = await run_bot_query(
            question=request.question,
            bot_config=request.bot_config,
            document_ids=request.document_ids,
            folder_context=request.folder_context or ""
        )

        return {
            "answer":    result["answer"],
            "sources":   result["sources"],
            "bot_id":    request.bot_id,
            "session_id": request.session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"WIDGET-QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ── Route 2: Internal bot query (authenticated users) ─────────────────────────
@router.post("/bot-query")
async def bot_query(request: InternalQueryRequest):
    """
    Called by internal Lovable app for authenticated users.
    Same core logic as widget-query but for logged-in employees.
    Lovable saves the conversation and messages to its own DB.
    """
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        result = await run_bot_query(
            question=request.question,
            bot_config=request.bot_config,
            document_ids=request.document_ids,
        )

        return {
            "answer":          result["answer"],
            "sources":         result["sources"],
            "bot_id":          request.bot_id,
            "conversation_id": request.conversation_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"BOT-QUERY ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# ── Route 3: Widget.js (injectable script for external websites) ───────────────
@router.get("/widget.js")
async def widget_js():
    """
    The embeddable JavaScript that companies paste into their website.
    Injects a fully white-labeled chat bubble using bot config from Lovable.
    """
    RAILWAY_URL = os.getenv(
        "RAILWAY_PUBLIC_DOMAIN",
        "https://kbingestion-production.up.railway.app"
    )
    if not RAILWAY_URL.startswith("http"):
        RAILWAY_URL = f"https://{RAILWAY_URL}"

    js_code = f"""
(function() {{
  'use strict';

  // Config injected by the embedding script tag
  var cfg = window.HireflowBot || {{}};
  var botId       = cfg.botId;
  var token       = cfg.token;
  var apiBase     = cfg.apiBase || '{RAILWAY_URL}';
  var botConfig   = cfg.botConfig || {{}};
  var documentIds = cfg.documentIds || null;

  if (!botId || !token) {{
    console.warn('[HireflowBot] Missing botId or token. Bot will not load.');
    return;
  }}

  // ── Session ID (persisted per browser) ──────────────────────────────────────
  var sessionKey = 'hfbot_session_' + botId;
  var sessionId  = localStorage.getItem(sessionKey);
  if (!sessionId) {{
    sessionId = 'ext_' + Math.random().toString(36).substr(2, 12);
    localStorage.setItem(sessionKey, sessionId);
  }}

  // ── Config defaults ──────────────────────────────────────────────────────────
  var name         = botConfig.name         || 'Assistant';
  var color        = botConfig.primaryColor  || '#1E2761';
  var greeting     = botConfig.greeting      || 'Hi! How can I help you today?';
  var avatarUrl    = botConfig.avatarUrl     || '';
  var avatarLetter = name.charAt(0).toUpperCase();

  // ── Conversation history ─────────────────────────────────────────────────────
  var history = [];

  // ── Styles ───────────────────────────────────────────────────────────────────
  var style = document.createElement('style');
  style.textContent = `
    #hfbot-bubble {{
      position: fixed; bottom: 24px; right: 24px; z-index: 99999;
      width: 56px; height: 56px; border-radius: 50%;
      background: ${{color}}; border: none; cursor: pointer;
      box-shadow: 0 4px 20px rgba(0,0,0,0.25);
      display: flex; align-items: center; justify-content: center;
      color: white; font-size: 22px; font-family: sans-serif; font-weight: bold;
      transition: transform 0.2s; user-select: none;
    }}
    #hfbot-bubble:hover {{ transform: scale(1.08); }}
    #hfbot-window {{
      position: fixed; bottom: 90px; right: 24px; z-index: 99999;
      width: 360px; height: 520px; border-radius: 16px;
      background: #fff; box-shadow: 0 8px 40px rgba(0,0,0,0.18);
      display: none; flex-direction: column; overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    #hfbot-header {{
      background: ${{color}}; color: white; padding: 14px 16px;
      display: flex; align-items: center; gap: 10px;
    }}
    #hfbot-avatar {{
      width: 36px; height: 36px; border-radius: 50%;
      background: rgba(255,255,255,0.25);
      display: flex; align-items: center; justify-content: center;
      font-weight: bold; font-size: 16px; flex-shrink: 0; overflow: hidden;
    }}
    #hfbot-avatar img {{ width: 100%; height: 100%; object-fit: cover; }}
    #hfbot-title {{ font-weight: 600; font-size: 15px; }}
    #hfbot-close {{
      margin-left: auto; background: none; border: none;
      color: rgba(255,255,255,0.8); cursor: pointer; font-size: 20px; padding: 0;
    }}
    #hfbot-messages {{
      flex: 1; overflow-y: auto; padding: 16px; display: flex;
      flex-direction: column; gap: 10px;
    }}
    .hfbot-msg {{ max-width: 85%; padding: 10px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5; }}
    .hfbot-msg-user {{
      background: ${{color}}; color: white;
      align-self: flex-end; border-bottom-right-radius: 4px;
    }}
    .hfbot-msg-bot {{
      background: #f4f4f5; color: #18181b;
      align-self: flex-start; border-bottom-left-radius: 4px;
    }}
    .hfbot-sources {{
      font-size: 11px; color: #888; margin-top: 4px; font-style: italic;
    }}
    .hfbot-typing {{
      display: flex; gap: 4px; padding: 10px 14px;
      background: #f4f4f5; border-radius: 12px; align-self: flex-start; width: fit-content;
    }}
    .hfbot-dot {{
      width: 7px; height: 7px; border-radius: 50%; background: #999;
      animation: hfbot-bounce 1.2s infinite;
    }}
    .hfbot-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .hfbot-dot:nth-child(3) {{ animation-delay: 0.4s; }}
    @keyframes hfbot-bounce {{
      0%,80%,100% {{ transform: translateY(0); }}
      40% {{ transform: translateY(-6px); }}
    }}
    #hfbot-input-area {{
      padding: 12px; border-top: 1px solid #e4e4e7;
      display: flex; gap: 8px; align-items: center;
    }}
    #hfbot-input {{
      flex: 1; border: 1px solid #e4e4e7; border-radius: 8px;
      padding: 8px 12px; font-size: 14px; outline: none;
      font-family: inherit;
    }}
    #hfbot-input:focus {{ border-color: ${{color}}; }}
    #hfbot-send {{
      background: ${{color}}; color: white; border: none;
      border-radius: 8px; padding: 8px 14px; cursor: pointer;
      font-size: 14px; font-weight: 600;
    }}
    #hfbot-send:disabled {{ opacity: 0.5; cursor: not-allowed; }}
  `;
  document.head.appendChild(style);

  // ── DOM ──────────────────────────────────────────────────────────────────────
  var bubble = document.createElement('button');
  bubble.id = 'hfbot-bubble';
  bubble.innerHTML = avatarUrl
    ? '<img src="' + avatarUrl + '" style="width:100%;height:100%;object-fit:cover;border-radius:50%;">'
    : avatarLetter;
  document.body.appendChild(bubble);

  var win = document.createElement('div');
  win.id = 'hfbot-window';
  win.innerHTML = `
    <div id="hfbot-header">
      <div id="hfbot-avatar">
        ${{avatarUrl ? '<img src="' + avatarUrl + '">' : avatarLetter}}
      </div>
      <span id="hfbot-title">${{name}}</span>
      <button id="hfbot-close">✕</button>
    </div>
    <div id="hfbot-messages"></div>
    <div id="hfbot-input-area">
      <input id="hfbot-input" type="text" placeholder="Type a message..." />
      <button id="hfbot-send">Send</button>
    </div>
  `;
  document.body.appendChild(win);

  var messagesEl = win.querySelector('#hfbot-messages');
  var inputEl    = win.querySelector('#hfbot-input');
  var sendBtn    = win.querySelector('#hfbot-send');
  var closeBtn   = win.querySelector('#hfbot-close');
  var isOpen     = false;

  // ── Helpers ──────────────────────────────────────────────────────────────────
  function addMessage(role, text, sources) {{
    var div = document.createElement('div');
    div.className = 'hfbot-msg hfbot-msg-' + role;
    div.textContent = text;

    if (sources && sources.length > 0) {{
      var src = document.createElement('div');
      src.className = 'hfbot-sources';
      src.textContent = 'Sources: ' + sources.join(', ');
      div.appendChild(src);
    }}

    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }}

  function showTyping() {{
    var div = document.createElement('div');
    div.className = 'hfbot-typing';
    div.innerHTML = '<div class="hfbot-dot"></div><div class="hfbot-dot"></div><div class="hfbot-dot"></div>';
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }}

  function toggleWindow() {{
    isOpen = !isOpen;
    win.style.display = isOpen ? 'flex' : 'none';
    if (isOpen && history.length === 0) {{
      addMessage('bot', greeting);
      history.push({{role: 'assistant', content: greeting}});
    }}
    if (isOpen) inputEl.focus();
  }}

  async function sendMessage() {{
    var q = inputEl.value.trim();
    if (!q) return;

    inputEl.value = '';
    sendBtn.disabled = true;
    addMessage('user', q);
    history.push({{role: 'user', content: q}});

    var typing = showTyping();

    try {{
      var body = {{
        bot_id:       botId,
        public_token: token,
        question:     q,
        session_id:   sessionId,
        bot_config:   botConfig,
        document_ids: documentIds
      }};

      var res = await fetch(apiBase + '/widget-query', {{
        method:  'POST',
        headers: {{'Content-Type': 'application/json'}},
        body:    JSON.stringify(body)
      }});

      var data = await res.json();
      typing.remove();

      var answer = data.answer || 'Sorry, I could not process that request.';
      addMessage('bot', answer, data.sources || []);
      history.push({{role: 'assistant', content: answer}});

    }} catch(err) {{
      typing.remove();
      addMessage('bot', 'Something went wrong. Please try again.');
      console.error('[HireflowBot] Error:', err);
    }} finally {{
      sendBtn.disabled = false;
      inputEl.focus();
    }}
  }}

  // ── Events ───────────────────────────────────────────────────────────────────
  bubble.addEventListener('click', toggleWindow);
  closeBtn.addEventListener('click', toggleWindow);
  sendBtn.addEventListener('click', sendMessage);
  inputEl.addEventListener('keydown', function(e) {{
    if (e.key === 'Enter' && !e.shiftKey) {{ e.preventDefault(); sendMessage(); }}
  }});

}})();
""".strip()

    return PlainTextResponse(
        content=js_code,
        media_type="application/javascript",
        headers={"Cache-Control": "public, max-age=300"}
    )
