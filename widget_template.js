(function () {
  'use strict';

  // ── Config from the page ────────────────────────────────────────────────────
  const cfg    = window.HireflowBot || {};
  const botId  = cfg.botId;
  const token  = cfg.token;
  const API    = 'https://kbingestion-production.up.railway.app';

  if (!botId || !token) {
    console.warn('[HireflowBot] Missing botId or token.');
    return;
  }

  // ── Session ID (persisted in localStorage per visitor) ──────────────────────
  const SESSION_KEY = `hf_session_${botId}`;
  let sessionId = localStorage.getItem(SESSION_KEY);
  if (!sessionId) {
    sessionId = crypto.randomUUID ? crypto.randomUUID()
      : Math.random().toString(36).slice(2) + Date.now();
    localStorage.setItem(SESSION_KEY, sessionId);
  }

  let conversationId = null;
  let botConfig      = {};

  // ── Load bot config ─────────────────────────────────────────────────────────
  async function loadConfig() {
    try {
      const res = await fetch(`${API}/widget-config/${botId}?token=${token}`);
      if (!res.ok) return;
      botConfig = await res.json();
      applyBranding();
      showGreeting();
    } catch (e) {
      console.warn('[HireflowBot] Could not load config:', e);
    }
  }

  // ── Inject CSS ───────────────────────────────────────────────────────────────
  function injectStyles(primaryColor) {
    const color = primaryColor || '#1E2761';
    const style = document.createElement('style');
    style.textContent = `
      #hf-bubble {
        position: fixed; bottom: 24px; right: 24px; z-index: 99999;
        width: 56px; height: 56px; border-radius: 50%;
        background: ${color}; border: none; cursor: pointer;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
        display: flex; align-items: center; justify-content: center;
        transition: transform 0.2s;
      }
      #hf-bubble:hover { transform: scale(1.08); }
      #hf-bubble img { width: 32px; height: 32px; border-radius: 50%; object-fit: cover; }
      #hf-bubble svg { width: 28px; height: 28px; fill: #fff; }
      #hf-window {
        position: fixed; bottom: 92px; right: 24px; z-index: 99999;
        width: 360px; height: 520px;
        background: #fff; border-radius: 16px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.18);
        display: flex; flex-direction: column;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 14px; overflow: hidden;
        transform: scale(0.9) translateY(16px); opacity: 0;
        transition: transform 0.2s ease, opacity 0.2s ease;
        pointer-events: none;
      }
      #hf-window.open {
        transform: scale(1) translateY(0); opacity: 1; pointer-events: all;
      }
      #hf-header {
        background: ${color}; color: #fff;
        padding: 16px; display: flex; align-items: center; gap: 10px;
        flex-shrink: 0;
      }
      #hf-header img { width: 36px; height: 36px; border-radius: 50%; object-fit: cover; }
      #hf-header-text strong { display: block; font-size: 15px; }
      #hf-header-text span { font-size: 12px; opacity: 0.8; }
      #hf-messages {
        flex: 1; overflow-y: auto; padding: 16px;
        display: flex; flex-direction: column; gap: 10px;
      }
      .hf-msg {
        max-width: 84%; padding: 10px 13px; border-radius: 14px;
        line-height: 1.5; word-break: break-word;
      }
      .hf-msg.user {
        align-self: flex-end; background: ${color}; color: #fff;
        border-bottom-right-radius: 4px;
      }
      .hf-msg.bot {
        align-self: flex-start; background: #f3f4f6; color: #1a1a2e;
        border-bottom-left-radius: 4px;
      }
      .hf-msg.typing { opacity: 0.6; font-style: italic; }
      .hf-sources {
        font-size: 11px; color: #888; margin-top: 4px;
        align-self: flex-start; padding: 0 4px;
      }
      #hf-input-area {
        padding: 12px; border-top: 1px solid #eee;
        display: flex; gap: 8px; flex-shrink: 0;
      }
      #hf-input {
        flex: 1; border: 1px solid #ddd; border-radius: 24px;
        padding: 8px 14px; font-size: 14px; outline: none;
        transition: border-color 0.2s;
      }
      #hf-input:focus { border-color: ${color}; }
      #hf-send {
        background: ${color}; color: #fff; border: none;
        border-radius: 50%; width: 36px; height: 36px;
        cursor: pointer; display: flex; align-items: center;
        justify-content: center; flex-shrink: 0;
      }
      #hf-send svg { width: 16px; height: 16px; fill: #fff; }
      #hf-powered {
        text-align: center; font-size: 11px; color: #bbb;
        padding: 4px 0 8px; flex-shrink: 0;
      }
    `;
    document.head.appendChild(style);
  }

  // ── Build DOM ────────────────────────────────────────────────────────────────
  function buildUI() {
    // Bubble
    const bubble = document.createElement('button');
    bubble.id = 'hf-bubble';
    bubble.innerHTML = `<svg viewBox="0 0 24 24"><path d="M12 2C6.477 2 2 6.477 2 12c0 1.89.525 3.66 1.438 5.168L2 22l4.832-1.438A9.96 9.96 0 0012 22c5.523 0 10-4.477 10-10S17.523 2 12 2z"/></svg>`;
    document.body.appendChild(bubble);

    // Window
    const win = document.createElement('div');
    win.id = 'hf-window';
    win.innerHTML = `
      <div id="hf-header">
        <img id="hf-avatar" src="" alt="" style="display:none"/>
        <div id="hf-avatar-placeholder" style="width:36px;height:36px;border-radius:50%;background:rgba(255,255,255,0.3);display:flex;align-items:center;justify-content:center;">
          <svg viewBox="0 0 24 24" style="width:20px;height:20px;fill:#fff"><path d="M12 2C6.477 2 2 6.477 2 12c0 1.89.525 3.66 1.438 5.168L2 22l4.832-1.438A9.96 9.96 0 0012 22c5.523 0 10-4.477 10-10S17.523 2 12 2z"/></svg>
        </div>
        <div id="hf-header-text">
          <strong id="hf-bot-name">Assistant</strong>
          <span>Online</span>
        </div>
      </div>
      <div id="hf-messages"></div>
      <div id="hf-input-area">
        <input id="hf-input" type="text" placeholder="Type your question..." autocomplete="off"/>
        <button id="hf-send">
          <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
      </div>
      <div id="hf-powered">Powered by Hireflow</div>
    `;
    document.body.appendChild(win);

    // Toggle
    bubble.addEventListener('click', () => win.classList.toggle('open'));

    // Send on click
    document.getElementById('hf-send').addEventListener('click', sendMessage);

    // Send on Enter
    document.getElementById('hf-input').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
  }

  // ── Apply branding from config ───────────────────────────────────────────────
  function applyBranding() {
    const nameEl   = document.getElementById('hf-bot-name');
    const avatarEl = document.getElementById('hf-avatar');
    const placeholder = document.getElementById('hf-avatar-placeholder');

    if (nameEl && botConfig.name)      nameEl.textContent  = botConfig.name;

    if (avatarEl && botConfig.avatar_url) {
      avatarEl.src              = botConfig.avatar_url;
      avatarEl.style.display    = 'block';
      if (placeholder) placeholder.style.display = 'none';
    }
  }

  // ── Show greeting message ────────────────────────────────────────────────────
  function showGreeting() {
    const greeting = botConfig.greeting_message || 'Hi! How can I help you today?';
    appendMessage('bot', greeting);
  }

  // ── Append message to chat ───────────────────────────────────────────────────
  function appendMessage(role, text, sources) {
    const msgs  = document.getElementById('hf-messages');
    const div   = document.createElement('div');
    div.className = `hf-msg ${role}`;
    div.textContent = text;
    msgs.appendChild(div);

    if (sources && sources.length > 0) {
      const src = document.createElement('div');
      src.className = 'hf-sources';
      src.textContent = `Sources: ${sources.join(', ')}`;
      msgs.appendChild(src);
    }

    msgs.scrollTop = msgs.scrollHeight;
    return div;
  }

  // ── Send message ─────────────────────────────────────────────────────────────
  async function sendMessage() {
    const input    = document.getElementById('hf-input');
    const question = input.value.trim();
    if (!question) return;

    input.value   = '';
    input.disabled = true;

    appendMessage('user', question);

    const typing = appendMessage('bot', 'Thinking...', null);
    typing.classList.add('typing');

    try {
      const res = await fetch(`${API}/widget-query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          bot_id:          botId,
          token:           token,
          question:        question,
          session_id:      sessionId,
          conversation_id: conversationId,
        }),
      });

      const data = await res.json();

      typing.remove();

      if (!res.ok) {
        appendMessage('bot', data.detail || 'Sorry, something went wrong. Please try again.');
      } else {
        appendMessage('bot', data.answer, data.sources);
        conversationId = data.conversation_id;
      }
    } catch (e) {
      typing.remove();
      appendMessage('bot', 'Connection error. Please check your internet and try again.');
    } finally {
      input.disabled = false;
      input.focus();
    }
  }

  // ── Init ─────────────────────────────────────────────────────────────────────
  async function init() {
    injectStyles(cfg.primaryColor);
    buildUI();
    await loadConfig();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
