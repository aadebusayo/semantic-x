/* ── State ───────────────────────────────────────────────── */
var activeChatId   = null;   // current open chat
var sending        = false;
var currentCites   = [];     // citations for the *latest* assistant turn
var turnReactions  = {};     // { turnIndex: 'up'|'down'|'none' }
var currentAudio   = null;   // HTMLAudioElement playing

/* ── DOM refs ────────────────────────────────────────────── */
var thread        = document.getElementById('thread');
var input         = document.getElementById('input');
var sendBtn       = document.getElementById('sendBtn');
var chatList      = document.getElementById('chatList');
var chatTitle     = document.getElementById('chatTitle');
var newChatBtn    = document.getElementById('newChatBtn');
var renameBtn     = document.getElementById('renameBtn');
var docSelect     = document.getElementById('docSelect');
var previewDocBtn = document.getElementById('previewDocBtn');
var selectedDocMeta = document.getElementById('selectedDocMeta');
var toggleCiteBtn = document.getElementById('toggleCiteBtn');
var citeCount     = document.getElementById('citeCount');
var citePanel     = document.getElementById('citePanel');
var citePanelBody = document.getElementById('citePanelBody');
var citePanelLabel= document.getElementById('citePanelLabel');
var citePanelMeta = document.getElementById('citePanelMeta');
var closeCiteBtn  = document.getElementById('closeCiteBtn');
var renameModal   = document.getElementById('renameModal');
var renameInput   = document.getElementById('renameInput');
var renameSave    = document.getElementById('renameSave');
var renameCancel  = document.getElementById('renameCancel');

/* ── API helper ──────────────────────────────────────────── */
async function api(path, options) {
  options = options || {};
  var res = await fetch('/api/v1' + path, Object.assign({
    headers: { 'Content-Type': 'application/json' }
  }, options));
  if (!res.ok) {
    var txt = await res.text();
    throw new Error(res.status + ': ' + txt);
  }
  return res.json();
}

/* ── Markdown renderer ───────────────────────────────────── */
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function inlineRender(line) {
  return escHtml(line)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\[(\d+)\]/g, '<sup class="cite-ref">$1</sup>');
}

function renderMarkdown(raw) {
  if (!raw) return '';
  var paras = raw.split(/\n{2,}/);
  return paras.map(function(para) {
    var lines = para.split('\n').filter(function(l) { return l.trim() !== '' || para.trim() !== l; });
    // Bullet list
    if (lines.length && lines.every(function(l) { return /^[\-\*\u2022]\s/.test(l.trim()); })) {
      return '<ul>' + lines.map(function(l) {
        return '<li>' + inlineRender(l.replace(/^[\-\*\u2022]\s+/, '').trim()) + '</li>';
      }).join('') + '</ul>';
    }
    // Numbered list
    if (lines.length && lines.every(function(l) { return /^\d+[\.\)]\s/.test(l.trim()); })) {
      return '<ol>' + lines.map(function(l) {
        return '<li>' + inlineRender(l.replace(/^\d+[\.\)]\s+/, '').trim()) + '</li>';
      }).join('') + '</ol>';
    }
    return '<p>' + lines.map(inlineRender).join('<br>') + '</p>';
  }).join('');
}

function basename(path) {
  if (!path) return '';
  return String(path).split('/').pop();
}

/* ── Sidebar ─────────────────────────────────────────────── */
function dateLabel(iso) {
  var d = new Date(iso);
  var now = new Date();
  var todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  var yestStart  = new Date(todayStart - 86400000);
  if (d >= todayStart) return 'Today';
  if (d >= yestStart)  return 'Yesterday';
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

async function loadHistory() {
  try {
    var items = await api('/chats?limit=200');
    chatList.innerHTML = '';
    if (!items.length) {
      chatList.innerHTML = '<p style="padding:10px 6px;font-size:0.78rem;color:rgba(255,255,255,0.28)">No chats yet</p>';
      return;
    }
    // Group by day
    var groups = {};
    var order  = [];
    items.forEach(function(item) {
      var label = dateLabel(item.updated_at);
      if (!groups[label]) { groups[label] = []; order.push(label); }
      groups[label].push(item);
    });
    order.forEach(function(label) {
      var grp = document.createElement('div');
      var lbl = document.createElement('div');
      lbl.className = 'hist-group-label';
      lbl.textContent = label;
      grp.appendChild(lbl);
      groups[label].forEach(function(item) {
        var el = document.createElement('div');
        el.className = 'chat-item' + (item.chat_id === activeChatId ? ' active' : '');
        el.dataset.chatId = item.chat_id;
        el.textContent = item.title || 'New chat';
        el.title       = item.title || 'New chat';
        el.addEventListener('click', function() { openChat(item.chat_id, item.title); });
        grp.appendChild(el);
      });
      chatList.appendChild(grp);
    });
  } catch (e) {
    console.error('loadHistory', e);
  }
}

function markActive(chatId) {
  chatList.querySelectorAll('.chat-item').forEach(function(el) {
    el.classList.toggle('active', el.dataset.chatId === chatId);
  });
}

async function openChat(chatId, knownTitle) {
  activeChatId = chatId;
  markActive(chatId);
  chatTitle.textContent = knownTitle || 'Loading\u2026';
  clearThread();
  turnReactions = {};
  hideCitePanel();

  try {
    var t = await api('/chats/' + encodeURIComponent(chatId));
    chatTitle.textContent = t.title || 'Chat';
    clearThread();
    var turns = t.turns || [];
    turns.forEach(function(turn, i) {
      appendBubble(turn.role, turn.content, [], i, turn.reaction || 'none');
    });
    showRenameBtn();
    scrollBottom();
  } catch (e) {
    appendBubble('assistant', '\u26a0\ufe0f Could not load transcript.', [], 0, 'none');
  }
}

function getSelectedDocument() {
  if (!docSelect || !docSelect.value) return null;
  var option = docSelect.options[docSelect.selectedIndex];
  return {
    id: docSelect.value,
    name: option && option.dataset && option.dataset.name ? option.dataset.name : basename(docSelect.value)
  };
}

function updateSelectedDocumentMeta() {
  var doc = getSelectedDocument();
  if (!doc) {
    selectedDocMeta.textContent = 'No document selected.';
    selectedDocMeta.title = '';
    previewDocBtn.disabled = true;
    return;
  }
  selectedDocMeta.textContent = 'Selected document: ' + doc.name;
  selectedDocMeta.title = 'Storage id: ' + doc.id;
  previewDocBtn.disabled = false;
}

async function loadDocuments() {
  if (!docSelect) return;
  try {
    var docs = await api('/documents?limit=200');
    docSelect.innerHTML = '<option value="">All documents</option>';
    docs.forEach(function(doc) {
      var option = document.createElement('option');
      option.value = doc.id;
      option.dataset.name = doc.name || basename(doc.id);
      option.textContent = doc.name || basename(doc.id);
      docSelect.appendChild(option);
    });
  } catch (e) {
    console.warn('loadDocuments', e);
  }
  updateSelectedDocumentMeta();
}

/* ── Thread ──────────────────────────────────────────────── */
function clearThread() {
  thread.innerHTML = '';
}

function showWelcome() {
  thread.innerHTML =
    '<div class="welcome" id="welcome">' +
      '<div class="welcome-icon">&#128269;</div>' +
      '<h2>Ask anything</h2>' +
      '<p>Search your documents or ask a general question.</p>' +
    '</div>';
}

function appendBubble(role, text, citations, turnIndex, reaction) {
  var welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();

  var row = document.createElement('div');
  row.className = 'msg-row ' + role;

  var bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (role === 'assistant') {
    bubble.innerHTML = renderMarkdown(text);

    // Action bar
    var actions = document.createElement('div');
    actions.className = 'msg-actions';

    // Thumbs up
    var upBtn = document.createElement('button');
    upBtn.className = 'action-btn' + (reaction === 'up' ? ' active-up' : '');
    upBtn.title = 'Helpful';
    upBtn.innerHTML = '&#128077;';
    upBtn.addEventListener('click', function() { doReaction(turnIndex, 'up', upBtn, dnBtn); });

    // Thumbs down
    var dnBtn = document.createElement('button');
    dnBtn.className = 'action-btn' + (reaction === 'down' ? ' active-dn' : '');
    dnBtn.title = 'Not helpful';
    dnBtn.innerHTML = '&#128078;';
    dnBtn.addEventListener('click', function() { doReaction(turnIndex, 'down', upBtn, dnBtn); });

    // Copy
    var cpBtn = document.createElement('button');
    cpBtn.className = 'action-btn';
    cpBtn.title = 'Copy';
    cpBtn.innerHTML = '&#128203;';
    cpBtn.addEventListener('click', function() {
      navigator.clipboard.writeText(text).then(function() {
        cpBtn.innerHTML = '&#10003;';
        setTimeout(function() { cpBtn.innerHTML = '&#128203;'; }, 1500);
      });
    });

    // TTS
    var ttsBtn = document.createElement('button');
    ttsBtn.className = 'action-btn';
    ttsBtn.title = 'Listen';
    ttsBtn.innerHTML = '&#128266;';
    ttsBtn.addEventListener('click', function() { playTTS(turnIndex, ttsBtn); });

    actions.appendChild(upBtn);
    actions.appendChild(dnBtn);
    actions.appendChild(cpBtn);
    actions.appendChild(ttsBtn);

    row.appendChild(bubble);
    row.appendChild(actions);
  } else {
    bubble.textContent = text;
    row.appendChild(bubble);
  }

  thread.appendChild(row);

  // Update cite panel if this turn has citations
  if (role === 'assistant' && citations && citations.length) {
    currentCites = citations;
    var usedCount = updateCitePanel(citations, text);
    showCiteToggle(usedCount);
  }

  return row;
}

function appendThinking() {
  var row = document.createElement('div');
  row.id = 'thinking-row';
  row.className = 'msg-row assistant';
  var b = document.createElement('div');
  b.className = 'bubble thinking';
  b.textContent = 'Thinking\u2026';
  row.appendChild(b);
  thread.appendChild(row);
  scrollBottom();
}

function removeThinking() {
  var el = document.getElementById('thinking-row');
  if (el) el.remove();
}

function scrollBottom() {
  thread.scrollTop = thread.scrollHeight;
}

/* ── Per-message actions ─────────────────────────────────── */
async function doReaction(turnIndex, reaction, upBtn, dnBtn) {
  if (!activeChatId) return;
  // Toggle off if same
  var current = turnReactions[turnIndex] || 'none';
  var newReaction = current === reaction ? 'none' : reaction;
  turnReactions[turnIndex] = newReaction;

  upBtn.className = 'action-btn' + (newReaction === 'up'   ? ' active-up' : '');
  dnBtn.className = 'action-btn' + (newReaction === 'down' ? ' active-dn' : '');

  try {
    await api('/chats/' + encodeURIComponent(activeChatId) + '/turns/' + turnIndex + '/reaction', {
      method: 'POST',
      body: JSON.stringify({ reaction: newReaction })
    });
  } catch (e) {
    console.warn('reaction save failed', e);
  }
}

async function playTTS(turnIndex, btn) {
  if (!activeChatId) return;
  if (currentAudio) { currentAudio.pause(); currentAudio = null; btn.innerHTML = '&#128266;'; return; }
  btn.innerHTML = '\u23f3';
  try {
    var res = await fetch('/api/v1/chats/' + encodeURIComponent(activeChatId) + '/turns/' + turnIndex + '/audio');
    if (!res.ok) { btn.innerHTML = '&#128266;'; return; }
    var blob = await res.blob();
    var url  = URL.createObjectURL(blob);
    currentAudio = new Audio(url);
    currentAudio.play();
    btn.innerHTML = '\u23f9';
    currentAudio.onended = function() {
      btn.innerHTML = '&#128266;';
      currentAudio = null;
      URL.revokeObjectURL(url);
    };
  } catch (e) {
    console.warn('TTS failed', e);
    btn.innerHTML = '&#128266;';
  }
}

/* ── Citation panel ──────────────────────────────────────── */
function showCiteToggle(n) {
  citeCount.textContent = n;
  toggleCiteBtn.style.display = '';
}

function hideCiteToggle() {
  toggleCiteBtn.style.display = 'none';
  citeCount.textContent = '0';
}

function updateCitePanel(citations, answerText, options) {
  options = options || {};
  // Collect which [N] indices actually appear in the answer text
  var usedIndices = {};
  if (answerText) {
    var pat = /\[(\d+)\]/g, m;
    while ((m = pat.exec(answerText)) !== null) {
      usedIndices[parseInt(m[1], 10)] = true;
    }
  }
  // Filter to only cited entries; fall back to all if none matched (e.g. no markers)
  var filtered = citations.filter(function(_, i) {
    return !answerText || usedIndices[i + 1];
  });
  if (filtered.length === 0) filtered = citations; // safety: show all if filter removes everything

  citePanelLabel.textContent = options.label || (filtered.length + ' Document' + (filtered.length !== 1 ? 's' : '') + ' Referenced');
  citePanelMeta.textContent = options.meta || 'Assistant citations and document previews appear here.';
  citePanelBody.innerHTML = '';
  filtered.forEach(function(c) {
    var origIdx = citations.indexOf(c);
    var card = document.createElement('div');
    card.className = 'cite-card';
    var excerpt = (c.excerpt || '').replace(/<[^>]+>/g, '');
    var documentName = c.document_name || options.documentName || 'Document';
    var documentId = c.document_id || options.documentId || '';
    card.innerHTML =
      '<div class="cite-card-name">' +
        '<span class="cite-card-num">' + (origIdx + 1) + '</span>' +
        escHtml(documentName) +
      '</div>' +
      (documentId ? '<div class="cite-card-id">' + escHtml(documentId) + '</div>' : '') +
      (excerpt ? '<div class="cite-card-excerpt">' + escHtml(excerpt) + '</div>' : '');
    citePanelBody.appendChild(card);
  });
  return filtered.length;
}

function openCitePanel() {
  citePanel.classList.add('open');
}

function hideCitePanel() {
  citePanel.classList.remove('open');
}

toggleCiteBtn.addEventListener('click', function() {
  if (citePanel.classList.contains('open')) {
    hideCitePanel();
  } else {
    openCitePanel();
  }
});
closeCiteBtn.addEventListener('click', hideCitePanel);
docSelect.addEventListener('change', updateSelectedDocumentMeta);
previewDocBtn.addEventListener('click', function() {
  previewSelectedDocument();
});

/* ── Rename ──────────────────────────────────────────────── */
function showRenameBtn() {
  renameBtn.style.display = '';
}

renameBtn.addEventListener('click', function() {
  renameInput.value = chatTitle.textContent.trim();
  renameModal.style.display = 'flex';
  renameInput.focus();
  renameInput.select();
});

renameCancel.addEventListener('click', function() {
  renameModal.style.display = 'none';
});

renameSave.addEventListener('click', async function() {
  var newTitle = renameInput.value.trim();
  if (!newTitle || !activeChatId) { renameModal.style.display = 'none'; return; }
  try {
    await api('/chats/' + encodeURIComponent(activeChatId) + '/title', {
      method: 'PATCH',
      body: JSON.stringify({ title: newTitle })
    });
    chatTitle.textContent = newTitle;
    renameModal.style.display = 'none';
    await loadHistory();
    markActive(activeChatId);
  } catch (e) {
    alert('Rename failed: ' + e.message);
  }
});

renameInput.addEventListener('keydown', function(e) {
  if (e.key === 'Enter') renameSave.click();
  if (e.key === 'Escape') renameCancel.click();
});

/* ── Composer ────────────────────────────────────────────── */
input.addEventListener('input', function() {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});

input.addEventListener('keydown', function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

sendBtn.addEventListener('click', sendMessage);

async function previewSelectedDocument(explicitDocument) {
  var documentRef = explicitDocument || getSelectedDocument();
  if (!documentRef || sending) return;

  sending = true;
  sendBtn.disabled = true;
  previewDocBtn.disabled = true;

  try {
    var result = await api('/documents/preview', {
      method: 'POST',
      body: JSON.stringify({ document: documentRef, top_k: 5 })
    });
    var citations = result.citations || [];
    var docName = (result.document && result.document.name) || documentRef.name || basename(documentRef.id);
    var docId = (result.document && result.document.id) || documentRef.id;
    var count = updateCitePanel(citations, '', {
      label: citations.length + ' Excerpt' + (citations.length !== 1 ? 's' : '') + ' from ' + docName,
      meta: 'Preview mode: empty input with a selected document opens indexed excerpts without sending a chat message.',
      documentName: docName,
      documentId: docId
    });
    showCiteToggle(count);
    openCitePanel();
  } catch (e) {
    alert('Preview failed: ' + e.message);
  } finally {
    sending = false;
    sendBtn.disabled = false;
    updateSelectedDocumentMeta();
    input.focus();
  }
}

async function sendMessage() {
  var text = input.value.trim();
  if (sending) return;

  // @docId: extraction  –  e.g. type "@docId:abc123 " before the question
  var docId   = null;
  var cleaned = text.replace(/^@docId:(\S+)\s*/, function(_, id) { docId = id; return ''; }).trim();
  var selectedDocument = getSelectedDocument();
  var payloadDocument = selectedDocument || (docId ? { id: docId, name: basename(docId) } : null);

  if (!text && payloadDocument) {
    await previewSelectedDocument(payloadDocument);
    return;
  }

  if (!text) return;
  if (!cleaned) cleaned = text;

  sending = true;
  sendBtn.disabled = true;
  previewDocBtn.disabled = true;
  input.value = '';
  input.style.height = 'auto';

  // Calculate turn index for the *assistant* response we're about to add
  // (user turn = N, assistant turn = N+1)
  var existingTurns = thread.querySelectorAll('.msg-row').length;

  appendBubble('user', cleaned, [], existingTurns, 'none');
  appendThinking();
  scrollBottom();

  try {
    var payload = { message: cleaned, chat_id: activeChatId || undefined };
    if (payloadDocument) payload.document = payloadDocument;

    var result = await api('/chat/messages', { method: 'POST', body: JSON.stringify(payload) });

    removeThinking();

    activeChatId = result.chat_id;
    chatTitle.textContent = result.title || 'Chat';
    showRenameBtn();

    // Turn index in cosmos: we count turns already stored before this exchange
    // The response is the last assistant turn, find its index by counting msg-rows
    var turnIdx = thread.querySelectorAll('.msg-row').length;  // before appending assistant
    appendBubble('assistant', result.answer, result.citations || [], turnIdx, 'none');
    scrollBottom();

    await loadHistory();
    markActive(activeChatId);
  } catch (e) {
    removeThinking();
    appendBubble('assistant', '\u26a0\ufe0f Error: ' + e.message, [], 0, 'none');
    scrollBottom();
  } finally {
    sending = false;
    sendBtn.disabled = false;
    updateSelectedDocumentMeta();
    input.focus();
  }
}

/* ── New chat ────────────────────────────────────────────── */
newChatBtn.addEventListener('click', function() {
  activeChatId = null;
  turnReactions = {};
  currentCites  = [];
  chatTitle.textContent = 'New chat';
  renameBtn.style.display = 'none';
  hideCiteToggle();
  hideCitePanel();
  clearThread();
  showWelcome();
  markActive(null);
  input.focus();
});

/* ── Boot ────────────────────────────────────────────────── */
(async function boot() {
  await loadDocuments();
  await loadHistory();
  input.focus();
}());
