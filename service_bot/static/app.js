'use strict';

const $ = (selector) => document.querySelector(selector);
const messages = $('#messages');
const question = $('#question');
const storedId = sessionStorage.getItem('service-demo-session');
const sessionId = storedId && /^[0-9a-f-]{36}$/i.test(storedId) ? storedId : crypto.randomUUID();
sessionStorage.setItem('service-demo-session', sessionId);
let busy = false;
let ready = false;

function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  svg.setAttribute('aria-hidden', 'true');
  use.setAttribute('href', `#i-${name}`);
  svg.append(use);
  return svg;
}

function setTab(name) {
  document.querySelectorAll('[data-tab]').forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    $(`#panel-${tab.dataset.tab}`).hidden = !active;
  });
}

document.querySelectorAll('[data-tab]').forEach((tab, index, tabs) => {
  tab.addEventListener('click', () => setTab(tab.dataset.tab));
  tab.addEventListener('keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
      (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    setTab(tabs[next].dataset.tab);
    tabs[next].focus();
  });
});
$('#telegram-nav').addEventListener('click', () => setTab('setup'));

function appendSafeText(element, text, withLinks = false) {
  if (!withLinks) { element.textContent = text; return; }
  const parts = text.split(/(https:\/\/[^\s<>]+)/g);
  parts.forEach((part) => {
    if (part.startsWith('https://')) {
      const a = document.createElement('a');
      a.href = part;
      a.textContent = part;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      element.append(a);
    } else element.append(document.createTextNode(part));
  });
}

function addMessage(text, type = 'info', response = null) {
  const message = document.createElement('div');
  message.className = `message ${type}`;
  const role = document.createElement('div');
  role.className = 'message-role';
  const labels = {user: 'Вы', answer: 'Ответ из базы знаний', missing: 'Нет информации в базе',
    clarify: 'Нужно уточнение', info: 'Сервисный помощник', error: 'Ошибка соединения'};
  if (type === 'answer') role.append(icon('check'));
  role.append(document.createTextNode(labels[type] || labels.info));
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  appendSafeText(bubble, text, type === 'answer');
  message.append(role, bubble);
  if (response?.source) {
    const source = document.createElement('div');
    source.className = 'message-source';
    source.title = `Запись: ${response.entry_id}`;
    source.append(icon('book'), document.createTextNode(response.source.label));
    message.append(source);
  }
  if (response?.options?.length) {
    const options = document.createElement('div');
    options.className = 'clarification-options';
    response.options.forEach((option) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = option.label;
      button.addEventListener('click', () => sendQuestion(option.label));
      options.append(button);
    });
    message.append(options);
  }
  messages.append(message);
  messages.scrollTop = messages.scrollHeight;
  return message;
}

function setBusy(value) {
  busy = value;
  $('#send-button').disabled = value;
  $('#reset-chat').disabled = value;
  document.querySelectorAll('[data-question], .clarification-options button, .knowledge-card button')
    .forEach((button) => { button.disabled = value; });
  $('#chat-form').setAttribute('aria-busy', String(value));
}

async function api(path, body) {
  const response = await fetch(path, body ? {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  } : {});
  if (!response.ok) throw new Error('API unavailable');
  return response.json();
}

async function sendQuestion(text, showUser = true) {
  text = text.trim();
  if (busy || !text) return;
  if (text.length > 1500) return;
  setTab('dialog');
  if (showUser) addMessage(text, 'user');
  setBusy(true);
  const pending = document.createElement('div');
  pending.className = 'message pending';
  pending.setAttribute('aria-label', 'Ищу ответ в базе знаний');
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';
  for (let i = 0; i < 3; i++) {
    const dot = document.createElement('span');
    dot.className = 'typing-dot';
    bubble.append(dot);
  }
  pending.append(bubble);
  messages.append(pending);
  messages.scrollTop = messages.scrollHeight;
  try {
    const result = await api('/api/chat', {text, session_id: sessionId});
    pending.remove();
    addMessage(result.text, result.status, result);
    if (showUser) question.value = '';
    ready = true;
  } catch {
    pending.remove();
    addMessage('Не удалось получить ответ от сервера. Проверьте соединение и повторите вопрос. Это техническая ошибка, а не ответ из базы знаний.', 'error');
  } finally {
    setBusy(false);
  }
}

$('#chat-form').addEventListener('submit', (event) => {
  event.preventDefault();
  sendQuestion(question.value);
});
question.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendQuestion(question.value);
  }
});
document.querySelectorAll('[data-question]').forEach((button) => {
  button.addEventListener('click', () => sendQuestion(button.dataset.question));
});
$('#reset-chat').addEventListener('click', async () => {
  if (busy) return;
  setBusy(true);
  try {
    await api('/api/reset', {session_id: sessionId});
    messages.replaceChildren();
  } catch {
    addMessage('Не удалось сбросить диалог. Повторите попытку.', 'error');
    setBusy(false);
    return;
  }
  setBusy(false);
  await sendQuestion('/start', false);
});

function entryWord(count) {
  const lastTwo = count % 100;
  if (lastTwo >= 11 && lastTwo <= 14) return 'записей';
  const last = count % 10;
  return last === 1 ? 'запись' : last >= 2 && last <= 4 ? 'записи' : 'записей';
}

async function refreshStatus() {
  try {
    const status = await api('/api/status');
    const connected = status.telegram.state === 'polling';
    $('#live-badge').lastChild.textContent = connected ? 'Telegram подключён' : 'Веб-демо работает';
    $('#chat-status').textContent = `База знаний · ${status.entry_count} ${entryWord(status.entry_count)}`;
    $('#entry-count').textContent = status.entry_count;
    const names = {not_configured: 'Токен не настроен', starting: 'Подключаемся к Telegram',
      polling: 'Бот подключён', retrying: 'Повторяем подключение', error: 'Не удалось подключиться',
      stopped: 'Бот остановлен'};
    $('#telegram-state').textContent = names[status.telegram.state] || 'Telegram';
    $('#telegram-description').textContent = status.telegram.message;
    $('#telegram-dot').classList.toggle('connected', connected);
    const link = $('#telegram-link');
    link.hidden = !connected;
    if (connected) link.href = `https://t.me/${status.telegram.username}`;
  } catch {
    $('#chat-status').textContent = 'Нет соединения с сервером';
    $('#live-badge').lastChild.textContent = 'Сервер недоступен';
    $('#telegram-state').textContent = 'Статус недоступен';
    $('#telegram-description').textContent = 'Проверьте соединение с сервером.';
  }
}

async function loadKnowledge() {
  try {
    const knowledge = await api('/api/knowledge');
    knowledge.entries.forEach((entry, index) => {
      const card = document.createElement('article');
      card.className = 'knowledge-card';
      const number = document.createElement('span');
      number.className = 'card-number';
      number.textContent = `${String(index + 1).padStart(2, '0')} / ПРОВЕРЕННАЯ ЗАПИСЬ`;
      const heading = document.createElement('h3');
      heading.textContent = entry.title;
      const answer = document.createElement('p');
      answer.className = 'card-answer';
      appendSafeText(answer, entry.answer, true);
      const source = document.createElement('p');
      source.className = 'card-source';
      source.textContent = entry.source.label;
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = 'Задать этот вопрос →';
      button.addEventListener('click', () => sendQuestion(entry.examples[0]));
      card.append(number, heading, answer, source, button);
      $('#knowledge-grid').append(card);
    });
  } catch {
    $('#knowledge-grid').textContent = 'База знаний недоступна. Обновите страницу после восстановления соединения.';
  }
}

async function initialize() {
  setTab('dialog');
  await Promise.all([refreshStatus(), loadKnowledge()]);
  if (!ready) await sendQuestion('/start', false);
}
initialize();
setInterval(refreshStatus, 15000);
