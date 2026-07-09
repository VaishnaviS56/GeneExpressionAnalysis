const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...(options.headers || {}),
  }

  const response = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
  })

  const text = await response.text()
  const data = text ? JSON.parse(text) : {}

  if (!response.ok) {
    throw new Error(data.detail || 'Request failed')
  }

  return data
}

export function setToken(token) {
  if (token) {
    localStorage.setItem('gea_token', token)
  } else {
    localStorage.removeItem('gea_token')
  }
}

export function getToken() {
  return localStorage.getItem('gea_token') || ''
}

export function authHeaders() {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export function register(payload) {
  return request('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function login(payload) {
  return request('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function me() {
  return request('/api/auth/me', {
    headers: authHeaders(),
  })
}

export function fetchChats() {
  return request('/api/chats', {
    headers: authHeaders(),
  })
}

export function createChat(title, agentType = 'general') {
  return request('/api/chats', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ title, agent_type: agentType }),
  })
}

export function fetchMessages(chatId) {
  return request(`/api/chats/${chatId}/messages`, {
    headers: authHeaders(),
  })
}

export function sendMessage(chatId, content) {
  return request(`/api/chats/${chatId}/messages`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ content }),
  })
}
