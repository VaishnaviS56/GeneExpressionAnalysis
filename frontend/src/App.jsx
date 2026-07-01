import { useEffect, useMemo, useState } from 'react'
import {
  createChat,
  fetchChats,
  fetchMessages,
  getToken,
  login,
  me,
  register,
  sendMessage,
  setToken,
} from './api'

const emptyAuth = {
  email: '',
  password: '',
  display_name: '',
}

function formatNumber(value) {
  if (value === null || value === undefined || value === '') return '—'
  const num = Number(value)
  if (Number.isNaN(num)) return String(value)
  if (num === 0) return '0'
  if (Math.abs(num) < 0.001 || Math.abs(num) >= 1000) {
    return num.toExponential(2)
  }
  return num.toFixed(4)
}

function formatValue(value) {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'number') return formatNumber(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]'
    return value.map((item) => formatValue(item)).join(', ')
  }
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

function renderKvList(data, keyPrefix) {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return null
  const entries = Object.entries(data)
  if (entries.length === 0) return null

  return (
    <div className="trace-kv-list">
      {entries.map(([key, value]) => (
        <div className="trace-kv-row" key={`${keyPrefix}-${key}`}>
          <span className="trace-kv-key">{key}</span>
          <pre className="trace-kv-value">{formatValue(value)}</pre>
        </div>
      ))}
    </div>
  )
}

function renderTechnicalPanel(meta) {
  if (!meta || typeof meta !== 'object') return null

  const degRows = Array.isArray(meta.deg_gene_records) ? meta.deg_gene_records.slice(0, 12) : []
  const enrichrLibs = meta.enrichr && typeof meta.enrichr === 'object' && meta.enrichr.libraries && typeof meta.enrichr.libraries === 'object'
    ? meta.enrichr.libraries
    : {}
  const rwrRows = Array.isArray(meta.rwr_genes) ? meta.rwr_genes.slice(0, 10) : []
  const openTargets = meta.opentargets_result && typeof meta.opentargets_result === 'object' ? meta.opentargets_result : null
  const primeKg = meta.primekg_result && typeof meta.primekg_result === 'object' ? meta.primekg_result : null
  const toolHistory = Array.isArray(meta.tool_history) ? meta.tool_history : []
  const pyvisHtmlPath = typeof meta.pyvis_html_path === 'string' ? meta.pyvis_html_path : ''
  const keggPath = typeof meta.kegg_pathway_path === 'string' ? meta.kegg_pathway_path : ''
  const volcanoPath = typeof meta.volcano_plot_path === 'string' ? meta.volcano_plot_path : ''

  return (
    <section className="technical-panel">
      <div className="technical-header">
        <div className="eyebrow">Technical analysis</div>
        <div className="chat-badge">{meta.analysis_arm || 'general'}</div>
      </div>

      {degRows.length > 0 && (
        <div className="technical-card">
          <h3>DEG genes</h3>
          <div className="technical-table">
            <div className="technical-row technical-head">
              <span>Gene</span>
              <span>log2FC</span>
              <span>p-value</span>
              <span>adj p-value</span>
            </div>
            {degRows.map((row, index) => (
              <div className="technical-row" key={`${row.gene || 'deg'}-${index}`}>
                <span>{row.gene || '—'}</span>
                <span>{formatNumber(row.log2FoldChange)}</span>
                <span>{formatNumber(row.pvalue)}</span>
                <span>{formatNumber(row.pdj)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {Object.entries(enrichrLibs).map(([library, terms]) => (
        Array.isArray(terms) && terms.length > 0 ? (
          <div className="technical-card" key={library}>
            <h3>{library}</h3>
            <div className="technical-table">
              <div className="technical-row technical-head technical-row-wide">
                <span>Term</span>
                <span>p-value</span>
                <span>adj p-value</span>
                <span>Score</span>
                <span>Overlap genes</span>
              </div>
              {terms.slice(0, 10).map((term, index) => (
                <div className="technical-row technical-row-wide" key={`${library}-${index}`}>
                  <span>{term.term || term.t || '—'}</span>
                  <span>{formatNumber(term.p_value ?? term.p)}</span>
                  <span>{formatNumber(term.adjusted_p_value ?? term.adj)}</span>
                  <span>{formatNumber(term.combined_score ?? term.cs)}</span>
                  <span>{Array.isArray(term.overlapping_genes) ? term.overlapping_genes.join(', ') : Array.isArray(term.genes) ? term.genes.join(', ') : '—'}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null
      ))}

      {rwrRows.length > 0 && (
        <div className="technical-card">
          <h3>RWR genes</h3>
          <div className="technical-table">
            <div className="technical-row technical-head">
              <span>Gene</span>
              <span>Score</span>
            </div>
            {rwrRows.map((row, index) => (
              <div className="technical-row" key={`${row[0] || 'rwr'}-${index}`}>
                <span>{Array.isArray(row) ? row[0] : row.gene || '—'}</span>
                <span>{formatNumber(Array.isArray(row) ? row[1] : row.score)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {openTargets && (
        <div className="technical-card">
          <h3>OpenTargets</h3>
          <p><strong>Gene:</strong> {openTargets.gene || '—'}</p>
          <p><strong>Disease:</strong> {openTargets.disease || '—'}</p>
          <p><strong>Associated:</strong> {String(openTargets.associated)}</p>
          <p><strong>Score:</strong> {formatNumber(openTargets.association_score)}</p>
        </div>
      )}

      {primeKg && (
        <div className="technical-card">
          <h3>PrimeKG</h3>
          {primeKg.answer && <p><strong>Answer:</strong> {primeKg.answer}</p>}
          {primeKg.cypher && <pre className="trace-code">{primeKg.cypher}</pre>}
          {Array.isArray(primeKg.rows) && primeKg.rows.length > 0 && (
            <div className="trace-result-block">
              <div className="trace-label">Rows</div>
              {primeKg.rows.slice(0, 8).map((row, index) => (
                <div className="trace-row-card" key={`kg-row-${index}`}>
                  {renderKvList(row, `kg-row-${index}`)}
                </div>
              ))}
            </div>
          )}
          {Array.isArray(primeKg.edges) && primeKg.edges.length > 0 && (
            <div className="technical-table">
              <div className="technical-row technical-head technical-row-wide">
                <span>Source</span>
                <span>Relation</span>
                <span>Target</span>
              </div>
              {primeKg.edges.slice(0, 10).map((edge, index) => (
                <div className="technical-row technical-row-wide" key={`kg-${index}`}>
                  <span>{edge.source?.name || edge.source || '—'}</span>
                  <span>{edge.display_relation || edge.relation || '—'}</span>
                  <span>{edge.target?.name || edge.target || '—'}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {(pyvisHtmlPath || keggPath || volcanoPath) && (
        <div className="technical-card">
          <h3>Visual outputs</h3>
          {pyvisHtmlPath && <p><strong>Network HTML:</strong> {pyvisHtmlPath}</p>}
          {keggPath && <p><strong>KEGG image:</strong> {keggPath}</p>}
          {volcanoPath && <p><strong>Volcano image:</strong> {volcanoPath}</p>}
        </div>
      )}

      {toolHistory.length > 0 && (
        <div className="technical-card">
          <h3>Tool trace</h3>
          <div className="trace-list">
            {toolHistory.map((step, index) => (
              <div className="trace-step" key={`trace-${index}`}>
                <div className="trace-step-header">
                  <div className="trace-step-index">Step {index + 1}</div>
                  <div className="trace-step-name">{step.tool || 'tool'}</div>
                </div>
                {step.args && typeof step.args === 'object' && (
                  <div className="trace-block">
                    <div className="trace-label">Args</div>
                    {renderKvList(step.args, `args-${index}`)}
                  </div>
                )}
                {step.result && typeof step.result === 'object' && (
                  <div className="trace-block">
                    <div className="trace-label">Result</div>
                    {renderKvList(step.result, `result-${index}`)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  )
}

function formatTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString()
}

function App() {
  const [user, setUser] = useState(null)
  const [authMode, setAuthMode] = useState('login')
  const [authForm, setAuthForm] = useState(emptyAuth)
  const [authError, setAuthError] = useState('')
  const [loadingAuth, setLoadingAuth] = useState(false)
  const [chats, setChats] = useState([])
  const [activeChatId, setActiveChatId] = useState(null)
  const [messages, setMessages] = useState([])
  const [messageText, setMessageText] = useState('')
  const [chatBusy, setChatBusy] = useState(false)
  const [appError, setAppError] = useState('')
  const [technicalMeta, setTechnicalMeta] = useState(null)

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) || null,
    [activeChatId, chats],
  )

  useEffect(() => {
    const token = getToken()
    if (!token) return

    ;(async () => {
      try {
        const meResult = await me()
        setUser(meResult)
      } catch {
        setToken('')
      }
    })()
  }, [])

  useEffect(() => {
    if (!user) return

    ;(async () => {
      try {
        const chatResult = await fetchChats()
        setChats(chatResult.chats || [])

        const storedChatId = localStorage.getItem(`gea_last_chat_${user.id}`)
        const fallbackChatId = chatResult.chats?.[0]?.id || null
        const selectedId = storedChatId && chatResult.chats?.some((chat) => chat.id === Number(storedChatId))
          ? Number(storedChatId)
          : fallbackChatId

        if (selectedId) {
          setActiveChatId(selectedId)
        } else {
          const created = await createChat('New chat')
          setChats((current) => [created, ...current])
          setActiveChatId(created.id)
        }
      } catch (error) {
        setAppError(error.message)
      }
    })()
  }, [user])

  useEffect(() => {
    if (!user || !activeChatId) return
    localStorage.setItem(`gea_last_chat_${user.id}`, String(activeChatId))
  }, [activeChatId, user])

  useEffect(() => {
    setTechnicalMeta(activeChat?.last_meta || null)
  }, [activeChat])

  useEffect(() => {
    if (!user || !activeChatId) return

    ;(async () => {
      try {
        const result = await fetchMessages(activeChatId)
        setMessages(result.messages || [])
      } catch (error) {
        setAppError(error.message)
      }
    })()
  }, [activeChatId, user])

  async function handleAuthSubmit(event) {
    event.preventDefault()
    setAuthError('')
    setLoadingAuth(true)

    try {
      const payload = {
        email: authForm.email,
        password: authForm.password,
        display_name: authForm.display_name || undefined,
      }
      const result = authMode === 'login' ? await login(payload) : await register(payload)
      setToken(result.token)
      setUser(result.user)
      setChats([])
      setMessages([])
      setActiveChatId(null)
      setAppError('')
    } catch (error) {
      setAuthError(error.message)
    } finally {
      setLoadingAuth(false)
    }
  }

  async function handleSendMessage(event) {
    event.preventDefault()
    if (!messageText.trim() || !activeChatId || chatBusy) return

    const text = messageText.trim()
    setMessageText('')
    setChatBusy(true)
    setAppError('')

    const optimisticMessage = {
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
      optimistic: true,
    }
    setMessages((current) => [...current, optimisticMessage])

    try {
      const result = await sendMessage(activeChatId, text)
      setMessages(result.messages || [])
      setTechnicalMeta(result.meta || null)
      setChats((current) =>
        current
          .map((chat) => (chat.id === activeChatId ? result.chat || chat : chat))
          .sort((a, b) => new Date(b.updated_at || 0) - new Date(a.updated_at || 0)),
      )
    } catch (error) {
      setAppError(error.message)
      setMessages((current) => current.filter((message) => !message.optimistic))
    } finally {
      setChatBusy(false)
    }
  }

  async function handleNewChat() {
    try {
      const created = await createChat('New chat')
      setChats((current) => [created, ...current])
      setActiveChatId(created.id)
      setMessages([])
    } catch (error) {
      setAppError(error.message)
    }
  }

  function handleLogout() {
    setToken('')
    setUser(null)
    setChats([])
    setMessages([])
    setActiveChatId(null)
    setAppError('')
    setAuthError('')
  }

  if (!user) {
    return (
      <div className="auth-shell">
        <div className="auth-card">
          <div className="brand">
            <div className="brand-mark">GEA</div>
            <div>
              <h1>GEA Agent</h1>
              <p>Login to continue your analysis chats.</p>
            </div>
          </div>

          <div className="auth-tabs">
            <button className={authMode === 'login' ? 'active' : ''} onClick={() => setAuthMode('login')}>Log in</button>
            <button className={authMode === 'register' ? 'active' : ''} onClick={() => setAuthMode('register')}>Register</button>
          </div>

          <form onSubmit={handleAuthSubmit} className="auth-form">
            <label>
              Email
              <input
                type="email"
                value={authForm.email}
                onChange={(event) => setAuthForm((current) => ({ ...current, email: event.target.value }))}
                placeholder="you@example.com"
                required
              />
            </label>
            {authMode === 'register' && (
              <label>
                Display name
                <input
                  type="text"
                  value={authForm.display_name}
                  onChange={(event) => setAuthForm((current) => ({ ...current, display_name: event.target.value }))}
                  placeholder="Your name"
                />
              </label>
            )}
            <label>
              Password
              <input
                type="password"
                value={authForm.password}
                onChange={(event) => setAuthForm((current) => ({ ...current, password: event.target.value }))}
                placeholder="••••••••"
                required
              />
            </label>
            {authError && <div className="alert error">{authError}</div>}
            <button className="primary" type="submit" disabled={loadingAuth}>
              {loadingAuth ? 'Please wait…' : authMode === 'login' ? 'Log in' : 'Create account'}
            </button>
          </form>
        </div>
      </div>
    )
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-header">
          <div>
            <div className="eyebrow">Signed in as</div>
            <div className="user-name">{user.display_name || user.email}</div>
          </div>
          <button className="ghost" onClick={handleLogout}>Log out</button>
        </div>

        <button className="primary new-chat" onClick={handleNewChat}>+ New chat</button>

        <div className="chat-list">
          {chats.map((chat) => (
            <button
              key={chat.id}
              className={`chat-item ${chat.id === activeChatId ? 'active' : ''}`}
              onClick={() => setActiveChatId(chat.id)}
            >
              <div className="chat-title">{chat.title}</div>
              <div className="chat-meta">{formatTime(chat.updated_at)}</div>
            </button>
          ))}
        </div>
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div>
            <div className="eyebrow">Current chat</div>
            <h2>{activeChat?.title || 'Select a chat'}</h2>
          </div>
          {activeChat && (
            <div className="chat-badge">
              {activeChat.analysis_arm || 'general'}
            </div>
          )}
        </header>

        {appError && <div className="alert error">{appError}</div>}

        <section className="message-list">
          {messages.length === 0 ? (
            <div className="empty-state">
              <h3>Start a biomedical analysis</h3>
              <p>This agent can answer general questions or call specialists for technical analysis.</p>

              <div className="empty-grid">
                <div className="empty-card">
                  <div className="empty-card-title">Specialists</div>
                  <ul>
                    <li>DEG analysis from SRP studies</li>
                    <li>Pathway enrichment with Enrichr</li>
                    <li>RWR target prioritization on STRING</li>
                    <li>Literature analysis from OpenAlex</li>
                    <li>OpenTargets gene-disease association</li>
                    <li>PrimeKG graph relationships</li>
                    <li>Visualization for network, KEGG, and volcano plots</li>
                  </ul>
                </div>

                <div className="empty-card">
                  <div className="empty-card-title">Important constraints</div>
                  <ul>
                    <li>For DEG analysis, provide SRP IDs plus control and treatment or test labels.</li>
                    <li>For follow-up pathway or disease-association questions, the agent can reuse genes already stored in chat memory.</li>
                    <li>Visualization works best after DEG, pathway, or RWR results already exist in the same chat.</li>
                  </ul>
                </div>

                <div className="empty-card">
                  <div className="empty-card-title">Example prompts</div>
                  <ul>
                    <li>Identify DEGs between Healthy lung tissue and COPD lung tissue for SRP123456.</li>
                    <li>Get pathways for the top 15 up-regulated genes.</li>
                    <li>Are the up-regulated genes associated with diabetes?</li>
                    <li>What are neighbors of CRISPLD2 in PrimeKG?</li>
                  </ul>
                </div>
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={`${message.role}-${index}-${message.created_at || ''}`} className={`message ${message.role}`}>
                <div className="message-role">{message.role}</div>
                <div className="message-content">{message.content}</div>
              </div>
            ))
          )}
        </section>

        {renderTechnicalPanel(technicalMeta)}

        <form className="composer" onSubmit={handleSendMessage}>
          <textarea
            value={messageText}
            onChange={(event) => setMessageText(event.target.value)}
            placeholder="Ask a follow-up or start a new analysis…"
            rows={3}
          />
          <div className="composer-actions">
            <div className="composer-hint">
              {activeChat ? `Chat updated ${formatTime(activeChat.updated_at)}` : 'Create or select a chat to begin.'}
            </div>
            <button className="primary" type="submit" disabled={chatBusy || !activeChatId}>
              {chatBusy ? 'Thinking…' : 'Send'}
            </button>
          </div>
        </form>
      </main>
    </div>
  )
}

export default App
