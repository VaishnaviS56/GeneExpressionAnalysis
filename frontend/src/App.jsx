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

const analysisArmLabels = {
  general: 'General',
  literature: 'Literature',
  srp: 'DEG',
  disease: 'Disease',
  research_literature: 'Literature',
  memory_rwr: 'RWR',
  primekg: 'PrimeKG',
  opentargets: 'OpenTargets',
  memory_lookup: 'Lookup',
  l1000cds2: 'L1000CDS2',
  pubchem: 'PubChem',
}

function formatNumber(value) {
  if (value === null || value === undefined || value === '') return '-'
  const num = Number(value)
  if (Number.isNaN(num)) return String(value)
  if (num === 0) return '0'
  if (Math.abs(num) < 0.001 || Math.abs(num) >= 1000) {
    return num.toExponential(2)
  }
  return num.toFixed(4)
}

function formatValue(value) {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'number') return formatNumber(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (Array.isArray(value)) return value.length ? value.map((item) => formatValue(item)).join(', ') : '[]'
  if (typeof value === 'object') {
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

function formatTime(value) {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleString()
}

function formatArm(value) {
  const key = String(value || 'general').trim().toLowerCase()
  return analysisArmLabels[key] || key.replaceAll('_', ' ')
}

function formatChatMode(chat, technicalMeta) {
  const agentType = String(chat?.agent_type || technicalMeta?.agent_type || '').trim().toLowerCase()
  if (agentType === 'literature') return 'Literature'
  return formatArm(chat?.analysis_arm || technicalMeta?.analysis_arm || 'general')
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

function renderInlineMarkdown(text, keyPrefix) {
  const value = String(text || '')
  const pattern = /(\[[^\]]+\]\([^)]+\)|`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|_[^_\n]+_)/g
  const nodes = []
  let lastIndex = 0
  let matchIndex = 0

  for (const match of value.matchAll(pattern)) {
    const token = match[0]
    const start = match.index ?? 0
    if (start > lastIndex) {
      nodes.push(value.slice(lastIndex, start))
    }

    if (token.startsWith('[')) {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/)
      if (linkMatch) {
        nodes.push(
          <a
            className="markdown-link"
            href={linkMatch[2]}
            key={`${keyPrefix}-link-${matchIndex}`}
            rel="noreferrer"
            target="_blank"
          >
            {linkMatch[1]}
          </a>,
        )
      } else {
        nodes.push(token)
      }
    } else if (token.startsWith('`')) {
      nodes.push(
        <code className="inline-code" key={`${keyPrefix}-code-${matchIndex}`}>
          {token.slice(1, -1)}
        </code>,
      )
    } else if (token.startsWith('**') || token.startsWith('__')) {
      nodes.push(
        <strong key={`${keyPrefix}-strong-${matchIndex}`}>
          {token.slice(2, -2)}
        </strong>,
      )
    } else {
      nodes.push(
        <em key={`${keyPrefix}-em-${matchIndex}`}>
          {token.slice(1, -1)}
        </em>,
      )
    }

    lastIndex = start + token.length
    matchIndex += 1
  }

  if (lastIndex < value.length) {
    nodes.push(value.slice(lastIndex))
  }

  return nodes.length ? nodes : [value]
}

function parseMarkdownBlocks(content) {
  const lines = String(content || '').replace(/\r\n/g, '\n').split('\n')
  const blocks = []
  let paragraphLines = []
  let listType = null
  let listItems = []
  let quoteLines = []
  let codeLines = []
  let inCode = false

  function flushParagraph() {
    if (!paragraphLines.length) return
    blocks.push({ type: 'paragraph', text: paragraphLines.join(' ').trim() })
    paragraphLines = []
  }

  function flushList() {
    if (!listType || !listItems.length) return
    blocks.push({ type: listType, items: [...listItems] })
    listType = null
    listItems = []
  }

  function flushQuote() {
    if (!quoteLines.length) return
    blocks.push({ type: 'quote', text: quoteLines.join(' ').trim() })
    quoteLines = []
  }

  function flushCode() {
    if (!codeLines.length) return
    blocks.push({ type: 'code', text: codeLines.join('\n') })
    codeLines = []
  }

  for (const rawLine of lines) {
    const line = rawLine ?? ''
    const trimmed = line.trim()

    if (trimmed.startsWith('```')) {
      flushParagraph()
      flushList()
      flushQuote()
      if (inCode) {
        flushCode()
        inCode = false
      } else {
        inCode = true
      }
      continue
    }

    if (inCode) {
      codeLines.push(line)
      continue
    }

    if (!trimmed) {
      flushParagraph()
      flushList()
      flushQuote()
      continue
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/)
    if (headingMatch) {
      flushParagraph()
      flushList()
      flushQuote()
      blocks.push({
        type: 'heading',
        level: headingMatch[1].length,
        text: headingMatch[2].trim(),
      })
      continue
    }

    const bulletMatch = trimmed.match(/^[-*]\s+(.*)$/)
    if (bulletMatch) {
      flushParagraph()
      flushQuote()
      if (listType !== 'ul') {
        flushList()
        listType = 'ul'
      }
      listItems.push(bulletMatch[1].trim())
      continue
    }

    const orderedMatch = trimmed.match(/^\d+\.\s+(.*)$/)
    if (orderedMatch) {
      flushParagraph()
      flushQuote()
      if (listType !== 'ol') {
        flushList()
        listType = 'ol'
      }
      listItems.push(orderedMatch[1].trim())
      continue
    }

    const quoteMatch = trimmed.match(/^>\s?(.*)$/)
    if (quoteMatch) {
      flushParagraph()
      flushList()
      quoteLines.push(quoteMatch[1].trim())
      continue
    }

    flushList()
    flushQuote()
    paragraphLines.push(trimmed)
  }

  flushParagraph()
  flushList()
  flushQuote()
  flushCode()

  return blocks
}

function MarkdownContent({ content }) {
  const blocks = useMemo(() => parseMarkdownBlocks(content), [content])

  return (
    <div className="markdown-body">
      {blocks.map((block, index) => {
        if (block.type === 'heading') {
          const Tag = `h${Math.min(block.level || 2, 6)}`
          return <Tag key={`block-${index}`}>{renderInlineMarkdown(block.text, `heading-${index}`)}</Tag>
        }
        if (block.type === 'ul') {
          return (
            <ul key={`block-${index}`}>
              {block.items.map((item, itemIndex) => (
                <li key={`item-${index}-${itemIndex}`}>{renderInlineMarkdown(item, `ul-${index}-${itemIndex}`)}</li>
              ))}
            </ul>
          )
        }
        if (block.type === 'ol') {
          return (
            <ol key={`block-${index}`}>
              {block.items.map((item, itemIndex) => (
                <li key={`item-${index}-${itemIndex}`}>{renderInlineMarkdown(item, `ol-${index}-${itemIndex}`)}</li>
              ))}
            </ol>
          )
        }
        if (block.type === 'quote') {
          return <blockquote key={`block-${index}`}>{renderInlineMarkdown(block.text, `quote-${index}`)}</blockquote>
        }
        if (block.type === 'code') {
          return (
            <pre className="markdown-code-block" key={`block-${index}`}>
              <code>{block.text}</code>
            </pre>
          )
        }
        return <p key={`block-${index}`}>{renderInlineMarkdown(block.text, `paragraph-${index}`)}</p>
      })}
    </div>
  )
}

function MessageBubble({ message }) {
  return (
    <article className={`message ${message.role}`}>
      <div className="message-meta-row">
        <div className="message-role">{message.role === 'assistant' ? 'Agent' : 'You'}</div>
        {message.created_at && <div className="message-time">{formatTime(message.created_at)}</div>}
      </div>
      <div className="message-content">
        <MarkdownContent content={message.content} />
      </div>
    </article>
  )
}

function getRunMetrics(meta) {
  if (!meta || typeof meta !== 'object') return []

  const metrics = []
  const degRows = Array.isArray(meta.deg_gene_records) ? meta.deg_gene_records.length : 0
  const pathwayLibraries = meta.enrichr && typeof meta.enrichr === 'object' && meta.enrichr.libraries && typeof meta.enrichr.libraries === 'object'
    ? Object.keys(meta.enrichr.libraries).length
    : 0
  const rwrHits = Array.isArray(meta.rwr_genes) ? meta.rwr_genes.length : 0
  const graphNodes = meta.network && typeof meta.network === 'object' ? Number(meta.network.nodes || 0) : 0
  const graphEdges = meta.network && typeof meta.network === 'object' ? Number(meta.network.edges || 0) : 0
  const l1000Hits = meta.l1000cds2_result && typeof meta.l1000cds2_result === 'object' && Array.isArray(meta.l1000cds2_result.top_drugs)
    ? meta.l1000cds2_result.top_drugs.length
    : 0

  if (degRows) metrics.push({ label: 'DEG rows', value: degRows })
  if (pathwayLibraries) metrics.push({ label: 'Libraries', value: pathwayLibraries })
  if (rwrHits) metrics.push({ label: 'RWR hits', value: rwrHits })
  if (graphNodes) metrics.push({ label: 'Nodes', value: graphNodes })
  if (graphEdges) metrics.push({ label: 'Edges', value: graphEdges })
  if (l1000Hits) metrics.push({ label: 'L1000 hits', value: l1000Hits })

  return metrics
}

function TechnicalPanel({ meta }) {
  if (!meta || typeof meta !== 'object') {
    return (
      <aside className="insights-panel">
        <section className="technical-panel">
          <div className="technical-hero placeholder">
            <div className="eyebrow">Technical results</div>
            <h3>No technical output yet</h3>
            <p>Run a DEG, enrichment, RWR, or disease query and the structured results will appear here.</p>
          </div>
        </section>
      </aside>
    )
  }

  const degRows = Array.isArray(meta.deg_gene_records) ? meta.deg_gene_records.slice(0, 10) : []
  const enrichrLibs = meta.enrichr && typeof meta.enrichr === 'object' && meta.enrichr.libraries && typeof meta.enrichr.libraries === 'object'
    ? meta.enrichr.libraries
    : {}
  const rwrRows = Array.isArray(meta.rwr_genes) ? meta.rwr_genes.slice(0, 10) : []
  const openTargets = meta.opentargets_result && typeof meta.opentargets_result === 'object' ? meta.opentargets_result : null
  const primeKg = meta.primekg_result && typeof meta.primekg_result === 'object' ? meta.primekg_result : null
  const l1000 = meta.l1000cds2_result && typeof meta.l1000cds2_result === 'object' ? meta.l1000cds2_result : null
  const pubchem = meta.pubchem_result && typeof meta.pubchem_result === 'object' ? meta.pubchem_result : null
  const toolHistory = Array.isArray(meta.tool_history) ? meta.tool_history : []
  const pyvisHtmlPath = typeof meta.pyvis_html_path === 'string' ? meta.pyvis_html_path : ''
  const keggPath = typeof meta.kegg_pathway_path === 'string' ? meta.kegg_pathway_path : ''
  const volcanoPath = typeof meta.volcano_plot_path === 'string' ? meta.volcano_plot_path : ''
  const requestedCellLines = Array.isArray(l1000?.requested_cell_lines) ? l1000.requested_cell_lines : []
  const runMetrics = getRunMetrics(meta)

  return (
    <aside className="insights-panel">
      <section className="technical-panel">
        <div className="technical-hero">
          <div className="eyebrow">Technical results</div>
          <div className="technical-hero-row">
            <h3>{formatArm(meta.analysis_arm)}</h3>
            <div className="chat-badge">{formatArm(meta.analysis_arm)}</div>
          </div>
          <p>Structured outputs from the current chat stay here so the main answer can remain readable.</p>
          {runMetrics.length > 0 && (
            <div className="metric-grid">
              {runMetrics.map((metric) => (
                <div className="metric-card" key={metric.label}>
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                </div>
              ))}
            </div>
          )}
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
                  <span>{row.gene || '-'}</span>
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
                    <span>{term.term || term.t || '-'}</span>
                    <span>{formatNumber(term.p_value ?? term.p)}</span>
                    <span>{formatNumber(term.adjusted_p_value ?? term.adj)}</span>
                    <span>{formatNumber(term.combined_score ?? term.cs)}</span>
                    <span>{Array.isArray(term.overlapping_genes) ? term.overlapping_genes.join(', ') : Array.isArray(term.genes) ? term.genes.join(', ') : '-'}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null
        ))}

        {rwrRows.length > 0 && (
          <div className="technical-card">
            <h3>RWR targets</h3>
            <div className="technical-table">
              <div className="technical-row technical-head">
                <span>Gene</span>
                <span>Score</span>
              </div>
              {rwrRows.map((row, index) => (
                <div className="technical-row technical-row-two" key={`${Array.isArray(row) ? row[0] : row.gene || 'rwr'}-${index}`}>
                  <span>{Array.isArray(row) ? row[0] : row.gene || '-'}</span>
                  <span>{formatNumber(Array.isArray(row) ? row[1] : row.score)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {l1000 && (
          <div className="technical-card">
            <h3>L1000CDS2 matches</h3>
            {requestedCellLines.length > 0 && (
              <p><strong>Cell lines:</strong> {requestedCellLines.join(', ')}</p>
            )}
            {Array.isArray(l1000.top_drugs) && l1000.top_drugs.length > 0 ? (
              <div className="technical-table">
                <div className="technical-row technical-head technical-row-wide">
                  <span>Drug</span>
                  <span>Rank</span>
                  <span>Score</span>
                  <span>Signatures</span>
                  <span>Cell lines</span>
                </div>
                {l1000.top_drugs.slice(0, 10).map((row, index) => (
                  <div className="technical-row technical-row-wide" key={`${row.name || 'drug'}-${index}`}>
                    <span>{row.name || '-'}</span>
                    <span>{formatValue(row.best_rank)}</span>
                    <span>{formatNumber(row.best_score)}</span>
                    <span>{formatValue(row.signature_count)}</span>
                    <span>{Array.isArray(row.cell_lines) ? row.cell_lines.join(', ') : '-'}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p>{l1000.message || 'No L1000CDS2 results available.'}</p>
            )}
          </div>
        )}

        {openTargets && (
          <div className="technical-card">
            <h3>OpenTargets</h3>
            <p><strong>Gene:</strong> {openTargets.gene || '-'}</p>
            <p><strong>Disease:</strong> {openTargets.disease || '-'}</p>
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
          </div>
        )}

        {pubchem && (
          <div className="technical-card">
            <h3>PubChem</h3>
            <p><strong>Compound:</strong> {pubchem.title || pubchem.drug_name || '-'}</p>
            <p><strong>CID:</strong> {pubchem.cid || '-'}</p>
            {pubchem.properties && typeof pubchem.properties === 'object' && (
              <div className="trace-result-block">
                {renderKvList(pubchem.properties, 'pubchem-properties')}
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
                <details className="trace-step" key={`trace-${index}`}>
                  <summary className="trace-step-header">
                    <div className="trace-step-index">Step {index + 1}</div>
                    <div className="trace-step-name">{step.tool || 'tool'}</div>
                  </summary>
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
                </details>
              ))}
            </div>
          </div>
        )}
      </section>
    </aside>
  )
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="eyebrow">Biomedical workspace</div>
      <h3>Start with a chat, then keep the technical context in one place.</h3>
      <p>The agent can handle differential expression, enrichment, RWR prioritization, literature context, and drug matching in the same thread.</p>

      <div className="empty-grid">
        <div className="empty-card accent-cyan">
          <div className="empty-card-title">Typical workflows</div>
          <ul>
            <li>Run DEG from an SRP study, then ask for top pathways.</li>
            <li>Use stored DEGs as seed genes for RWR target prioritization.</li>
            <li>Follow up with disease association or L1000 drug matching.</li>
          </ul>
        </div>

        <div className="empty-card accent-amber">
          <div className="empty-card-title">Good prompts</div>
          <ul>
            <li>Identify DEGs between Healthy lung tissue and COPD lung tissue for SRP123456.</li>
            <li>Get pathways for the top 15 up-regulated genes.</li>
            <li>Run RWR on the stored DEG genes and rank candidate targets.</li>
          </ul>
        </div>

        <div className="empty-card accent-slate">
          <div className="empty-card-title">UI behavior</div>
          <ul>
            <li>The left side is the conversation and rendered answer.</li>
            <li>The right rail keeps structured technical outputs and tool traces.</li>
            <li>Chat previews and run summaries update as each turn completes.</li>
          </ul>
        </div>
      </div>
    </div>
  )
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

  const runMetrics = useMemo(() => getRunMetrics(technicalMeta), [technicalMeta])

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
          const created = await createChat('New chat', 'general')
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

  async function handleNewChat(agentType = 'general') {
    try {
      const title = agentType === 'literature' ? 'Literature chat' : 'New chat'
      const created = await createChat(title, agentType)
      setChats((current) => [created, ...current])
      setActiveChatId(created.id)
      setMessages([])
      setTechnicalMeta(created.last_meta || null)
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
              <p>Sign in to continue your biomedical analysis workspace.</p>
            </div>
          </div>

          <div className="auth-tabs">
            <button className={authMode === 'login' ? 'active' : ''} onClick={() => setAuthMode('login')} type="button">Log in</button>
            <button className={authMode === 'register' ? 'active' : ''} onClick={() => setAuthMode('register')} type="button">Register</button>
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
                placeholder="********"
                required
              />
            </label>

            {authError && <div className="alert error">{authError}</div>}

            <button className="primary" type="submit" disabled={loadingAuth}>
              {loadingAuth ? 'Please wait...' : authMode === 'login' ? 'Log in' : 'Create account'}
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
          <button className="ghost" onClick={handleLogout} type="button">Log out</button>
        </div>

        <div className="sidebar-panel">
          <div className="sidebar-panel-top">
            <div>
              <div className="eyebrow">Current run</div>
              <h3>{formatChatMode(activeChat, technicalMeta)}</h3>
            </div>
            <div className="chat-badge">{activeChat?.message_count || 0} msgs</div>
          </div>
          <p className="sidebar-panel-copy">
            {activeChat?.last_message_preview || 'Pick a chat or start a new analysis to populate the workspace.'}
          </p>
          {runMetrics.length > 0 && (
            <div className="sidebar-metrics">
              {runMetrics.slice(0, 4).map((metric) => (
                <div className="sidebar-metric" key={metric.label}>
                  <span>{metric.label}</span>
                  <strong>{metric.value}</strong>
                </div>
              ))}
            </div>
          )}
        </div>

        <button className="primary new-chat" onClick={() => handleNewChat('general')} type="button">+ New chat</button>

        <div className="chat-list">
          {chats.map((chat) => (
            <button
              key={chat.id}
              className={`chat-item ${chat.id === activeChatId ? 'active' : ''}`}
              onClick={() => setActiveChatId(chat.id)}
              type="button"
            >
              <div className="chat-item-top">
                <div className="chat-title">{chat.title}</div>
                <div className="chat-mini-badge">{formatChatMode(chat, chat.last_meta)}</div>
              </div>
              <div className="chat-preview">{chat.last_message_preview || 'No messages yet.'}</div>
              <div className="chat-meta-row">
                <span>{chat.message_count || 0} msgs</span>
                <span>{formatTime(chat.updated_at)}</span>
              </div>
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
          <div className="header-actions">
            {technicalMeta?.route_rationale && (
              <div className="header-caption">{technicalMeta.route_rationale}</div>
            )}
            <div className="chat-badge">{formatChatMode(activeChat, technicalMeta)}</div>
          </div>
        </header>

        {appError && <div className="alert error">{appError}</div>}

        <div className="workspace-layout">
          <section className="conversation-panel">
            {runMetrics.length > 0 && (
              <div className="summary-strip">
                {runMetrics.map((metric) => (
                  <div className="summary-pill" key={metric.label}>
                    <span>{metric.label}</span>
                    <strong>{metric.value}</strong>
                  </div>
                ))}
              </div>
            )}

            <section className="message-list">
              {messages.length === 0 ? (
                <EmptyState />
              ) : (
                messages.map((message, index) => (
                  <MessageBubble
                    key={`${message.role}-${index}-${message.created_at || ''}`}
                    message={message}
                  />
                ))
              )}
            </section>

            <form className="composer" onSubmit={handleSendMessage}>
              <div className="composer-shell">
                <textarea
                  value={messageText}
                  onChange={(event) => setMessageText(event.target.value)}
                  placeholder="Ask a follow-up or start a new analysis..."
                  rows={4}
                />
                <div className="composer-actions">
                  <div className="composer-hint">
                    {activeChat ? `Chat updated ${formatTime(activeChat.updated_at)}` : 'Create or select a chat to begin.'}
                  </div>
                  <button className="primary" type="submit" disabled={chatBusy || !activeChatId}>
                    {chatBusy ? 'Thinking...' : 'Send'}
                  </button>
                </div>
              </div>
            </form>
          </section>

          <TechnicalPanel meta={technicalMeta} />
        </div>
      </main>
    </div>
  )
}

export default App
