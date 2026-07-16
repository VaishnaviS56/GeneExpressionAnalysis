import { useEffect, useMemo, useState } from 'react'
import {
  buildAssetUrl,
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
  hypothesis: 'Hypothesis',
}

function formatNumber(value) {
  if (value === null || value === undefined || value === '') return '-'
  const num = Number(value)
  if (Number.isNaN(num)) return String(value)
  if (num === 0) return '0'
  if (Math.abs(num) < 0.001 || Math.abs(num) >= 1000) return num.toExponential(2)
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
  const literatureHits = Array.isArray(meta.ranked_openalex_papers)
    ? meta.ranked_openalex_papers.length
    : Array.isArray(meta.openalex_papers)
      ? meta.openalex_papers.length
      : 0
  const l1000Hits = meta.l1000cds2_result && typeof meta.l1000cds2_result === 'object' && Array.isArray(meta.l1000cds2_result.top_drugs)
    ? meta.l1000cds2_result.top_drugs.length
    : 0

  if (degRows) metrics.push({ label: 'DEG rows', value: degRows })
  if (pathwayLibraries) metrics.push({ label: 'Libraries', value: pathwayLibraries })
  if (rwrHits) metrics.push({ label: 'RWR hits', value: rwrHits })
  if (graphNodes) metrics.push({ label: 'Nodes', value: graphNodes })
  if (graphEdges) metrics.push({ label: 'Edges', value: graphEdges })
  if (literatureHits) metrics.push({ label: 'Papers', value: literatureHits })
  if (l1000Hits) metrics.push({ label: 'L1000 hits', value: l1000Hits })

  return metrics
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
    if (start > lastIndex) nodes.push(value.slice(lastIndex, start))

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

  if (lastIndex < value.length) nodes.push(value.slice(lastIndex))
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
      blocks.push({ type: 'heading', level: headingMatch[1].length, text: headingMatch[2].trim() })
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
    <article className={`message ${message.role} ${message.pending ? 'pending' : ''}`}>
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

function LoadingState({ label = 'Loading workspace' }) {
  return (
    <div className="loading-state" aria-live="polite">
      <div className="loading-orb">
        <span />
        <span />
        <span />
      </div>
      <div className="loading-copy">{label}</div>
    </div>
  )
}

function SectionCard({ title, subtitle, children }) {
  return (
    <section className="technical-card">
      <div className="section-heading">
        <h3>{title}</h3>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {children}
    </section>
  )
}

function ArtifactPreview({ label, path, mode }) {
  if (!path) return null
  const src = buildAssetUrl(path)

  return (
    <div className="artifact-block">
      <div className="artifact-head">
        <div>
          <div className="artifact-label">{label}</div>
          <div className="artifact-path">{path}</div>
        </div>
        <a className="ghost mini" href={src} rel="noreferrer" target="_blank">Open</a>
      </div>
      {mode === 'html' ? (
        <iframe className="artifact-frame" src={src} title={label} />
      ) : (
        <img className="artifact-image" src={src} alt={label} loading="lazy" />
      )}
    </div>
  )
}

function TechnicalPanel({ meta }) {
  if (!meta || typeof meta !== 'object') {
    return (
      <aside className="insights-panel">
        <section className="technical-panel">
          <div className="technical-hero placeholder">
            <div className="eyebrow">Technical results</div>
            <h3>No technical output yet</h3>
            <p>Run a DEG, enrichment, RWR, literature, or drug lookup query and the structured outputs will land here.</p>
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
  const hypothesis = meta.hypothesis_result && typeof meta.hypothesis_result === 'object' ? meta.hypothesis_result : null
  const memoryLookup = meta.memory_lookup_result && typeof meta.memory_lookup_result === 'object' ? meta.memory_lookup_result : null
  const memorySlice = meta.memory_slice_result && typeof meta.memory_slice_result === 'object' ? meta.memory_slice_result : null
  const toolHistory = Array.isArray(meta.tool_history) ? meta.tool_history : []
  const rankedPapers = Array.isArray(meta.ranked_openalex_papers) ? meta.ranked_openalex_papers.slice(0, 5) : []
  const scannedPapers = Array.isArray(meta.openalex_papers) ? meta.openalex_papers.slice(0, 5) : []
  const literaturePoints = Array.isArray(meta.literature_key_points) ? meta.literature_key_points.slice(0, 5) : []
  const literatureReferences = Array.isArray(meta.literature_references) ? meta.literature_references.slice(0, 8) : []
  const network = meta.network && typeof meta.network === 'object' ? meta.network : null
  const topDegree = Array.isArray(network?.top_degree) ? network.top_degree.slice(0, 10) : []
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
          <p>Structured outputs stay anchored here so the answer thread can stay readable.</p>
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

        {(pyvisHtmlPath || keggPath || volcanoPath) && (
          <SectionCard title="Visual outputs" subtitle="Generated artifacts are embedded here, matching the Streamlit workspace.">
            <div className="artifact-grid">
              <ArtifactPreview label="Network visualization" mode="html" path={pyvisHtmlPath} />
              <ArtifactPreview label="KEGG pathway" mode="image" path={keggPath} />
              <ArtifactPreview label="Volcano plot" mode="image" path={volcanoPath} />
            </div>
          </SectionCard>
        )}

        {network && (network.nodes || network.edges || topDegree.length > 0) && (
          <SectionCard title="Network summary" subtitle="Graph context from the current analysis run.">
            <div className="summary-pill-grid">
              <div className="summary-pill">
                <span>Nodes</span>
                <strong>{network.nodes || 0}</strong>
              </div>
              <div className="summary-pill">
                <span>Edges</span>
                <strong>{network.edges || 0}</strong>
              </div>
            </div>
            {topDegree.length > 0 && (
              <div className="technical-table">
                <div className="technical-row technical-head technical-row-two">
                  <span>Gene</span>
                  <span>Degree</span>
                </div>
                {topDegree.map((row, index) => (
                  <div className="technical-row technical-row-two" key={`${row.gene || 'degree'}-${index}`}>
                    <span>{row.gene || '-'}</span>
                    <span>{formatValue(row.degree)}</span>
                  </div>
                ))}
              </div>
            )}
          </SectionCard>
        )}

        {(rankedPapers.length > 0 || scannedPapers.length > 0 || literaturePoints.length > 0 || literatureReferences.length > 0 || meta.literature_summary) && (
          <SectionCard
            title="Literature results"
            subtitle={meta.disease_name ? `Context: ${meta.disease_name}` : 'Evidence gathered from literature retrieval.'}
          >
            {meta.literature_summary ? <p>{meta.literature_summary}</p> : null}

            {literaturePoints.length > 0 && (
              <div className="bullet-panel">
                <div className="trace-label">Key points</div>
                <ul>
                  {literaturePoints.map((row, index) => (
                    <li key={`lit-point-${index}`}>
                      {row.point || '-'}
                      {Array.isArray(row.paper_ids) && row.paper_ids.length > 0 ? ` (${row.paper_ids.join(', ')})` : ''}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {rankedPapers.length > 0 && (
              <div className="technical-table">
                <div className="technical-row technical-head technical-row-lit">
                  <span>Title</span>
                  <span>Year</span>
                  <span>Relevance</span>
                  <span>Reason</span>
                </div>
                {rankedPapers.map((paper, index) => (
                  <div className="technical-row technical-row-lit" key={`${paper.title || 'paper'}-${index}`}>
                    <span>{paper.title || '-'}</span>
                    <span>{formatValue(paper.year)}</span>
                    <span>{formatNumber(paper.relevance)}</span>
                    <span>{paper.reason || '-'}</span>
                  </div>
                ))}
              </div>
            )}

            {!rankedPapers.length && scannedPapers.length > 0 && (
              <div className="technical-table">
                <div className="technical-row technical-head technical-row-three">
                  <span>Title</span>
                  <span>Year</span>
                  <span>Source</span>
                </div>
                {scannedPapers.map((paper, index) => (
                  <div className="technical-row technical-row-three" key={`${paper.title || 'scan'}-${index}`}>
                    <span>{paper.title || '-'}</span>
                    <span>{formatValue(paper.year)}</span>
                    <span>{paper.source || '-'}</span>
                  </div>
                ))}
              </div>
            )}

            {literatureReferences.length > 0 && (
              <div className="technical-table">
                <div className="trace-label">References</div>
                {literatureReferences.map((row, index) => (
                  <div className="reference-row" key={`${row.paper_id || 'ref'}-${index}`}>
                    <strong>{row.title || 'Untitled reference'}</strong>
                    <span>{[row.source, row.year, row.pmid ? `PMID ${row.pmid}` : '', row.doi ? `DOI ${row.doi}` : ''].filter(Boolean).join(' | ')}</span>
                  </div>
                ))}
              </div>
            )}
          </SectionCard>
        )}

        {degRows.length > 0 && (
          <SectionCard title="Differential expression" subtitle="Top DEG rows from the current comparison.">
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
          </SectionCard>
        )}

        {Object.entries(enrichrLibs).map(([library, terms]) => (
          Array.isArray(terms) && terms.length > 0 ? (
            <SectionCard key={library} title={library} subtitle="Pathway enrichment results.">
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
            </SectionCard>
          ) : null
        ))}

        {rwrRows.length > 0 && (
          <SectionCard title="Random Walk with Restart" subtitle="Ranked targets from the stored or current seed genes.">
            <div className="technical-table">
              <div className="technical-row technical-head technical-row-two">
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
          </SectionCard>
        )}

        {l1000 && (
          <SectionCard title="L1000CDS2 matches" subtitle="Top compound matches from the current reversal search.">
            {requestedCellLines.length > 0 ? <p><strong>Cell lines:</strong> {requestedCellLines.join(', ')}</p> : null}
            {Array.isArray(l1000.top_drugs) && l1000.top_drugs.length > 0 ? (
              <div className="technical-table">
                <div className="technical-row technical-head technical-row-drug">
                  <span>Drug</span>
                  <span>Rank</span>
                  <span>Score</span>
                  <span>Signatures</span>
                  <span>Cell lines</span>
                </div>
                {l1000.top_drugs.slice(0, 10).map((row, index) => (
                  <div className="technical-row technical-row-drug" key={`${row.name || 'drug'}-${index}`}>
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
          </SectionCard>
        )}

        {openTargets && (
          <SectionCard title="OpenTargets" subtitle="Association summary from the current OpenTargets lookup.">
            <div className="summary-pill-grid">
              <div className="summary-pill">
                <span>Gene</span>
                <strong>{openTargets.gene || '-'}</strong>
              </div>
              <div className="summary-pill">
                <span>Disease</span>
                <strong>{openTargets.disease || '-'}</strong>
              </div>
              <div className="summary-pill">
                <span>Associated</span>
                <strong>{String(openTargets.associated)}</strong>
              </div>
              <div className="summary-pill">
                <span>Score</span>
                <strong>{formatNumber(openTargets.association_score)}</strong>
              </div>
            </div>
          </SectionCard>
        )}

        {primeKg && (
          <SectionCard title="PrimeKG" subtitle="Knowledge graph answer and returned rows.">
            {primeKg.answer ? <p><strong>Answer:</strong> {primeKg.answer}</p> : null}
            {primeKg.cypher ? <pre className="trace-code">{primeKg.cypher}</pre> : null}
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
          </SectionCard>
        )}

        {pubchem && (
          <SectionCard title="PubChem" subtitle="Compound metadata and synthesis context.">
            <p><strong>Compound:</strong> {pubchem.title || pubchem.drug_name || '-'}</p>
            <p><strong>CID:</strong> {pubchem.cid || '-'}</p>
            {pubchem.matched_query ? <p><strong>Matched query:</strong> {pubchem.matched_query}</p> : null}
            {pubchem.matched_strategy ? <p><strong>Match type:</strong> {pubchem.matched_strategy}</p> : null}
            {pubchem.properties && typeof pubchem.properties === 'object' ? (
              <div className="trace-result-block">
                {renderKvList(pubchem.properties, 'pubchem-properties')}
              </div>
            ) : null}
            {Array.isArray(pubchem.synonyms) && pubchem.synonyms.length > 0 && (
              <div className="bullet-panel">
                <div className="trace-label">Synonyms</div>
                <ul>
                  {pubchem.synonyms.slice(0, 12).map((value, index) => (
                    <li key={`synonym-${index}`}>{value}</li>
                  ))}
                </ul>
              </div>
            )}
            {Array.isArray(pubchem.annotation_lines) && pubchem.annotation_lines.length > 0 && (
              <div className="bullet-panel">
                <div className="trace-label">Annotation snippets</div>
                <ul>
                  {pubchem.annotation_lines.slice(0, 8).map((value, index) => (
                    <li key={`annotation-${index}`}>{value}</li>
                  ))}
                </ul>
              </div>
            )}
          </SectionCard>
        )}

        {hypothesis && (
          <SectionCard title="Experimental hypotheses" subtitle="LLM-generated validation ideas grounded in prior chat context and stored memory.">
            {hypothesis.hypothesis_summary ? <p>{hypothesis.hypothesis_summary}</p> : null}
            {Array.isArray(hypothesis.hypotheses) && hypothesis.hypotheses.length > 0 ? (
              <div className="trace-list">
                {hypothesis.hypotheses.slice(0, 8).map((item, index) => (
                  <div className="trace-step" key={`hypothesis-${index}`}>
                    <div className="trace-step-header">
                      <div className="trace-step-index">Hypothesis {index + 1}</div>
                      <div className="trace-step-name">{item.title || 'Untitled hypothesis'}</div>
                    </div>
                    {renderKvList(
                      {
                        rationale: item.rationale,
                        experiment_design: item.experiment_design,
                        expected_observation: item.expected_observation,
                        readouts: item.readouts,
                        existing_evidence: item.existing_evidence,
                        novelty_assessment: item.novelty_assessment,
                        supporting_reference_ids: item.supporting_reference_ids,
                      },
                      `hypothesis-${index}`,
                    )}
                  </div>
                ))}
              </div>
            ) : null}
          </SectionCard>
        )}

        {memoryLookup && Object.keys(memoryLookup).length > 0 && (
          <SectionCard title="Lookup results" subtitle="Structured memory lookup values returned by the agent.">
            {renderKvList(memoryLookup, 'memory-lookup')}
          </SectionCard>
        )}

        {memorySlice && Object.keys(memorySlice).length > 0 && (
          <SectionCard title="Memory slice" subtitle="Subset selections from stored technical state.">
            {renderKvList(memorySlice, 'memory-slice')}
          </SectionCard>
        )}

        {toolHistory.length > 0 && (
          <SectionCard title="Tool trace" subtitle="Recent structured tool calls for this chat.">
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
          </SectionCard>
        )}
      </section>
    </aside>
  )
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="eyebrow">Biomedical workspace</div>
      <h3>Keep the narrative answer and the technical evidence in sync.</h3>
      <p>The React workspace now mirrors the Streamlit outputs, including literature evidence, network artifacts, DEG tables, and visual plots.</p>

      <div className="empty-grid">
        <div className="empty-card accent-cyan">
          <div className="empty-card-title">Typical workflows</div>
          <ul>
            <li>Run DEG from an SRP study, then inspect pathways and volcano output.</li>
            <li>Reuse stored DEG genes for RWR target ranking and network exploration.</li>
            <li>Follow with disease literature, OpenTargets, PubChem, or L1000 matches.</li>
          </ul>
        </div>

        <div className="empty-card accent-amber">
          <div className="empty-card-title">Good prompts</div>
          <ul>
            <li>Identify DEGs between Healthy lung tissue and COPD lung tissue for SRP123456.</li>
            <li>Use the stored DEG genes to run pathway enrichment and visualize a volcano plot.</li>
            <li>Summarize the literature evidence and show the current technical outputs.</li>
          </ul>
        </div>

        <div className="empty-card accent-slate">
          <div className="empty-card-title">Workspace layout</div>
          <ul>
            <li>The center column keeps the chat and loading status focused.</li>
            <li>The right rail renders all technical artifacts and structured results.</li>
            <li>Each chat preserves its own metadata so follow-up analysis stays grounded.</li>
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
  const [bootLoading, setBootLoading] = useState(true)
  const [loadingChats, setLoadingChats] = useState(false)
  const [loadingMessages, setLoadingMessages] = useState(false)
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
    if (!token) {
      setBootLoading(false)
      return
    }

    ;(async () => {
      try {
        const meResult = await me()
        setUser(meResult)
      } catch {
        setToken('')
      } finally {
        setBootLoading(false)
      }
    })()
  }, [])

  useEffect(() => {
    if (!user) return

    ;(async () => {
      setLoadingChats(true)
      try {
        const chatResult = await fetchChats()
        const nextChats = chatResult.chats || []
        setChats(nextChats)

        const storedChatId = localStorage.getItem(`gea_last_chat_${user.id}`)
        const fallbackChatId = nextChats[0]?.id || null
        const selectedId = storedChatId && nextChats.some((chat) => chat.id === Number(storedChatId))
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
      } finally {
        setLoadingChats(false)
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
      setLoadingMessages(true)
      try {
        const result = await fetchMessages(activeChatId)
        setMessages(result.messages || [])
      } catch (error) {
        setAppError(error.message)
      } finally {
        setLoadingMessages(false)
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

    const optimisticUserMessage = {
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
      optimistic: true,
    }
    setMessages((current) => [...current, optimisticUserMessage])

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
    setAppError('')
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
    setTechnicalMeta(null)
  }

  if (bootLoading) {
    return (
      <div className="auth-shell">
        <div className="auth-card center-card">
          <LoadingState label="Loading your analysis workspace" />
        </div>
      </div>
    )
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

            {authError ? <div className="alert error">{authError}</div> : null}

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

        <div className="sidebar-actions">
          <button className="primary new-chat" onClick={() => handleNewChat('general')} type="button">New analysis</button>
          <button className="ghost new-chat" onClick={() => handleNewChat('literature')} type="button">Literature chat</button>
        </div>

        {loadingChats ? (
          <LoadingState label="Loading chats" />
        ) : (
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
        )}
      </aside>

      <main className="chat-panel">
        <header className="chat-header">
          <div>
            <div className="eyebrow">Current chat</div>
            <h2>{activeChat?.title || 'Select a chat'}</h2>
          </div>
          <div className="header-actions">
            {technicalMeta?.route_rationale ? <div className="header-caption">{technicalMeta.route_rationale}</div> : null}
            <div className="chat-badge">{formatChatMode(activeChat, technicalMeta)}</div>
          </div>
        </header>

        {appError ? <div className="alert error">{appError}</div> : null}

        <div className="workspace-layout">
          <section className="conversation-panel">
            <div className="summary-strip">
              {runMetrics.length > 0 ? (
                runMetrics.map((metric) => (
                  <div className="summary-pill" key={metric.label}>
                    <span>{metric.label}</span>
                    <strong>{metric.value}</strong>
                  </div>
                ))
              ) : (
                <div className="summary-banner">
                  Ask for DEG, enrichment, RWR, literature, or drug matching to populate the technical workspace.
                </div>
              )}
            </div>

            <section className="message-list">
              {loadingMessages ? (
                <LoadingState label="Loading conversation" />
              ) : messages.length === 0 ? (
                <EmptyState />
              ) : (
                <>
                  {messages.map((message, index) => (
                    <MessageBubble
                      key={`${message.role}-${index}-${message.created_at || ''}`}
                      message={message}
                    />
                  ))}
                  {chatBusy ? (
                    <MessageBubble
                      message={{
                        role: 'assistant',
                        content: 'Generating the answer and updating the technical outputs...',
                        created_at: new Date().toISOString(),
                        pending: true,
                      }}
                    />
                  ) : null}
                </>
              )}
            </section>

            <form className="composer" onSubmit={handleSendMessage}>
              <div className={`composer-shell ${chatBusy ? 'busy' : ''}`}>
                <textarea
                  value={messageText}
                  onChange={(event) => setMessageText(event.target.value)}
                  placeholder="Ask a follow-up or start a new analysis..."
                  rows={4}
                  disabled={!activeChatId || chatBusy}
                />
                <div className="composer-actions">
                  <div className="composer-hint">
                    {chatBusy
                      ? 'The agent is working through the current request.'
                      : activeChat
                        ? `Chat updated ${formatTime(activeChat.updated_at)}`
                        : 'Create or select a chat to begin.'}
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
