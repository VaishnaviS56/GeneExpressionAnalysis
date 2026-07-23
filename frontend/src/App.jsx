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
  const abs = Math.abs(num)
  if (abs < 0.001 || abs >= 1000) return num.toExponential(3)
  return num.toFixed(3)
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

function formatGeneList(value) {
  return Array.isArray(value) && value.length ? value.map((gene) => String(gene)).join(', ') : '-'
}

function csvCell(value) {
  if (value === null || value === undefined) return ''
  const text = Array.isArray(value)
    ? value.join('; ')
    : typeof value === 'object'
      ? JSON.stringify(value)
      : String(value)
  return `"${text.replaceAll('"', '""')}"`
}

function buildCsvContent(rows, columns) {
  const header = columns.map((column) => csvCell(column.label)).join(',')
  const body = rows.map((row) => columns.map((column) => csvCell(column.value(row))).join(','))
  return [header, ...body].join('\r\n')
}

function buildCsvDataUrl(rows, columns) {
  return `data:text/csv;charset=utf-8,${encodeURIComponent(buildCsvContent(rows, columns))}`
}

function DownloadLink({ children, disabled = false, filename, href }) {
  if (disabled || !href) {
    return <span className="ghost mini disabled">{children}</span>
  }
  return (
    <a className="ghost mini" download={filename} href={href}>
      {children}
    </a>
  )
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

  function isTableSeparator(line) {
    const cells = splitTableRow(line)
    return cells.length > 1 && cells.every((cell) => /^:?-{3,}:?$/.test(cell.trim()))
  }

  function isTableRow(line) {
    return line.includes('|') && splitTableRow(line).length > 1
  }

  function splitTableRow(line) {
    return String(line || '')
      .trim()
      .replace(/^\|/, '')
      .replace(/\|$/, '')
      .split('|')
      .map((cell) => cell.trim())
  }

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

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const rawLine = lines[lineIndex]
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

    const nextLine = lines[lineIndex + 1]?.trim() || ''
    if (isTableRow(trimmed) && isTableSeparator(nextLine)) {
      flushParagraph()
      flushList()
      flushQuote()
      const headers = splitTableRow(trimmed)
      const rows = []
      lineIndex += 1
      while (lineIndex + 1 < lines.length && isTableRow(lines[lineIndex + 1])) {
        lineIndex += 1
        const row = splitTableRow(lines[lineIndex])
        rows.push(headers.map((_, columnIndex) => row[columnIndex] || ''))
      }
      blocks.push({ type: 'table', headers, rows })
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
        if (block.type === 'table') {
          return (
            <div className="markdown-table-wrap" key={`block-${index}`}>
              <table className="markdown-table">
                <thead>
                  <tr>
                    {block.headers.map((header, headerIndex) => (
                      <th key={`table-head-${index}-${headerIndex}`}>
                        {renderInlineMarkdown(header, `table-head-${index}-${headerIndex}`)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, rowIndex) => (
                    <tr key={`table-row-${index}-${rowIndex}`}>
                      {row.map((cell, cellIndex) => (
                        <td key={`table-cell-${index}-${rowIndex}-${cellIndex}`}>
                          {renderInlineMarkdown(cell, `table-cell-${index}-${rowIndex}-${cellIndex}`)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
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
        {message.role === 'assistant' && message.meta ? <TechnicalOutput meta={message.meta} /> : null}
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

function TechnicalSection({ title, subtitle, children }) {
  return (
    <section className="technical-section">
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
  const downloadSrc = buildAssetUrl(path, { download: true })

  return (
    <div className="artifact-block">
      <div className="artifact-head">
        <div>
          <div className="artifact-label">{label}</div>
          <div className="artifact-path">{path}</div>
        </div>
        <div className="artifact-actions">
          <a className="ghost mini" href={src} rel="noreferrer" target="_blank">Open</a>
          <a className="ghost mini" download href={downloadSrc}>Download</a>
        </div>
      </div>
      {mode === 'html' ? (
        <iframe className="artifact-frame" src={src} title={label} />
      ) : (
        <img className="artifact-image" src={src} alt={label} loading="lazy" />
      )}
    </div>
  )
}

function TechnicalOutput({ meta }) {
  if (!meta || typeof meta !== 'object') {
    return null
  }

  const analysisArm = String(meta.analysis_arm || '').trim().toLowerCase()
  const isDegTurn = analysisArm === 'srp'
  const isPathwayTurn = analysisArm === 'pathway'
  const isRwrTurn = analysisArm === 'memory_rwr' || meta.rwr_result_is_current === true
  const isLiteratureTurn = ['disease', 'research_literature', 'literature'].includes(analysisArm)
  const isVisualTurn = analysisArm === 'visualize'
  const isNetworkTurn = isRwrTurn || isVisualTurn
  const isHypothesisTurn = analysisArm === 'hypothesis'

  const degUpRows = isDegTurn && Array.isArray(meta.deg_analysis?.upregulated_rows)
    ? meta.deg_analysis.upregulated_rows.slice(0, 10)
    : isDegTurn && Array.isArray(meta.deg_gene_records)
      ? meta.deg_gene_records
        .filter((row) => Number(row?.log2FoldChange) > 0)
        .sort((a, b) => Number(b?.log2FoldChange || 0) - Number(a?.log2FoldChange || 0))
        .slice(0, 10)
      : []
  const degDownRows = isDegTurn && Array.isArray(meta.deg_analysis?.downregulated_rows)
    ? meta.deg_analysis.downregulated_rows.slice(0, 10)
    : isDegTurn && Array.isArray(meta.deg_gene_records)
      ? meta.deg_gene_records
        .filter((row) => Number(row?.log2FoldChange) < 0)
        .sort((a, b) => Number(a?.log2FoldChange || 0) - Number(b?.log2FoldChange || 0))
        .slice(0, 10)
      : []
  const degDownloadRows = isDegTurn && Array.isArray(meta.deg_gene_records)
    ? meta.deg_gene_records
    : isDegTurn && Array.isArray(meta.deg_analysis?.rows)
      ? meta.deg_analysis.rows
      : []
  const enrichrLibs = isPathwayTurn && meta.enrichr && typeof meta.enrichr === 'object' && meta.enrichr.libraries && typeof meta.enrichr.libraries === 'object'
    ? meta.enrichr.libraries
    : {}
  const rwrRows = isRwrTurn && Array.isArray(meta.rwr_genes) ? meta.rwr_genes.slice(0, 10) : []
  const openTargets = analysisArm === 'opentargets' && meta.opentargets_result && typeof meta.opentargets_result === 'object' ? meta.opentargets_result : null
  const primeKg = analysisArm === 'primekg' && meta.primekg_result && typeof meta.primekg_result === 'object' ? meta.primekg_result : null
  const l1000 = analysisArm === 'l1000cds2' && meta.l1000cds2_result && typeof meta.l1000cds2_result === 'object' ? meta.l1000cds2_result : null
  const pubchem = analysisArm === 'pubchem' && meta.pubchem_result && typeof meta.pubchem_result === 'object' ? meta.pubchem_result : null
  const hypothesis = isHypothesisTurn && meta.hypothesis_result && typeof meta.hypothesis_result === 'object' ? meta.hypothesis_result : null
  const druggability = analysisArm === 'druggability' && meta.druggability_result && typeof meta.druggability_result === 'object' ? meta.druggability_result : null
  const pdbVisualization = analysisArm === 'pdb_visualizer' && meta.pdb_visualization_result && typeof meta.pdb_visualization_result === 'object' ? meta.pdb_visualization_result : null
  const rankedPapers = isLiteratureTurn && Array.isArray(meta.ranked_openalex_papers) ? meta.ranked_openalex_papers.slice(0, 5) : []
  const scannedPapers = isLiteratureTurn && Array.isArray(meta.openalex_papers) ? meta.openalex_papers.slice(0, 5) : []
  const literaturePoints = isLiteratureTurn && Array.isArray(meta.literature_key_points) ? meta.literature_key_points.slice(0, 5) : []
  const literatureReferences = isLiteratureTurn && Array.isArray(meta.literature_references) ? meta.literature_references.slice(0, 8) : []
  const network = isNetworkTurn && meta.network && typeof meta.network === 'object' ? meta.network : null
  const topDegree = Array.isArray(network?.top_degree) ? network.top_degree.slice(0, 10) : []
  const pyvisHtmlPath = isVisualTurn && typeof meta.pyvis_html_path === 'string' ? meta.pyvis_html_path : ''
  const keggPath = isVisualTurn && typeof meta.kegg_pathway_path === 'string' ? meta.kegg_pathway_path : ''
  const volcanoPath = (isVisualTurn || isDegTurn) && typeof meta.volcano_plot_path === 'string' ? meta.volcano_plot_path : ''
  const graphmlPath = isNetworkTurn && typeof meta.graphml_path === 'string' ? meta.graphml_path : ''
  const pdbViewerPath = typeof druggability?.pdb_viewer_html_path === 'string' ? druggability.pdb_viewer_html_path : ''
  const fixedPdbPath = typeof druggability?.fixed_pdb_path === 'string' ? druggability.fixed_pdb_path : ''
  const rawPdbPath = typeof druggability?.raw_pdb_path === 'string' ? druggability.raw_pdb_path : ''
  const dogsiteTablePath = typeof druggability?.result_table_path === 'string' ? druggability.result_table_path : ''
  const proteinViewerPath = typeof pdbVisualization?.pdb_viewer_html_path === 'string' ? pdbVisualization.pdb_viewer_html_path : ''
  const proteinPdbPath = typeof pdbVisualization?.pdb_path === 'string'
    ? pdbVisualization.pdb_path
    : typeof pdbVisualization?.raw_pdb_path === 'string'
      ? pdbVisualization.raw_pdb_path
      : ''
  const volcanoMode = volcanoPath.toLowerCase().endsWith('.html') || volcanoPath.toLowerCase().endsWith('.htm') ? 'html' : 'image'
  const requestedCellLines = Array.isArray(l1000?.requested_cell_lines) ? l1000.requested_cell_lines : []
  const pathwayDownloadRows = Object.entries(enrichrLibs).flatMap(([library, terms]) => (
    Array.isArray(terms)
      ? terms.map((term, index) => ({ ...term, library, rank: index + 1 }))
      : []
  ))
  const l1000DownloadRows = Array.isArray(l1000?.top_drugs) ? l1000.top_drugs : []
  const degCsvHref = degDownloadRows.length > 0
    ? buildCsvDataUrl(degDownloadRows, [
      { label: 'gene', value: (row) => row.gene || row.hgnc_symbol || row.external_gene_name || row.Ensembl || '' },
      { label: 'log2FoldChange', value: (row) => formatNumber(row.log2FoldChange) },
      { label: 'pvalue', value: (row) => formatNumber(row.pvalue) },
      { label: 'description', value: (row) => row.description },
    ])
    : ''
  const pathwayCsvHref = pathwayDownloadRows.length > 0
    ? buildCsvDataUrl(pathwayDownloadRows, [
      { label: 'library', value: (row) => row.library },
      { label: 'rank', value: (row) => row.rank },
      { label: 'term', value: (row) => row.term || row.t },
      { label: 'p_value', value: (row) => row.p_value ?? row.p },
      { label: 'adjusted_p_value', value: (row) => row.adjusted_p_value ?? row.adj },
      { label: 'combined_score', value: (row) => row.combined_score ?? row.cs },
      { label: 'overlapping_genes', value: (row) => row.overlapping_genes || row.genes },
      { label: 'n_overlap_genes', value: (row) => row.n_overlap_genes || (Array.isArray(row.overlapping_genes) ? row.overlapping_genes.length : '') },
    ])
    : ''
  const l1000CsvHref = l1000DownloadRows.length > 0
    ? buildCsvDataUrl(l1000DownloadRows, [
      { label: 'drug', value: (row) => row.name },
      { label: 'pert_id', value: (row) => row.pert_id },
      { label: 'best_rank', value: (row) => row.best_rank },
      { label: 'best_score', value: (row) => row.best_score },
      { label: 'signature_count', value: (row) => row.signature_count },
      { label: 'cell_lines', value: (row) => row.cell_lines },
    ])
    : ''
  const hasDownloads = Boolean(degCsvHref || volcanoPath || l1000CsvHref || pathwayCsvHref || graphmlPath || pdbViewerPath || fixedPdbPath || rawPdbPath || dogsiteTablePath || proteinViewerPath || proteinPdbPath)
  const hasVisuals = Boolean(pyvisHtmlPath || keggPath || volcanoPath || pdbViewerPath || proteinViewerPath)
  const hasNetwork = Boolean(network && (network.nodes || network.edges || topDegree.length > 0))
  const hasLiterature = Boolean(
    rankedPapers.length > 0
    || scannedPapers.length > 0
    || literaturePoints.length > 0
    || literatureReferences.length > 0
    || meta.literature_summary,
  )
  const hasDeg = degUpRows.length > 0 || degDownRows.length > 0
  const hasPathway = Object.values(enrichrLibs).some((terms) => Array.isArray(terms) && terms.length > 0)
  const hasRwr = rwrRows.length > 0
  const hasL1000 = Boolean(l1000 && (Array.isArray(l1000.top_drugs) || l1000.message))
  const hasOpenTargets = Boolean(openTargets)
  const hasPrimeKg = Boolean(primeKg && (primeKg.answer || primeKg.cypher || (Array.isArray(primeKg.rows) && primeKg.rows.length > 0)))
  const hasPubchem = Boolean(pubchem)
  const hasHypothesis = Boolean(hypothesis && (hypothesis.hypothesis_summary || (Array.isArray(hypothesis.hypotheses) && hypothesis.hypotheses.length > 0)))
  const hasDruggability = Boolean(druggability && (Array.isArray(druggability.top_pockets) || pdbViewerPath || druggability.message))
  const hasPdbVisualization = Boolean(pdbVisualization && (proteinViewerPath || proteinPdbPath || pdbVisualization.message))

  if (!(
    hasDownloads
    || hasVisuals
    || hasNetwork
    || hasLiterature
    || hasDeg
    || hasPathway
    || hasRwr
    || hasL1000
    || hasOpenTargets
    || hasPrimeKg
    || hasPubchem
    || hasHypothesis
    || hasDruggability
    || hasPdbVisualization
  )) {
    return null
  }

  return (
    <div className="technical-output">
      <section className="technical-panel">
        <div className="technical-hero">
          <div className="eyebrow">Technical outputs</div>
          <div className="technical-hero-row">
            <h3>{formatArm(meta.analysis_arm)}</h3>
            <div className="chat-badge">{formatArm(meta.analysis_arm)}</div>
          </div>
          <p>Structured tables and files from this response.</p>
        </div>

        {hasDownloads && (
          <TechnicalSection title="Downloads" subtitle="Export the current technical outputs for downstream analysis.">
            <div className="download-grid">
              <DownloadLink
                disabled={!degCsvHref}
                filename="deg_genes.csv"
                href={degCsvHref}
              >
                DEG genes CSV
              </DownloadLink>
              <DownloadLink
                disabled={!volcanoPath}
                filename={volcanoMode === 'html' ? 'deg_volcano.html' : 'deg_volcano.png'}
                href={volcanoPath ? buildAssetUrl(volcanoPath, { download: true }) : ''}
              >
                Volcano plot
              </DownloadLink>
              <DownloadLink
                disabled={!l1000CsvHref}
                filename="l1000cds2_results.csv"
                href={l1000CsvHref}
              >
                L1000 table CSV
              </DownloadLink>
              <DownloadLink
                disabled={!pathwayCsvHref}
                filename="pathway_enrichment.csv"
                href={pathwayCsvHref}
              >
                Pathway CSV
              </DownloadLink>
              <DownloadLink
                disabled={!graphmlPath}
                filename="string_network.graphml"
                href={graphmlPath ? buildAssetUrl(graphmlPath, { download: true }) : ''}
              >
                STRING graph GraphML
              </DownloadLink>
              <DownloadLink
                disabled={!pdbViewerPath}
                filename={`${druggability?.gene || 'protein'}_pocket_viewer.html`}
                href={pdbViewerPath ? buildAssetUrl(pdbViewerPath, { download: true }) : ''}
              >
                PDB viewer
              </DownloadLink>
              <DownloadLink
                disabled={!fixedPdbPath}
                filename={`${druggability?.gene || 'protein'}_fixed.pdb`}
                href={fixedPdbPath ? buildAssetUrl(fixedPdbPath, { download: true }) : ''}
              >
                Fixed PDB
              </DownloadLink>
              <DownloadLink
                disabled={!rawPdbPath}
                filename={`${druggability?.gene || 'protein'}_raw.pdb`}
                href={rawPdbPath ? buildAssetUrl(rawPdbPath, { download: true }) : ''}
              >
                Raw PDB
              </DownloadLink>
              <DownloadLink
                disabled={!dogsiteTablePath}
                filename={`${druggability?.gene || 'protein'}_dogsite_table.txt`}
                href={dogsiteTablePath ? buildAssetUrl(dogsiteTablePath, { download: true }) : ''}
              >
                DoGSite table
              </DownloadLink>
              <DownloadLink
                disabled={!proteinViewerPath}
                filename={`${pdbVisualization?.gene || pdbVisualization?.uniprot_id || 'protein'}_pdb_viewer.html`}
                href={proteinViewerPath ? buildAssetUrl(proteinViewerPath, { download: true }) : ''}
              >
                Protein viewer
              </DownloadLink>
              <DownloadLink
                disabled={!proteinPdbPath}
                filename={`${pdbVisualization?.gene || pdbVisualization?.uniprot_id || 'protein'}.pdb`}
                href={proteinPdbPath ? buildAssetUrl(proteinPdbPath, { download: true }) : ''}
              >
                Protein PDB
              </DownloadLink>
            </div>
          </TechnicalSection>
        )}

        {hasVisuals && (
          <TechnicalSection title="Visual outputs" subtitle="Generated artifacts are embedded here, matching the Streamlit workspace.">
            <div className="artifact-grid">
              <ArtifactPreview label="Network visualization" mode="html" path={pyvisHtmlPath} />
              <ArtifactPreview label="KEGG pathway" mode="image" path={keggPath} />
              <ArtifactPreview label="Volcano plot" mode={volcanoMode} path={volcanoPath} />
              <ArtifactPreview label="PDB pocket viewer" mode="html" path={pdbViewerPath} />
              <ArtifactPreview label="Protein PDB viewer" mode="html" path={proteinViewerPath} />
            </div>
          </TechnicalSection>
        )}

        {hasNetwork && (
          <TechnicalSection title="Network summary" subtitle="Graph context from the current analysis run.">
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
          </TechnicalSection>
        )}

        {hasLiterature && (
          <TechnicalSection
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
                    <span>{[row.authors, row.journal, row.source, row.year, row.pmid ? `PMID ${row.pmid}` : '', row.doi ? `DOI ${row.doi}` : '', row.url].filter(Boolean).join(' | ')}</span>
                    {row.note ? <span>{row.note}</span> : null}
                  </div>
                ))}
              </div>
            )}
          </TechnicalSection>
        )}

        {hasDeg && (
          <TechnicalSection title="Differential expression" subtitle="Top DEG rows from the current comparison.">
            {degUpRows.length > 0 && (
              <div className="technical-table">
                <h4>Top up-regulated genes</h4>
                <div className="technical-row technical-row-three technical-head">
                  <span>Gene</span>
                  <span>log2FC</span>
                  <span>p-value</span>
                </div>
                {degUpRows.map((row, index) => (
                  <div className="technical-row technical-row-three" key={`${row.gene || 'up-deg'}-${index}`}>
                    <span>{row.gene || '-'}</span>
                    <span>{formatNumber(row.log2FoldChange)}</span>
                    <span>{formatNumber(row.pvalue)}</span>
                  </div>
                ))}
              </div>
            )}
            {degDownRows.length > 0 && (
              <div className="technical-table">
                <h4>Top down-regulated genes</h4>
                <div className="technical-row technical-row-three technical-head">
                  <span>Gene</span>
                  <span>log2FC</span>
                  <span>p-value</span>
                </div>
                {degDownRows.map((row, index) => (
                  <div className="technical-row technical-row-three" key={`${row.gene || 'down-deg'}-${index}`}>
                    <span>{row.gene || '-'}</span>
                    <span>{formatNumber(row.log2FoldChange)}</span>
                    <span>{formatNumber(row.pvalue)}</span>
                  </div>
                ))}
              </div>
            )}
          </TechnicalSection>
        )}

        {Object.entries(enrichrLibs).map(([library, terms]) => (
          Array.isArray(terms) && terms.length > 0 ? (
            <TechnicalSection key={library} title={library} subtitle="Pathway enrichment results.">
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
                    <span>{formatGeneList(term.overlapping_genes || term.genes)}</span>
                  </div>
                ))}
              </div>
            </TechnicalSection>
          ) : null
        ))}

        {hasRwr && (
          <TechnicalSection title="Random Walk with Restart" subtitle="Ranked targets from the stored or current seed genes.">
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
          </TechnicalSection>
        )}

        {hasL1000 && (
          <TechnicalSection title="L1000CDS2 matches" subtitle="Top compound matches from the current reversal search.">
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
          </TechnicalSection>
        )}

        {hasOpenTargets && (
          <TechnicalSection title="OpenTargets" subtitle="Association summary from the current OpenTargets lookup.">
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
          </TechnicalSection>
        )}

        {hasPrimeKg && (
          <TechnicalSection title="PrimeKG" subtitle="Knowledge graph answer and returned rows.">
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
          </TechnicalSection>
        )}

        {hasPubchem && (
          <TechnicalSection title="PubChem" subtitle="Compound metadata and synthesis context.">
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
          </TechnicalSection>
        )}

        {hasDruggability && (
          <TechnicalSection title="Druggability" subtitle="Structure-backed DoGSite pocket results and local PDB artifacts.">
            <div className="summary-pill-grid">
              <div className="summary-pill">
                <span>Gene</span>
                <strong>{druggability.gene || '-'}</strong>
              </div>
              <div className="summary-pill">
                <span>UniProt</span>
                <strong>{druggability.uniprot_id || '-'}</strong>
              </div>
              <div className="summary-pill">
                <span>Source</span>
                <strong>{druggability.structure_source || '-'}</strong>
              </div>
              <div className="summary-pill">
                <span>DoGSite job</span>
                <strong>{druggability.dogsite_job_id || '-'}</strong>
              </div>
            </div>
            {Array.isArray(druggability.top_pockets) && druggability.top_pockets.length > 0 ? (
              <div className="technical-table">
                <div className="technical-row technical-head technical-row-pocket">
                  <span>Pocket</span>
                  <span>Drug score</span>
                  <span>Volume</span>
                  <span>Residue PDB</span>
                </div>
                {druggability.top_pockets.slice(0, 10).map((row, index) => (
                  <div className="technical-row technical-row-pocket" key={`${row.name || 'pocket'}-${index}`}>
                    <span>{row.name || `Pocket ${index + 1}`}</span>
                    <span>{formatNumber(row.drug_score)}</span>
                    <span>{formatNumber(row.volume)}</span>
                    <span>{row.residue_file || '-'}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p>{druggability.message || 'No pocket rows available.'}</p>
            )}
          </TechnicalSection>
        )}

        {hasPdbVisualization && (
          <TechnicalSection title="PDB visualization" subtitle="Fetched protein structure and generated interactive 3D viewer.">
            <div className="summary-pill-grid">
              <div className="summary-pill">
                <span>Gene</span>
                <strong>{pdbVisualization.gene || '-'}</strong>
              </div>
              <div className="summary-pill">
                <span>UniProt</span>
                <strong>{pdbVisualization.uniprot_id || '-'}</strong>
              </div>
              <div className="summary-pill">
                <span>Source</span>
                <strong>{pdbVisualization.structure_source || '-'}</strong>
              </div>
              <div className="summary-pill">
                <span>PDB</span>
                <strong>{pdbVisualization.pdb_id || '-'}</strong>
              </div>
            </div>
            {proteinPdbPath ? <p><strong>PDB file:</strong> {proteinPdbPath}</p> : <p>{pdbVisualization.message || 'No PDB file available.'}</p>}
          </TechnicalSection>
        )}

        {hasHypothesis && (
          <TechnicalSection title="Experimental hypotheses" subtitle="LLM-generated validation ideas grounded in prior chat context and stored memory.">
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
          </TechnicalSection>
        )}

      </section>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="eyebrow">GEA Agent</div>
      <h3>Ask a biomedical question to begin.</h3>
      <p>Your question will appear as a bubble here, followed by the agent answer, then your next follow-up.</p>
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
  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) || null,
    [activeChatId, chats],
  )

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
    setAppError('')
    try {
      const created = await createChat('New chat', 'general')
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

        <div className="sidebar-actions">
          <button className="primary new-chat" onClick={handleNewChat} type="button">New chat</button>
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
        </header>

        {appError ? <div className="alert error">{appError}</div> : null}

        <section className="conversation-panel">
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
                      content: 'Generating the answer...',
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
                placeholder="Ask a follow-up or start a new chat..."
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
      </main>
    </div>
  )
}

export default App
