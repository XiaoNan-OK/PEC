const fs = require('fs');
const path = require('path');
const cp = require('child_process');

const texPath = String.raw`D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\main_develop.tex`;
const bibPath = String.raw`D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu\reference.bib`;
const refRoot = String.raw`D:\Lab\Quantum_Project_Exercise\Reference`;
const outDir = String.raw`D:\Lab\Quantum_Project_Exercise\Essay\Master Essay\HsiaoNanLu`;
const jsonPath = path.join(outDir, 'citation_review.json');
const csvPath = path.join(outDir, 'citation_review.csv');

function read(filePath) {
  return fs.readFileSync(filePath, 'utf8');
}

function walkPdfs(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walkPdfs(full, out);
    else if (entry.isFile() && entry.name.toLowerCase().endsWith('.pdf')) out.push(full);
  }
  return out;
}

function normalize(text) {
  return (text || '')
    .toLowerCase()
    .replace(/\\[a-zA-Z]+\{([^}]*)\}/g, '$1')
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function parseBibEntries(text) {
  const entries = [];
  const regex = /@article\{([^,]+),([\s\S]*?)\n\}/g;
  let m;
  while ((m = regex.exec(text))) {
    const key = m[1].trim();
    const body = m[2];
    const getField = (name) => {
      const mm = body.match(new RegExp(name + String.raw`\s*=\s*\{([\s\S]*?)\}\s*,?`, 'i'));
      if (!mm) return '';
      return mm[1]
        .replace(/\\href\{[^}]*\}\{([^}]*)\}/g, '$1')
        .replace(/[{}]/g, '')
        .replace(/\s+/g, ' ')
        .trim();
    };
    const note = getField('note');
    const doiMatch = note.match(/10\.[^\s}]+/i);
    entries.push({
      key,
      title: getField('title'),
      author: getField('author'),
      journal: getField('journal'),
      volume: getField('volume'),
      pages: getField('pages'),
      year: getField('year'),
      doi: doiMatch ? doiMatch[0] : '',
    });
  }
  return entries;
}

function findPdf(entry, pdfs) {
  const titleWords = normalize(entry.title).split(' ').filter((w) => w.length > 3);
  let best = null;
  for (const pdf of pdfs) {
    const n = normalize(pdf);
    let score = 0;
    for (const word of titleWords) {
      if (n.includes(word)) score += 1;
    }
    if (entry.doi) {
      const doiNorm = normalize(entry.doi);
      const doiTail = normalize(entry.doi.split('/').slice(-1)[0] || '');
      if (doiNorm && n.includes(doiNorm)) score += 20;
      if (doiTail && n.includes(doiTail)) score += 6;
    }
    if (entry.journal && n.includes(normalize(entry.journal))) score += 1;
    if (entry.pages && n.includes(normalize(entry.pages))) score += 0.5;
    if (!best || score > best.score) best = { path: pdf, score };
  }
  return best && best.score >= 4 ? best.path : '';
}

function extractCitationContexts(tex) {
  const contexts = new Map();
  const lines = tex.split(/\r?\n/);
  const citeRegex = /\\cite[t|p]?(\[[^\]]*\])?\{([^}]+)\}/g;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    let m;
    while ((m = citeRegex.exec(line))) {
      const keys = m[2].split(',').map((x) => x.trim()).filter(Boolean);
      for (const key of keys) {
        if (!contexts.has(key)) contexts.set(key, []);
        contexts.get(key).push({
          line: i + 1,
          text: line.trim(),
        });
      }
    }
  }
  return contexts;
}

function summarizeRelation(contextTexts) {
  const joined = contextTexts.join(' ');
  const lower = joined.toLowerCase();
  if (lower.includes('nisq')) return 'Used as background support for the NISQ-era description and current device limitations.';
  if (lower.includes('hardware noise') || lower.includes('noise sources')) return 'Used to support the discussion of hardware noise sources and their impact on circuit performance.';
  if (lower.includes('dynamic decoupling')) return 'Used as the representative citation for dynamic decoupling as a low-level noise-suppression technique.';
  if (lower.includes('error correction')) return 'Used to support the distinction between quantum error correction and near-term error mitigation.';
  if (lower.includes('error mitigation') && lower.includes('near-term alternative')) return 'Used as foundational support for quantum error mitigation as a near-term strategy.';
  if (lower.includes('zero-noise extrapolation') || lower.includes('zne')) return 'Used to support the description, motivation, or limitations of zero-noise extrapolation.';
  if (lower.includes('probabilistic error cancellation') || lower.includes('pec')) return 'Used to support the definition, implementation, or practical overhead of probabilistic error cancellation.';
  if (lower.includes('readout')) return 'Used to support observable-level readout-noise mitigation or readout-error characterization.';
  if (lower.includes('variance') || lower.includes('broader than')) return 'Used to support the claim that mitigation reduces bias but increases estimator variance / sampling cost.';
  if (lower.includes('machine-learning') || lower.includes('adaptive')) return 'Used as future-work support for machine-learning-assisted or adaptive mitigation methods.';
  return 'Used as contextual support for the cited thesis paragraph.';
}

function runPdfToText(pdfPath) {
  const result = cp.spawnSync('pdftotext', ['-f', '1', '-l', '2', '-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 20 * 1024 * 1024,
    windowsHide: true,
  });
  return ((result.stdout || '') + '\n' + (result.stderr || '')).replace(/\u0000/g, '');
}

function splitParagraphs(text) {
  const cleaned = text
    .replace(/\r/g, '')
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n');
  const parts = cleaned
    .split(/\n\s*\n/)
    .map((x) => x.replace(/\s+/g, ' ').trim())
    .filter((x) => x.length > 80);
  return parts;
}

function chooseSupportParagraph(entry, contextTexts, pdfText) {
  const paras = splitParagraphs(pdfText).slice(0, 18);
  const contextWords = new Set(
    normalize([entry.title, entry.journal, ...contextTexts].join(' '))
      .split(' ')
      .filter((w) => w.length > 3)
  );

  let best = paras[0] || '';
  let bestScore = -1;
  for (const para of paras) {
    const pnorm = normalize(para);
    let score = 0;
    for (const word of contextWords) {
      if (pnorm.includes(word)) score += 1;
    }
    if (/abstract|introduction|noise|error|mitigation|quantum|readout|circuit|qubit/i.test(para)) score += 1;
    if (score > bestScore) {
      best = para;
      bestScore = score;
    }
  }
  return best;
}

function inferPaperSection(paragraph) {
  const lower = paragraph.toLowerCase();
  if (lower.includes('abstract')) return 'Abstract';
  if (lower.includes('introduction')) return 'Introduction / opening discussion';
  return 'Opening section of the paper';
}

function toCsv(rows, headers) {
  const esc = (value) => {
    const s = String(value ?? '');
    return '"' + s.replace(/"/g, '""') + '"';
  };
  return [headers.map(esc).join(','), ...rows.map((row) => headers.map((h) => esc(row[h])).join(','))].join('\r\n');
}

const bib = parseBibEntries(read(bibPath));
const tex = read(texPath);
const contexts = extractCitationContexts(tex);
const pdfs = walkPdfs(refRoot);

const rows = bib.map((entry) => {
  const entryContexts = contexts.get(entry.key) || [];
  const contextTexts = entryContexts.map((x) => `L${x.line}: ${x.text}`);
  const pdfPath = findPdf(entry, pdfs);
  const pdfText = pdfPath ? runPdfToText(pdfPath) : '';
  const supportParagraph = pdfText ? chooseSupportParagraph(entry, contextTexts, pdfText) : '';
  const supportSection = supportParagraph ? inferPaperSection(supportParagraph) : '';
  return {
    CitationKey: entry.key,
    PaperTitle: entry.title,
    ThesisCitationLocation: entryContexts.map((x) => `Line ${x.line}`).join('; '),
    ThesisCitingParagraph: contextTexts.join('\n\n'),
    RelationToCitingParagraph: summarizeRelation(contextTexts),
    RelevantPassageInPaper: supportParagraph,
    RelevantPassageSection: supportSection,
    PDFPath: pdfPath,
  };
});

fs.writeFileSync(jsonPath, JSON.stringify(rows, null, 2), 'utf8');
fs.writeFileSync(
  csvPath,
  toCsv(rows, [
    'CitationKey',
    'PaperTitle',
    'ThesisCitationLocation',
    'ThesisCitingParagraph',
    'RelationToCitingParagraph',
    'RelevantPassageSection',
    'RelevantPassageInPaper',
    'PDFPath',
  ]),
  'utf8'
);

console.log(JSON.stringify({ jsonPath, csvPath, rowCount: rows.length }, null, 2));
