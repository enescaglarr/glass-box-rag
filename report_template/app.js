const esc = s => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
// dataset entity placeholders like {{Order Number}} get their own treatment
const ents = s => esc(s).replace(/\{\{([^}]+)\}\}/g, (_,n) => `<span class="ent">{{${n}}}</span>`);
const band = c => c >= 0.85 ? 'q-good' : c >= 0.78 ? 'q-mid' : 'q-weak';

const m = DATA.meta;
document.getElementById('s-embed').textContent = m.embedding_model.replace('models/','');
document.getElementById('s-llm').textContent   = m.llm_model.replace('models/','');
document.getElementById('s-index').textContent = `${m.index_vectors} / ${m.dataset_rows.toLocaleString('en-US')} rows`;
document.getElementById('s-dim').textContent   = `${m.dim} · IndexFlatL2`;
document.getElementById('s-intent').textContent = `${m.intents} of 27`;

const SHORT = ['Blocked and urgent','Typos and shorthand','Short and low-stakes','Repeat billing complaint'];

const picker = document.getElementById('picker');
picker.innerHTML = DATA.traces.map((t,i) => `
  <button class="qbtn" role="tab" id="tab-${i}" aria-controls="trace" aria-selected="${i===0}" data-i="${i}">
    <span class="qn">Run ${String(i+1).padStart(2,'0')} — ${esc(SHORT[i])}</span>
    <span class="qt">“${esc(t.query)}”</span>
  </button>`).join('');

// The prompt arrives pre-split by src/trace.py, so the page keeps no copy of the template.
const SLOT = {query:'q', context:'r'};
function promptHTML(t){
  return t.prompt_parts.map(p =>
    p.kind === 'literal' ? esc(p.text) : `<span class="slot ${SLOT[p.kind]}">${esc(p.text)}</span>`
  ).join('');
}

// The drafted reply arrives as plain text; keep its numbered steps and bullets readable.
function replyHTML(raw){
  const lines = raw.split('\n').map(l => l.trim()).filter(Boolean);
  let out = '', list = null;
  const close = () => { if(list){ out += `</${list}>`; list = null; } };
  for(const l of lines){
    const num = l.match(/^\d+\.\s+(.*)$/), bul = l.match(/^[-*]\s+(.*)$/);
    if(num){ if(list!=='ol'){ close(); out += '<ol>'; list='ol'; } out += `<li>${fmt(num[1])}</li>`; }
    else if(bul){ if(list!=='ul'){ close(); out += '<ul>'; list='ul'; } out += `<li>${fmt(bul[1])}</li>`; }
    else { close(); out += `<p>${fmt(l)}</p>`; }
  }
  close();
  return out;
}
const fmt = s => ents(s).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

function render(i){
  const t = DATA.traces[i];
  const hits = t.retrieved.map((r,k) => `
    <article class="hit ${band(r.cos)}">
      <div class="hit-top">
        <span class="rank">#${k+1}</span>
        <div class="hit-main">
          <p class="hit-instr">${ents(r.instruction)}</p>
          <div class="tags">
            <span class="tag intent">${esc(r.intent)}</span>
            <span class="tag">${esc(r.category)}</span>
            <span class="tag" title="Bitext language-variation flags">flags ${esc(r.flags)}</span>
          </div>
        </div>
        <div class="gauge">
          <span class="gauge-num">${r.cos.toFixed(3)}</span>
          <span class="bar"><i style="width:${Math.max(2,(r.cos*100)).toFixed(1)}%"></i></span>
          <span class="gauge-lbl">cos · L2 ${r.distance.toFixed(3)}</span>
        </div>
      </div>
      <details>
        <summary>Response text handed to the model (${r.response.length} chars)</summary>
        <p class="resp">${ents(r.response)}</p>
      </details>
    </article>`).join('');

  const intents = t.intents_hit.length === 1
    ? `all 3 hits agree on <code>${esc(t.intents_hit[0])}</code>`
    : `hits split across ${t.intents_hit.map(x=>`<code>${esc(x)}</code>`).join(' and ')}`;

  document.getElementById('trace').innerHTML = `
  <div class="stage">
    <div class="stage-head">
      <span class="stage-n r">01</span><h3>The query, turned into a vector</h3>
      <span class="stage-meta">task_type=retrieval_query · ${t.timing.embed.toFixed(2)}s</span>
    </div>
    <div class="stage-body">
      <p class="qtext">${esc(t.query)}</p>
      <div class="vecrow">
        <div class="vecbox">
          <p class="eyebrow">First 8 of ${m.dim} dimensions</p>
          <div class="vecvals">
            ${t.query_vector_head.map(v => `<span>${v>=0?' ':''}${v.toFixed(4)}</span>`).join('')}
            <span class="ell">… ${m.dim - 8} more</span>
          </div>
        </div>
        <div class="vecbox">
          <p class="eyebrow">L2 norm</p>
          <div class="vecvals"><span>${t.vector_norm.toFixed(4)}</span></div>
          <p class="note">Gemini truncates to ${m.dim} dimensions without renormalising, so <code>helper.py</code> normalises explicitly. With unit vectors, L2 distance ranks identically to cosine — which is what makes the numbers below comparable across queries.</p>
        </div>
      </div>
    </div>
  </div>

  <div class="stage">
    <div class="stage-head">
      <span class="stage-n r">02</span><h3>What the retriever pulled back</h3>
      <span class="stage-meta">k=3 of ${m.index_vectors} · ${(t.timing.search*1000).toFixed(2)}ms</span>
    </div>
    <div class="stage-body">
      <div class="hits">${hits}</div>
      <p class="note">Best match ${t.best_cos.toFixed(3)} cosine — ${intents}. Only the <em>response</em> field of these three rows travels onward; the instruction, intent and flags shown here are for the human reviewing the draft.</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-head">
      <span class="stage-n g">03</span><h3>The prompt that got assembled</h3>
      <span class="stage-meta">${t.prompt_chars.toLocaleString('en-US')} chars</span>
    </div>
    <div class="stage-body">
      <ul class="legend">
        <li><span class="swatch q"></span>customer query, interpolated verbatim</li>
        <li><span class="swatch r"></span>retrieved responses, as a Python list repr</li>
      </ul>
      <pre class="prompt">${promptHTML(t)}</pre>
      <p class="note">The retrieved text is injected as a stringified Python list, quotes and all — the model reads the <code>[' … ', ' … ']</code> syntax as-is. That is roughly ${Math.round(100*(t.prompt_chars-700)/t.prompt_chars)}% of the prompt: the fixed instructions are the small part.</p>
    </div>
  </div>

  <div class="stage">
    <div class="stage-head">
      <span class="stage-n g">04</span><h3>What came back</h3>
      <span class="stage-meta">temperature 0 · ${t.timing.llm.toFixed(2)}s</span>
    </div>
    <div class="stage-body">
      <div class="outgrid">
        <div class="judg">
          <div class="judg-card">
            <p class="eyebrow">1 · Urgency</p>
            <div class="urg">${[1,2,3,4,5].map(n=>`<i class="${n<=t.urgency?'on':''}" style="height:${40+n*12}%"></i>`).join('')}</div>
            <p class="urg-read"><b>${t.urgency}</b>of 5</p>
          </div>
          <div class="judg-card">
            <p class="eyebrow">2 · Category</p>
            <p class="catout">${esc(t.category_out)}</p>
          </div>
        </div>
        <div class="reply">
          <p class="eyebrow">3 · Drafted reply — the agent accepts this or sends it back with feedback</p>
          ${replyHTML(t.reply)}
        </div>
      </div>
      <dl class="timing">
        <div><dt>Embed query</dt><dd>${t.timing.embed.toFixed(2)}<small>s</small></dd></div>
        <div><dt>FAISS search</dt><dd>${(t.timing.search*1000).toFixed(2)}<small>ms</small></dd></div>
        <div><dt>Generation</dt><dd>${t.timing.llm.toFixed(2)}<small>s</small></dd></div>
        <div><dt>Total</dt><dd>${(t.timing.embed+t.timing.search+t.timing.llm).toFixed(2)}<small>s</small></dd></div>
      </dl>
    </div>
  </div>`;
}

picker.addEventListener('click', e => {
  const b = e.target.closest('.qbtn');
  if(!b) return;
  picker.querySelectorAll('.qbtn').forEach(x => x.setAttribute('aria-selected', x === b));
  render(+b.dataset.i);
});
render(0);
