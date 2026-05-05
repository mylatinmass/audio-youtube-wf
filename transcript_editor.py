import argparse
import json
import os
import webbrowser
from html import escape
from pathlib import Path


DEFAULT_THRESHOLD = 0.995


def _json_for_script(data):
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def generate_transcript_editor(json_path, output_path=None, default_threshold=DEFAULT_THRESHOLD, audio_path=None):
    json_path = os.path.abspath(json_path)
    if output_path is None:
        output_path = os.path.join(os.path.dirname(json_path), "transcript_editor.html")
    output_path = os.path.abspath(output_path)
    audio_url = ""
    audio_label = "No audio file was attached to this editor."
    if audio_path:
        audio_path = os.path.abspath(audio_path)
        if os.path.exists(audio_path):
            audio_url = Path(audio_path).resolve().as_uri()
            audio_label = audio_path
        else:
            audio_label = f"Audio file not found: {audio_path}"

    with open(json_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    segments = transcript.get("segments") or []
    if not segments:
        raise ValueError(f"No segments found in {json_path}")

    word_count = sum(len(seg.get("words") or []) for seg in segments)
    low_count = sum(
        1
        for seg in segments
        for word in seg.get("words") or []
        if isinstance(word.get("probability"), (int, float)) and word["probability"] < default_threshold
    )

    title = "Homily Transcript Editor"
    json_name = os.path.basename(json_path)
    data_json = _json_for_script(transcript)
    audio_source = f'<audio id="audioPlayer" controls preload="metadata" src="{escape(audio_url, quote=True)}"></audio>'
    if not audio_url:
        audio_source = '<audio id="audioPlayer" controls preload="metadata"></audio>'

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1e2428;
      --muted: #66737c;
      --line: #d9e1e5;
      --soft: #f5f7f8;
      --panel: #ffffff;
      --accent: #0b6bcb;
      --warn: #fff0a8;
      --warn-line: #d59700;
      --bad: #b3261e;
      --good: #146c43;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #edf2f4;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 4;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(8px);
    }}
    .bar {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) auto;
      gap: 16px;
      align-items: center;
      max-width: 1180px;
      margin: 0 auto;
      padding: 14px 18px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .path {{
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      justify-content: flex-end;
      align-items: center;
    }}
    .audio-panel {{
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      gap: 12px;
      align-items: center;
      max-width: 1180px;
      margin: 0 auto;
      padding: 0 18px 14px;
    }}
    audio {{
      display: block;
      width: 100%;
      height: 38px;
    }}
    .audio-actions {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      align-items: center;
    }}
    .audio-meta {{
      grid-column: 1 / -1;
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    input[type="range"] {{ width: 170px; }}
    input[type="number"] {{
      width: 78px;
      padding: 7px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font: inherit;
    }}
    button {{
      appearance: none;
      border: 1px solid #aab7c0;
      background: #fff;
      color: var(--ink);
      padding: 8px 11px;
      border-radius: 6px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }}
    button.primary {{
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }}
    button:disabled {{
      cursor: not-allowed;
      opacity: 0.55;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 18px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(130px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .stat strong {{
      display: block;
      font-size: 18px;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .segment {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 12px 0;
      overflow: hidden;
    }}
    .segment-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      background: var(--soft);
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    textarea {{
      display: block;
      width: 100%;
      min-height: 84px;
      resize: vertical;
      border: 0;
      border-bottom: 1px solid var(--line);
      padding: 12px;
      font: 17px/1.45 Georgia, "Times New Roman", serif;
      color: var(--ink);
      outline: none;
    }}
    .preview {{
      padding: 12px;
      font: 16px/1.7 Georgia, "Times New Roman", serif;
      white-space: pre-wrap;
    }}
    .word {{
      border-radius: 4px;
      padding: 1px 2px;
    }}
    .word.low {{
      background: var(--warn);
      box-shadow: inset 0 -2px 0 var(--warn-line);
      cursor: pointer;
    }}
    .word.active {{
      outline: 2px solid var(--accent);
      outline-offset: 2px;
    }}
    .warning {{
      display: none;
      padding: 8px 12px;
      color: var(--bad);
      background: #fff7f6;
      border-top: 1px solid #f1c8c4;
      font-size: 13px;
    }}
    .warning.show {{ display: block; }}
    .status {{
      min-width: 190px;
      color: var(--muted);
      font-size: 13px;
      text-align: right;
    }}
    .status.good {{ color: var(--good); }}
    .status.bad {{ color: var(--bad); }}
    @media (max-width: 820px) {{
      .bar {{ grid-template-columns: 1fr; }}
      .audio-panel {{ grid-template-columns: 1fr; }}
      .audio-actions {{ justify-content: flex-start; }}
      .controls {{ justify-content: flex-start; }}
      .stats {{ grid-template-columns: repeat(2, minmax(130px, 1fr)); }}
      .status {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="bar">
      <div>
        <h1>{escape(title)}</h1>
        <div class="path">{escape(json_path)}</div>
      </div>
      <div class="controls">
        <label>
          Confidence
          <input id="thresholdRange" type="range" min="0" max="1" step="0.001" value="{default_threshold:.3f}">
          <input id="thresholdNumber" type="number" min="0" max="1" step="0.001" value="{default_threshold:.3f}">
        </label>
        <button id="downloadButton" type="button">Download JSON</button>
        <button id="saveButton" class="primary" type="button">Save JSON</button>
        <div id="status" class="status">Ready</div>
      </div>
    </div>
    <div class="audio-panel">
      {audio_source}
      <div class="audio-actions">
        <button id="prevHighlightButton" type="button">Prev Highlight</button>
        <button id="playHighlightButton" type="button">Play Highlight</button>
        <button id="nextHighlightButton" type="button">Next Highlight</button>
      </div>
      <div class="audio-meta">
        Audio: {escape(audio_label)}
        <span id="currentHighlight">No highlight selected</span>
      </div>
    </div>
  </header>

  <main>
    <section class="stats" aria-label="Transcript stats">
      <div class="stat"><strong id="segmentCount">{len(segments)}</strong><span>segments</span></div>
      <div class="stat"><strong id="wordCount">{word_count}</strong><span>timed words</span></div>
      <div class="stat"><strong id="lowCount">{low_count}</strong><span>highlighted</span></div>
      <div class="stat"><strong id="editedCount">0</strong><span>edited segments</span></div>
    </section>
    <section id="segments"></section>
  </main>

  <script id="transcript-data" type="application/json">{data_json}</script>
  <script>
    const sourcePath = {json.dumps(json_path)};
    const suggestedName = {json.dumps(json_name)};
    const transcript = JSON.parse(document.getElementById("transcript-data").textContent);
    const originalTexts = (transcript.segments || []).map(seg => segmentText(seg));
    const segmentsRoot = document.getElementById("segments");
    const audioPlayer = document.getElementById("audioPlayer");
    const thresholdRange = document.getElementById("thresholdRange");
    const thresholdNumber = document.getElementById("thresholdNumber");
    const lowCount = document.getElementById("lowCount");
    const editedCount = document.getElementById("editedCount");
    const statusEl = document.getElementById("status");
    const currentHighlight = document.getElementById("currentHighlight");
    const prevHighlightButton = document.getElementById("prevHighlightButton");
    const playHighlightButton = document.getElementById("playHighlightButton");
    const nextHighlightButton = document.getElementById("nextHighlightButton");
    let highlightedLocations = [];
    let activeHighlightIndex = -1;

    function segmentText(seg) {{
      const text = (seg.text || "").replace(/\\s+/g, " ").trim();
      if (text) return text;
      return (seg.words || []).map(word => word.word || "").join(" ").replace(/\\s+/g, " ").trim();
    }}

    function timeLabel(seconds) {{
      const value = Math.max(0, Number(seconds || 0));
      const mins = Math.floor(value / 60);
      const secs = Math.floor(value % 60).toString().padStart(2, "0");
      return `${{mins}}:${{secs}}`;
    }}

    function tokens(text) {{
      return (text || "").trim().match(/\\S+/g) || [];
    }}

    function setStatus(message, kind = "") {{
      statusEl.textContent = message;
      statusEl.className = `status ${{kind}}`;
    }}

    function seekAudio(seconds, play = true) {{
      if (!audioPlayer || !audioPlayer.src) {{
        setStatus("No audio loaded", "bad");
        return;
      }}
      audioPlayer.currentTime = Math.max(0, Number(seconds || 0) - 0.25);
      if (play) {{
        audioPlayer.play().catch(() => setStatus("Click play in audio controls", "bad"));
      }}
    }}

    function setHighlightButtons() {{
      const hasHighlights = highlightedLocations.length > 0;
      prevHighlightButton.disabled = !hasHighlights;
      playHighlightButton.disabled = !hasHighlights;
      nextHighlightButton.disabled = !hasHighlights;
      if (!hasHighlights) {{
        currentHighlight.textContent = " No highlights at this threshold";
      }} else if (activeHighlightIndex === -1) {{
        currentHighlight.textContent = " Select a highlighted word or use Next Highlight";
      }}
    }}

    function markActiveWord(location) {{
      document.querySelectorAll(".word.active").forEach(el => el.classList.remove("active"));
      if (!location) return;
      const selector = `.word[data-segment-index="${{location.segmentIndex}}"][data-word-index="${{location.wordIndex}}"]`;
      const el = document.querySelector(selector);
      if (!el) return;
      el.classList.add("active");
      el.scrollIntoView({{ behavior: "smooth", block: "center" }});
    }}

    function setActiveHighlight(index, play = true) {{
      if (!highlightedLocations.length) {{
        setHighlightButtons();
        return;
      }}
      activeHighlightIndex = (index + highlightedLocations.length) % highlightedLocations.length;
      const location = highlightedLocations[activeHighlightIndex];
      currentHighlight.textContent = ` Highlight ${{activeHighlightIndex + 1}} of ${{highlightedLocations.length}} at ${{timeLabel(location.start)}}`;
      markActiveWord(location);
      seekAudio(location.start, play);
      setHighlightButtons();
    }}

    function setActiveHighlightByWord(segmentIndex, wordIndex) {{
      const index = highlightedLocations.findIndex(item => item.segmentIndex === segmentIndex && item.wordIndex === wordIndex);
      if (index === -1) {{
        const word = (transcript.segments[segmentIndex].words || [])[wordIndex];
        seekAudio(word ? word.start : 0, true);
        return;
      }}
      setActiveHighlight(index, true);
    }}

    function renderSegments() {{
      segmentsRoot.textContent = "";
      (transcript.segments || []).forEach((seg, index) => {{
        const section = document.createElement("article");
        section.className = "segment";

        const head = document.createElement("div");
        head.className = "segment-head";
        head.innerHTML = `<span>Segment ${{index + 1}}</span><span>${{timeLabel(seg.start)}} to ${{timeLabel(seg.end)}}</span>`;

        const textarea = document.createElement("textarea");
        textarea.value = segmentText(seg);
        textarea.dataset.index = index;

        const preview = document.createElement("div");
        preview.className = "preview";
        preview.dataset.index = index;

        const warning = document.createElement("div");
        warning.className = "warning";
        warning.dataset.index = index;

        textarea.addEventListener("input", () => {{
          applyTextToSegment(index, textarea.value);
          renderPreview(index);
          updateStats();
          setStatus("Unsaved changes");
        }});

        section.append(head, textarea, preview, warning);
        segmentsRoot.append(section);
        applyTextToSegment(index, textarea.value);
        renderPreview(index);
      }});
      updateStats();
    }}

    function applyTextToSegment(index, rawText) {{
      const seg = transcript.segments[index];
      const cleanText = (rawText || "").replace(/\\s+/g, " ").trim();
      seg.text = cleanText;

      const newTokens = tokens(cleanText);
      const words = seg.words || [];
      const warning = document.querySelector(`.warning[data-index="${{index}}"]`);
      if (!words.length) {{
        if (warning) warning.classList.remove("show");
        return;
      }}

      if (newTokens.length === words.length) {{
        words.forEach((word, wordIndex) => {{
          const oldWord = word.word || "";
          const prefix = (oldWord.match(/^\\s+/) || [" "])[0];
          word.word = `${{prefix}}${{newTokens[wordIndex]}}`;
        }});
        if (warning) warning.classList.remove("show");
      }} else if (warning) {{
        warning.textContent = `Word timing map kept unchanged: ${{newTokens.length}} edited words for ${{words.length}} timed words.`;
        warning.classList.add("show");
      }}
    }}

    function renderPreview(index) {{
      const seg = transcript.segments[index];
      const preview = document.querySelector(`.preview[data-index="${{index}}"]`);
      if (!preview) return;
      preview.textContent = "";

      const words = seg.words || [];
      if (!words.length) {{
        preview.textContent = seg.text || "";
        return;
      }}

      const threshold = Number(thresholdNumber.value);
      words.forEach((word, wordIndex) => {{
        const span = document.createElement("span");
        const prob = Number(word.probability);
        const isLow = Number.isFinite(prob) && prob < threshold;
        span.className = isLow ? "word low" : "word";
        span.dataset.segmentIndex = index;
        span.dataset.wordIndex = wordIndex;
        if (isLow) {{
          span.addEventListener("click", () => setActiveHighlightByWord(index, wordIndex));
        }}
        span.textContent = word.word || "";
        span.title = Number.isFinite(prob) ? `${{(prob * 100).toFixed(2)}}% at ${{timeLabel(word.start)}}` : `No probability at ${{timeLabel(word.start)}}`;
        preview.appendChild(span);
      }});
    }}

    function updateStats() {{
      const threshold = Number(thresholdNumber.value);
      let highlighted = 0;
      let edited = 0;
      highlightedLocations = [];
      (transcript.segments || []).forEach((seg, index) => {{
        if (segmentText(seg) !== originalTexts[index]) edited += 1;
        (seg.words || []).forEach((word, wordIndex) => {{
          const prob = Number(word.probability);
          if (Number.isFinite(prob) && prob < threshold) {{
            highlighted += 1;
            highlightedLocations.push({{
              segmentIndex: index,
              wordIndex,
              start: Number(word.start || 0),
              end: Number(word.end || word.start || 0),
              probability: prob
            }});
          }}
        }});
      }});
      lowCount.textContent = highlighted;
      editedCount.textContent = edited;
      if (activeHighlightIndex >= highlightedLocations.length) {{
        activeHighlightIndex = highlightedLocations.length - 1;
      }}
      setHighlightButtons();
    }}

    function syncThreshold(value) {{
      const bounded = Math.min(1, Math.max(0, Number(value) || 0));
      const fixed = bounded.toFixed(3);
      thresholdRange.value = fixed;
      thresholdNumber.value = fixed;
      (transcript.segments || []).forEach((_, index) => renderPreview(index));
      updateStats();
      if (activeHighlightIndex >= 0) markActiveWord(highlightedLocations[activeHighlightIndex]);
    }}

    function rebuildTranscript() {{
      document.querySelectorAll("textarea[data-index]").forEach(textarea => {{
        applyTextToSegment(Number(textarea.dataset.index), textarea.value);
      }});
      transcript.homily_text = (transcript.segments || []).map(seg => seg.text || "").join(" ").replace(/\\s+/g, " ").trim();
      return JSON.stringify(transcript, null, 2);
    }}

    function downloadJson() {{
      const blob = new Blob([rebuildTranscript()], {{ type: "application/json" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = suggestedName;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setStatus("Downloaded", "good");
    }}

    async function saveJson() {{
      const content = rebuildTranscript();
      try {{
        if (window.showOpenFilePicker) {{
          const [handle] = await window.showOpenFilePicker({{
            multiple: false,
            startIn: "downloads",
            types: [{{ description: "JSON", accept: {{ "application/json": [".json"] }} }}]
          }});
          const writable = await handle.createWritable();
          await writable.write(content);
          await writable.close();
          setStatus("Saved", "good");
          return;
        }}
        if (window.showSaveFilePicker) {{
          const handle = await window.showSaveFilePicker({{
            suggestedName,
            types: [{{ description: "JSON", accept: {{ "application/json": [".json"] }} }}]
          }});
          const writable = await handle.createWritable();
          await writable.write(content);
          await writable.close();
          setStatus("Saved", "good");
          return;
        }}
        downloadJson();
      }} catch (error) {{
        if (error && error.name === "AbortError") {{
          setStatus("Save canceled");
        }} else {{
          console.error(error);
          setStatus("Save failed", "bad");
          downloadJson();
        }}
      }}
    }}

    thresholdRange.addEventListener("input", event => syncThreshold(event.target.value));
    thresholdNumber.addEventListener("input", event => syncThreshold(event.target.value));
    document.getElementById("downloadButton").addEventListener("click", downloadJson);
    document.getElementById("saveButton").addEventListener("click", saveJson);
    prevHighlightButton.addEventListener("click", () => setActiveHighlight(activeHighlightIndex <= 0 ? highlightedLocations.length - 1 : activeHighlightIndex - 1));
    playHighlightButton.addEventListener("click", () => setActiveHighlight(activeHighlightIndex === -1 ? 0 : activeHighlightIndex));
    nextHighlightButton.addEventListener("click", () => setActiveHighlight(activeHighlightIndex + 1));

    renderSegments();
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a local homily transcript editor HTML file.")
    parser.add_argument("json_path", help="Path to video_script.json")
    parser.add_argument("-o", "--output", help="Output HTML path")
    parser.add_argument("--audio", help="Path to the homily audio file")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--open", action="store_true", help="Open the generated editor in the default browser")
    args = parser.parse_args()

    output_path = generate_transcript_editor(args.json_path, args.output, args.threshold, args.audio)
    print(output_path)
    if args.open:
        webbrowser.open(f"file://{output_path}")


if __name__ == "__main__":
    main()
