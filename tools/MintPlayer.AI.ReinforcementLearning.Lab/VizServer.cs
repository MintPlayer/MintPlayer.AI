using System.Net;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using MintPlayer.AI.ReinforcementLearning.Core.Telemetry;

/// <summary>
/// A self-contained, dependency-free live-network viewer hosted <b>by the training process itself</b>: a tiny
/// <see cref="HttpListener"/> on localhost that serves one HTML page (<c>GET /</c>) and streams telemetry to it
/// over a <b>WebSocket</b> (<c>GET /ws</c>). It samples an <see cref="INetworkTelemetrySource"/> (the running
/// campaign) on a fixed cadence — a pull model, so it works for every trainer without any of them knowing it
/// exists. A WebSocket (rather than one-way SSE) keeps the channel bidirectional, so the viewer can later send
/// control messages back (pause/step, change cadence, pick a layer) without changing the transport.
/// <para>
/// Fully async: each viewer has its own bounded outbound queue drained by an async send pump, and a background
/// sample loop awaits the cadence timer — there is no blocking network I/O on any hot path. Fault-tolerant: a
/// viewer that never connects, disconnects, or falls behind is dropped (its bounded queue discards stale frames),
/// so telemetry never blocks or breaks training. Development-only — never wired into the deployed web app.
/// </para>
/// </summary>
internal sealed class VizServer : IDisposable
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web)
    {
        // Metrics can be non-finite (e.g. ε = NaN for a supervised campaign, eval = -Infinity before the first
        // eval); emit them as the named JSON literals rather than throwing — the viewer renders non-finite as "—".
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
    };

    /// <summary>A connected viewer: its socket plus a bounded outbound queue drained by one async send pump.
    /// The queue drops the oldest frame under backpressure, so a slow viewer can never stall the sampler.</summary>
    private sealed class Client(WebSocket socket)
    {
        public WebSocket Socket { get; } = socket;
        public Channel<byte[]> Outbox { get; } = Channel.CreateBounded<byte[]>(
            new BoundedChannelOptions(32) { FullMode = BoundedChannelFullMode.DropOldest, SingleReader = true });
    }

    private readonly HttpListener _listener = new();
    private readonly List<Client> _clients = [];
    private readonly Lock _gate = new();
    private readonly CancellationTokenSource _shutdown = new();
    private readonly INetworkTelemetrySource _source;
    private readonly int _intervalMs;
    private volatile bool _running = true;

    public string Url { get; }

    private VizServer(int port, INetworkTelemetrySource source, int intervalMs)
    {
        _source = source;
        _intervalMs = intervalMs;
        Url = $"http://localhost:{port}/";
        _listener.Prefixes.Add(Url);
    }

    /// <summary>Starts the viewer on <paramref name="port"/> and begins sampling <paramref name="source"/>.</summary>
    public static VizServer Start(int port, INetworkTelemetrySource source, int intervalMs = 150)
    {
        var server = new VizServer(port, source, intervalMs);
        server._listener.Start();
        _ = Task.Run(server.AcceptLoop);
        _ = Task.Run(server.SampleLoop);
        return server;
    }

    private async Task AcceptLoop()
    {
        while (_running)
        {
            HttpListenerContext ctx;
            try { ctx = await _listener.GetContextAsync(); }
            catch { return; } // listener stopped
            _ = Task.Run(() => Handle(ctx));
        }
    }

    private async Task Handle(HttpListenerContext ctx)
    {
        string path = ctx.Request.Url?.AbsolutePath ?? "/";
        if (path == "/ws" && ctx.Request.IsWebSocketRequest) { await ServeWebSocket(ctx); return; }

        var html = Encoding.UTF8.GetBytes(Page);
        ctx.Response.ContentType = "text/html; charset=utf-8";
        ctx.Response.ContentLength64 = html.Length;
        try { await ctx.Response.OutputStream.WriteAsync(html); } catch { /* client went away */ }
        finally { ctx.Response.Close(); }
    }

    private async Task ServeWebSocket(HttpListenerContext ctx)
    {
        WebSocket socket;
        try { socket = (await ctx.AcceptWebSocketAsync(subProtocol: null)).WebSocket; }
        catch { ctx.Response.Abort(); return; }

        var client = new Client(socket);
        lock (_gate) _clients.Add(client);

        // Give the newcomer the current graph at once (before the next sample tick), so it never sits blank.
        if (CurrentTopology() is { } topo) client.Outbox.Writer.TryWrite(topo);

        var pump = Task.Run(() => SendPump(client));

        // Hold the connection open and drain anything the viewer sends (control messages will arrive here later);
        // this loop also detects the close so the client can be dropped.
        var buffer = new byte[4096];
        try
        {
            while (socket.State == WebSocketState.Open && !_shutdown.IsCancellationRequested)
            {
                var result = await socket.ReceiveAsync(buffer, _shutdown.Token);
                if (result.MessageType == WebSocketMessageType.Close) break;
            }
        }
        catch { /* client dropped or shutting down */ }
        finally { Drop(client); await pump; }
    }

    /// <summary>Drains one client's outbound queue, awaiting each send — the only place a socket is written.</summary>
    private async Task SendPump(Client client)
    {
        try
        {
            await foreach (var message in client.Outbox.Reader.ReadAllAsync(_shutdown.Token))
                await client.Socket.SendAsync(new ArraySegment<byte>(message), WebSocketMessageType.Text, endOfMessage: true, _shutdown.Token);
        }
        catch { /* client gone or shutting down */ }
        finally { Drop(client); }
    }

    /// <summary>Samples the source on the cadence and broadcasts topology (when it changes) + a weight frame.</summary>
    private async Task SampleLoop()
    {
        string? lastTopologyJson = null;
        while (!_shutdown.IsCancellationRequested)
        {
            try { await Task.Delay(_intervalMs, _shutdown.Token); }
            catch { break; }

            // Nothing to do (and nothing to pay) while no one is watching.
            lock (_gate) { if (_clients.Count == 0) continue; }

            try
            {
                var parameters = _source.SnapshotParameters();
                if (parameters is null || parameters.Count == 0) continue;

                var topology = NetworkInspector.Describe(parameters, _source.NetKind);
                string topologyJson = JsonSerializer.Serialize(topology, Json);
                if (topologyJson != lastTopologyJson)
                {
                    lastTopologyJson = topologyJson;
                    Broadcast(Envelope("topology", topologyJson));
                }

                var frame = NetworkInspector.CaptureFrame(parameters, _source.Sample());
                Broadcast(Envelope("frame", JsonSerializer.Serialize(frame, Json)));
            }
            catch { /* transient (e.g. net swapped mid-sample) — skip this frame */ }
        }
    }

    /// <summary>The current graph as a ready-to-send topology envelope, or null if the net doesn't exist yet.</summary>
    private byte[]? CurrentTopology()
    {
        try
        {
            var parameters = _source.SnapshotParameters();
            if (parameters is null || parameters.Count == 0) return null;
            return Envelope("topology", JsonSerializer.Serialize(NetworkInspector.Describe(parameters, _source.NetKind), Json));
        }
        catch { return null; }
    }

    private void Broadcast(byte[] message)
    {
        Client[] snapshot;
        lock (_gate) snapshot = [.. _clients];
        foreach (var c in snapshot) c.Outbox.Writer.TryWrite(message); // non-blocking; DropOldest handles backpressure
    }

    // WebSocket has no SSE-style event names, so each message self-describes: {"type":<t>,"data":<payload>}.
    // `json` is already-serialized, so this splices it in without a second serialize pass.
    private static byte[] Envelope(string type, string json)
        => Encoding.UTF8.GetBytes($"{{\"type\":\"{type}\",\"data\":{json}}}");

    private void Drop(Client client)
    {
        lock (_gate) { if (!_clients.Remove(client)) return; }
        client.Outbox.Writer.TryComplete(); // ends the send pump's ReadAllAsync
        try { client.Socket.Abort(); } catch { }
        try { client.Socket.Dispose(); } catch { }
    }

    public void Dispose()
    {
        _running = false;
        _shutdown.Cancel();
        lock (_gate)
        {
            foreach (var c in _clients) { c.Outbox.Writer.TryComplete(); try { c.Socket.Abort(); c.Socket.Dispose(); } catch { } }
            _clients.Clear();
        }
        try { _listener.Stop(); } catch { }
        _listener.Close();
        _shutdown.Dispose();
    }

    // The whole viewer: one HTML file with inline CSS/JS, no external requests (CSP-safe, works offline). It lays
    // the net out from the `topology` message, repaints on every `frame`, and — for people new to neural nets —
    // explains each part on hover. The WebSocket auto-reconnects on drop so restarting a run re-attaches the viewer.
    private const string Page = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Network — live training</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         background: #0b0f14; color: #d7dee7; }
  header { padding: 12px 18px; border-bottom: 1px solid #1c2530; display: flex; flex-wrap: wrap;
           gap: 18px 28px; align-items: baseline; }
  h1 { font-size: 15px; font-weight: 600; margin: 0; color: #7cc7ff; letter-spacing: .3px; }
  .stat b { color: #f0f4f9; font-weight: 600; }
  .stat span { color: #6b7683; }
  #bar { height: 4px; background: #1c2530; border-radius: 3px; overflow: hidden; flex: 1 1 220px;
         min-width: 160px; align-self: center; }
  #bar > i { display: block; height: 100%; width: 0; background: linear-gradient(90deg,#2f81f7,#7cc7ff); }
  #status { color: #6b7683; }
  main { padding: 16px; }
  canvas { width: 100%; height: auto; display: block; background: #0b0f14; }
  #net { cursor: crosshair; }
  .legend { color: #6b7683; padding: 6px 18px 14px; font-size: 12px; }
  #tip { position: fixed; z-index: 10; max-width: 320px; pointer-events: none; display: none;
         background: #14202c; border: 1px solid #2f4658; border-radius: 6px; padding: 8px 10px;
         color: #dbe6f0; font-size: 12px; line-height: 1.45; box-shadow: 0 6px 20px rgba(0,0,0,.5); }
  #tip b { color: #7cc7ff; }
</style>
</head>
<body>
<header>
  <h1>▚ live network</h1>
  <span class="stat">step <b id="step">—</b><span id="steptot"></span></span>
  <span class="stat">loss <b id="loss">—</b></span>
  <span class="stat">eval <b id="eval">—</b></span>
  <span class="stat">ε <b id="eps">—</b></span>
  <div id="bar"><i></i></div>
  <span id="status">connecting…</span>
</header>
<main>
  <canvas id="net"></canvas>
</main>
<div class="legend">Hover any neuron, connection, or heatmap to learn what it means. Nodes are layers of neurons
  (capped for display); edge brightness = weight magnitude. For dueling/value/policy nets the final small columns are
  the network's output "heads".</div>
<canvas id="loss-spark" style="width:100%;height:60px;margin-top:4px"></canvas>
<div id="tip"></div>

<script>
const $ = id => document.getElementById(id);
let topo = null, frame = null;
const lossHist = [];
// hit-test regions rebuilt each draw() for the hover tooltips
let hitNodes = [], hitGaps = [], hitHeat = [];

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => $('status').textContent = 'streaming';
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type === 'topology') { topo = m.data; frame = null; draw(); }
    else if (m.type === 'frame') {
      frame = m.data;
      if (isFinite(frame.loss)) { lossHist.push(frame.loss); if (lossHist.length > 400) lossHist.shift(); }
      updateStats(); draw(); drawSpark();
    }
  };
  ws.onclose = () => { $('status').textContent = 'reconnecting…'; setTimeout(connect, 1000); };
  ws.onerror = () => { try { ws.close(); } catch (_) {} };
}

function updateStats() {
  if (!frame) return;
  $('step').textContent = frame.step.toLocaleString();
  $('steptot').textContent = frame.maxSteps ? ' / ' + frame.maxSteps.toLocaleString() : '';
  $('loss').textContent = isFinite(frame.loss) ? frame.loss.toFixed(4) : '—';
  $('eval').textContent = isFinite(frame.eval) ? frame.eval.toFixed(2) : '—';
  $('eps').textContent  = isFinite(frame.epsilon) ? frame.epsilon.toFixed(3) : '—';
  const pct = frame.maxSteps ? Math.min(1, frame.step / frame.maxSteps) : 0;
  $('bar').firstElementChild.style.width = (pct * 100) + '%';
}

// magnitude → colour ramp (dark→cyan→white), t in [0,1]
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(20 + t*t*235), g = Math.round(60 + t*175), b = Math.round(90 + t*165);
  return `rgb(${r},${g},${b})`;
}

function draw() {
  const cv = $('net'), ctx = cv.getContext('2d');
  const W = cv.clientWidth || 900, dpr = window.devicePixelRatio || 1;
  hitNodes = []; hitGaps = []; hitHeat = [];
  if (!topo || !topo.layers.length) {
    cv.width = W*dpr; cv.height = 200*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.clearRect(0,0,W,200);
    ctx.fillStyle = '#6b7683'; ctx.font = '13px monospace';
    ctx.fillText('waiting for the network topology…', 20, 40);
    return;
  }
  const cols = [topo.inputSize, ...topo.layers.map(l => l.outputSize)];
  const capN = 16;
  const H = 320, padX = 60, padTop = 30, nodeR = 6;
  cv.width = W*dpr; cv.height = (H+150)*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,H+150);

  const colX = i => padX + (W-2*padX) * (cols.length===1?0.5:i/(cols.length-1));
  const disp = n => Math.min(n, capN);
  const nodeY = (k, n) => { const d = disp(n); return padTop + (H-2*padTop) * (d===1?0.5:k/(d-1)); };
  const frameLayers = frame ? frame.layers : null;

  // Edges: per layer, brightness from the downsampled heatmap sampled onto the capped node pairs.
  for (let li=0; li<topo.layers.length; li++) {
    const inN = disp(cols[li]), outN = disp(cols[li+1]);
    const lf = frameLayers && frameLayers[li];
    let hmax = 1e-9, heat=null, hR=0, hC=0;
    if (lf) { heat=lf.heat; hR=lf.hRows; hC=lf.hCols; for (const v of heat) if (v>hmax) hmax=v; }
    const x1 = colX(li), x2 = colX(li+1);
    for (let a=0; a<inN; a++) for (let b=0; b<outN; b++) {
      let t = 0.12;
      if (heat && hR && hC) { const hr = Math.floor(a*hR/inN), hc = Math.floor(b*hC/outN); t = heat[hr*hC+hc]/hmax; }
      ctx.strokeStyle = ramp(t); ctx.globalAlpha = 0.10 + 0.7*t; ctx.lineWidth = 0.5 + 1.2*t;
      ctx.beginPath(); ctx.moveTo(x1, nodeY(a,cols[li])); ctx.lineTo(x2, nodeY(b,cols[li+1])); ctx.stroke();
    }
    hitGaps.push({ x1, x2, li, rows: cols[li], cols: cols[li+1], meanAbs: lf ? lf.wMeanAbs : NaN });
  }
  ctx.globalAlpha = 1;

  // Nodes + column labels.
  for (let ci=0; ci<cols.length; ci++) {
    const n = cols[ci], d = disp(n), x = colX(ci);
    const role = ci===0 ? 'input' : (ci===cols.length-1 ? 'output' : 'hidden');
    for (let k=0;k<d;k++){ const y = nodeY(k,n);
      ctx.beginPath(); ctx.arc(x, y, nodeR, 0, 7); ctx.fillStyle='#12202f';
      ctx.strokeStyle='#3a5570'; ctx.lineWidth=1.5; ctx.fill(); ctx.stroke();
      hitNodes.push({ x, y, r: nodeR+4, ci, role, n }); }
    ctx.fillStyle='#9aa7b4'; ctx.font='11px monospace'; ctx.textAlign='center';
    ctx.fillText(ci===0?`in ${n}`:(ci===cols.length-1?`out ${n}`:`${n}`), x, padTop-12);
    if (n>capN){ ctx.fillStyle='#5a6570'; ctx.fillText(`(${n}, capped)`, x, H-padTop+22); }
  }

  // Per-layer weight heatmaps in a strip beneath the graph.
  if (frameLayers) {
    const stripY = H+8, cell = 3, gap = 24;
    let x = padX;
    ctx.textAlign='left';
    for (let li=0; li<frameLayers.length; li++) {
      const lf = frameLayers[li]; let hmax=1e-9; for (const v of lf.heat) if (v>hmax) hmax=v;
      for (let r=0;r<lf.hRows;r++) for (let c=0;c<lf.hCols;c++){
        ctx.fillStyle = ramp(lf.heat[r*lf.hCols+c]/hmax);
        ctx.fillRect(x + c*cell, stripY + r*cell, cell, cell);
      }
      hitHeat.push({ x, y: stripY, w: lf.hCols*cell, h: lf.hRows*cell, li, rows: lf.rows, cols: lf.cols, meanAbs: lf.wMeanAbs });
      ctx.fillStyle='#6b7683'; ctx.font='10px monospace';
      ctx.fillText(`L${li} ${lf.rows}×${lf.cols}`, x, stripY + lf.hRows*cell + 12);
      ctx.fillText(`|w| ${lf.wMeanAbs.toFixed(3)}`, x, stripY + lf.hRows*cell + 24);
      x += lf.hCols*cell + gap;
    }
  }
}

function drawSpark() {
  const cv = $('loss-spark'), ctx = cv.getContext('2d');
  const W = cv.clientWidth || 900, Hs = 60, dpr = window.devicePixelRatio||1;
  cv.width = W*dpr; cv.height = Hs*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,W,Hs);
  if (lossHist.length < 2) return;
  const lo = Math.min(...lossHist), hi = Math.max(...lossHist), rng = (hi-lo)||1;
  ctx.strokeStyle='#2f81f7'; ctx.lineWidth=1.5; ctx.beginPath();
  lossHist.forEach((v,i)=>{ const x=W*i/(lossHist.length-1), y=Hs-6-(Hs-12)*(v-lo)/rng;
    i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.stroke();
  ctx.fillStyle='#6b7683'; ctx.font='10px monospace';
  ctx.fillText('loss '+lo.toFixed(3)+' … '+hi.toFixed(3), 6, 12);
}

// ---- beginner-friendly hover tooltips -------------------------------------------------
function outputMeaning() {
  const os = topo ? topo.outputSize : 0, k = (topo && topo.netKind) || '';
  if (os === 1) return "a single number — the network's estimate of how good the current situation is (its \"value\").";
  if (k.indexOf('policy') >= 0) return "the network's preference for one possible move; higher means more likely to be chosen.";
  return "the network's score (a \"Q-value\") for one possible action; the AI plays the highest-scoring one.";
}
function nodeTip(hn) {
  if (hn.role === 'input')
    return `<b>Input neuron</b><br>One number the AI reads from the game each step — its "observation" (e.g. a board cell, a distance, a speed). This layer takes in ${hn.n} such numbers.`;
  if (hn.role === 'output')
    return `<b>Output neuron</b><br>` + outputMeaning();
  return `<b>Hidden neuron</b><br>It blends the previous layer's numbers using learned "weights", then keeps the result only if positive (a ReLU). Each one learns to detect a useful pattern. ${hn.n} in this layer.`;
}
function gapTip(g) {
  const w = isFinite(g.meanAbs) ? `<br>Average strength right now: <b>${g.meanAbs.toFixed(3)}</b>.` : '';
  return `<b>Connections (weights)</b><br>The ${g.rows}×${g.cols} numbers linking these two layers. <b>Learning is the AI slowly adjusting them.</b> Brighter lines = larger weights.${w}`;
}
function heatTip(h) {
  return `<b>Weight heatmap — layer L${h.li}</b><br>Every cell is one connection's strength (brighter = stronger), for all ${h.rows}×${h.cols} weights. Watch the pattern sharpen as the AI learns.`;
}
function showTip(html, ev) {
  const t = $('tip'); t.innerHTML = html; t.style.display = 'block';
  const x = Math.min(ev.clientX + 14, innerWidth - t.offsetWidth - 8);
  const y = Math.min(ev.clientY + 16, innerHeight - t.offsetHeight - 8);
  t.style.left = x + 'px'; t.style.top = y + 'px';
}
function hideTip() { $('tip').style.display = 'none'; }

$('net').addEventListener('mousemove', ev => {
  const x = ev.offsetX, y = ev.offsetY;
  for (const hn of hitNodes) { const dx=x-hn.x, dy=y-hn.y; if (dx*dx+dy*dy <= hn.r*hn.r) return showTip(nodeTip(hn), ev); }
  for (const h of hitHeat) { if (x>=h.x && x<=h.x+h.w && y>=h.y && y<=h.y+h.h) return showTip(heatTip(h), ev); }
  for (const g of hitGaps) { if (x>g.x1+8 && x<g.x2-8 && y>=10 && y<=320) return showTip(gapTip(g), ev); }
  hideTip();
});
$('net').addEventListener('mouseleave', hideTip);

addEventListener('resize', () => { draw(); drawSpark(); });
draw();
connect();
</script>
</body>
</html>
""";
}
