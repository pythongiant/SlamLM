# SlamLM menubar ↔ bridge protocol

Frozen contract. The Swift app (`menubar/Sources/SlamLM/`) and the Python bridge
(`menubar/sidecar/slam_lm_bridge/`) both implement this file. Change it here
first, then both sides.

The app spawns the bridge as a child process and exchanges **newline-delimited
JSON**: one object per line, UTF-8, no embedded newlines. `stderr` is free-form
logging and is never parsed.

```
app ──stdin──▶  {"id":1,"cmd":"catalog"}
app ◀─stdout──  {"id":1,"ok":true,"result":{"models":[…]}}
bridge ◀──────  {"event":"metrics","data":{…}}        # unsolicited, 5 Hz
```

## Envelopes

| shape | meaning |
|---|---|
| `{"id":Int,"cmd":String,…fields}` | request |
| `{"id":Int,"ok":true,"result":{…}}` | success reply, exactly one per request |
| `{"id":Int,"ok":false,"error":String}` | failure reply, exactly one per request |
| `{"event":String,"data":{…}}` | unsolicited event, `id`-less |

Replies may interleave with events. `result` is `{}` when a command returns nothing.

## Commands (app → bridge)

| `cmd` | fields | `result` |
|---|---|---|
| `hello` | — | `{"protocol":1,"bridge":String,"hardware":Hardware,"categories":[String]}` |
| `catalog` | — | `{"models":[Model,…]}` — rescans the local model stores |
| `load` | `model:String` | `{"model":String,"loadMs":Float,"memoryBytes":Int}` |
| `unload` | — | `{}` — frees the model and empties the MLX cache |
| `generate` | `prompt:String`, `max_tokens:Int`, `request:Int`, `chat:Bool` | `{"request":Int}` immediately; output arrives as `token` events, terminated by `request_end` |
| `cancel` | — | `{"cancelled":Bool}` |
| `serve` | `port:Int` | `{"port":Int,"url":String}` — start the OpenAI-compatible HTTP API (idempotent) |
| `stop_serve` | — | `{"stopped":Bool}` |
| `ping` | — | `{"pong":true,"ts":Float}` |

`load` while a different model is loaded replaces it. `load` of an already-loaded
model is a no-op returning the original `loadMs` and `0` for `memoryBytes`.
`generate` with no model loaded returns `ok:false`.

`generate` with `chat: true` renders `prompt` as a single `user` chat message
through the tokenizer's chat template before generating — the same rendering the
HTTP API applies — so a panel prompt reads as a chat turn rather than as text to
continue. Without `chat` (or with any falsey value) the prompt is generated
verbatim, which is what prefill/throughput measurement wants. A model whose
tokenizer has no chat template falls back to the message content unchanged.

## Events (bridge → app)

| `event` | `data` | cadence |
|---|---|---|
| `metrics` | `LiveMetrics` | 5 Hz, always (memory fields are valid with no model loaded) |
| `state` | `{"status":Status,"phase":Phase,"model":String?,"message":String?}` | on every transition |
| `token` | `{"request":Int,"index":Int,"text":String,"ttsMs":Float}` | per generated token |
| `request_end` | `RequestRecord` | once per request, including on `cancel` and on error (`finish_reason:"error"`) |
| `log` | `{"level":"info"\|"warn"\|"error","message":String}` | as needed; surfaced in the app's log strip |

## Types

Keys are **exact**. Byte counts are JSON integers (safe: < 2^53).

### `Model`
```jsonc
{
  "id": "mlx-community/Qwen3-1.7B-4bit",  // repo id, unique key
  "name": "Qwen3 1.7B",                  // display name, derived from id
  "params": "1.7B",                      // from config.json, "" if unknown
  "quant": "4-bit",                      // from quantization block, "fp16" if absent
  "bytes": 1800000000,                   // on-disk size of the weight files
  "path": "/Users/…/models--mlx-community--Qwen3-1.7B-4bit/snapshots/<sha>",
  "categories": ["Chat","Instruct"],     // see derivation rules
  "architecture": "qwen3",               // config.json "model_type"
  "contextLength": 40960,                // config.json max_position_embeddings, 0 if unknown
  "hasChatTemplate": true,               // chat_template present in tokenizer_config.json
  "lastUsed": 1758700000.0                // epoch s from the bridge's state file, 0.0 if never
}
```

### `Hardware`
```jsonc
{
  "model": "Mac16,1",                    // sysctl hw.model
  "chip": "Apple M4",                    // sysctl machdep.cpu.brand_string
  "gpuCores": 10,                        // system_profiler SPDisplaysDataType, 0 if unavailable
  "totalBytes": 17179869184,             // mx.device_info()["memory_size"]
  "recommendedBytes": 12713115648,       // mx.device_info()["max_recommended_working_set_size"]
  "memoryActiveBytes": 0,                // mx.get_active_memory()
  "memoryPeakBytes": 0,                  // mx.get_peak_memory()
  "cacheBytes": 0,                       // mx.get_cache_memory()
  "processRssBytes": 0,                  // resident set of the bridge process
  "systemUsedBytes": 0,                  // physical - free - cached
  "systemAppBytes": 0,                   // anonymous, minus purgeable
  "systemWiredBytes": 0,                 // unpageable
  "systemCompressedBytes": 0,            // held by the compressor
  "systemCachedBytes": 0,                // file-backed + purgeable
  "systemSwapBytes": 0                   // swap in use
}
```
In `hello`, the live memory fields are a first sample. In `metrics` they are the
current values. The system fields are machine-wide, not per-process: they are the
same quantities Activity Monitor reports, so the panel can show what the machine
is actually using. `memoryTotalBytes` is physical RAM (`mx.device_info()`
`memory_size`), which on Apple silicon is the same pool the GPU uses — there is no
separate GPU memory to report.

### `LiveMetrics`
```jsonc
{
  "ts": 1758700000.123,
  "memoryActiveBytes": 0, "memoryPeakBytes": 0, "memoryCacheBytes": 0,
  "memoryTotalBytes": 17179869184, "memoryRecommendedBytes": 12713115648,
  "processRssBytes": 0,
  "systemUsedBytes": 0,    // physical - free - cached, as Activity Monitor reports
  "systemAppBytes": 0, "systemWiredBytes": 0, "systemCompressedBytes": 0,
  "systemCachedBytes": 0, "systemSwapBytes": 0,
  "decodeTps": 0.0,        // tokens/s over the trailing 2 s window, 0.0 when idle
  "prefillTps": 0.0,       // prompt tokens/s of the most recent request, 0.0 if none
  "ttftMs": 0.0,           // time to first token of the most recent request, 0.0 if none
  "tokensGenerated": 0,    // session total
  "requests": 0,           // session total, counting failures
  "status": "idle",        // "idle"|"loading"|"ready"|"generating"|"error"
  "phase": "idle",         // "idle"|"load"|"prefill"|"decode"
  "model": null,           // loaded model id, or null
  "loadMs": 0.0            // load time of the loaded model, 0.0 if none
}
```

### `RequestRecord`
```jsonc
{
  "request": 3, "model": "mlx-community/Qwen3-1.7B-4bit",
  "promptTokens": 24, "genTokens": 128,
  "ttftMs": 41.2,          // submit → first token
  "prefillTps": 582.5,     // promptTokens / ttft
  "decodeTps": 77.4,       // (genTokens-1) / (last token t - first token t)
  "peakMemBytes": 1868000000,
  "startedAt": 1758700000.123,
  "totalMs": 1690.4,
  "finishReason": "length" // "length"|"stop"|"cancel"|"error"
}
```

## Metric definitions

- **TTFT** is measured wall-clock in the bridge from request submit to the first
  `stream_generate` yield, which includes the prefill forward pass. `prefillTps =
  promptTokens / (ttftMs/1000)`.
- **decodeTps** per request is `(genTokens - 1) / (t_first - t_last)`; the live
  `decodeTps` in `metrics` is a 2 s trailing window so the UI number is stable
  while streaming.
- **Memory**: `memoryActiveBytes` is MLX-owned live arrays; `cacheBytes` is MLX's
  reusable buffer pool; `processRssBytes` is the bridge's own resident set and
  therefore includes both, plus non-MLX allocations. Nothing else is counted, and
  no value is estimated — a metric with no data reports `0` and the UI renders an
  empty state rather than a placeholder number.
- **System memory** (machine-wide, sampled from Mach `host_statistics64` page
  counts and `sysctl`, matching Activity Monitor's definitions):
  `systemAppBytes` = anonymous (`internal − purgeable`) pages, `systemWiredBytes`
  = `wire` pages, `systemCompressedBytes` = `compressor` pages,
  `systemCachedBytes` = `external + purgeable` pages, `systemUsedBytes` =
  `physical − free − cached`, `systemSwapBytes` = `vm.swapusage` used. These are
  what the machine is using, not what this bridge is using — the bridge's own
  share is `memoryActiveBytes + cacheBytes`.
- **Footer bars.** The panel's "Memory" bar reports `systemUsedBytes` against
  physical RAM and its "Model" bar reports `memoryActiveBytes + memoryCacheBytes`
  against the same total, because Apple silicon has one unified pool: the model's
  MLX allocation is part of Memory, not a second pool. Totals print in binary
  gigabytes (`16 GB` for 16 GiB) while byte amounts print in decimal, so the
  numbers line up with what Activity Monitor shows. The raw metrics are unchanged
  and the analytics strip shows each of them separately.

## Category derivation (no invented labels)

Per model, from real signals only:

| category | rule |
|---|---|
| `Instruct` | `chat_template` present in `tokenizer_config.json` |
| `Chat` | causal LM architecture that is not embedding/audio/vision-only |
| `Code` | `code`/`coder` in the repo id |
| `Vision` | `vision_config` present, or `-vl`/`vision` in the repo id |
| `Embedding` | architecture contains `Bert`/`Embedding`, or `-embed` in the repo id |
| `Audio` | architecture contains `Whisper`/`Wav2Vec`, or `-tts`/`-audio` in the repo id |
| `Multilingual` | `vocab_size` ≥ 100_000 |
| `Reasoning` | `reasoning`/`r1`/`thinking` in the repo id |

`hello.categories` is the canonical display order, deduped, filtered to what was
actually discovered: `["Chat","Code","Vision","Embedding","Audio"]`.

## Local stores scanned by `catalog`

1. `$HF_HOME` (default `~/.cache/huggingface/hub`) — `models--<org>--<name>` dirs.
2. `$SLAM_LM_MODELS` — a colon-separated list of extra directories containing
   `org/name` folders.

A directory qualifies only if it has `config.json` and at least one weight file
(`*.safetensors`, `*.npz`, `*.gguf`). Directories without weights are skipped, not
listed as broken.

## HTTP API (only while `serve` is active)

Bound to `127.0.0.1` on the requested port. This is the surface that makes the
app's "Running locally" status meaningful: any OpenAI-compatible client can use
the loaded model, and every request feeds the same metrics as the app's own
prompt box.

| method | path | notes |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","model":String?}` |
| `GET` | `/v1/models` | OpenAI list shape, one entry (the loaded model) |
| `POST` | `/v1/chat/completions` | OpenAI schema; `stream:true` → SSE `chat.completion.chunk` + `data: [DONE]` |
| `POST` | `/v1/completions` | OpenAI schema; `stream:true` → SSE `text_completion` |
| `GET` | `/metrics` | `{"live":LiveMetrics,"requests":[RequestRecord,…]}` |

`503` when no model is loaded. `404` for unknown paths. Chat prompts are rendered
through the model's chat template when it has one, else the messages are
concatenated with newlines.

## Swift side

`Protocol.swift` mirrors the types above as `Codable` structs and owns the single
observable state object both surfaces read:

```swift
final class MetricsBus: ObservableObject {
    @Published var status: RunnerStatus, phase: RunnerPhase
    @Published var models: [ModelInfo], selected: ModelInfo?, categories: [String]
    @Published var hardware: HardwareInfo?, live: LiveMetrics?
    @Published var history: [LiveMetrics]        // ring buffer, newest last, ≤ 600
    @Published var requests: [RequestRecord]     // newest first, ≤ 100
    @Published var servingURL: String?, logLine: String?, streamText: String
    @Published var busy: Bool, errorText: String?, bridgeReady: Bool

    func ingest(_ sample: LiveMetrics)           // called by BridgeClient
    func ingest(_ record: RequestRecord)
    func apply(_ state: StatePayload)
    func clearStream()
}
```

State lives in the bus, transport does not: `BridgeClient` decodes process output
and pushes it in, `ModelStore` owns the bridge process plus catalog search/filter
and the commands below, and every view observes the bus. The analytics surface
depends on `Protocol.swift` and `Design.swift` only, so it never touches the
transport.

```swift
// ModelStore — the picker surface's command surface
func refreshCatalog()
func toggle(_ model: ModelInfo)          // load if stopped, else unload
func send(prompt: String, maxTokens: Int)
func cancel()
func toggleServe()

// AnalyticsView — the analytics surface's sole entry point
struct AnalyticsView: View { init(metrics: MetricsBus, actions: AnalyticsActions) }
// AnalyticsActions is declared in Protocol.swift: send(prompt, maxTokens),
// cancel(), and showModels() so the expanded tab is never a dead end.
```

## Preview and snapshot flags

The app accepts, for development and verification:

| flag / env | effect |
|---|---|
| `--preview` | render the panel in a normal window instead of the menu bar popover |
| `--snapshot <path.png>` | render the panel offscreen to a PNG at 2x and exit 0 |
| `--snapshot-analytics <path.png>` | same, with the analytics tab open |
| `--model <id>` | model the snapshot modes load before rendering (default: the first catalog entry) |
| `--tokens <n>` | generated tokens for the snapshot's real request (default 64) |
| `--requests <n>` | real runs `--snapshot-analytics` drives before rendering (default 1; 3 makes the per-request throughput chart meaningful) |
| `--prompt <text>` | overrides the built-in snapshot prompt, to capture a particular shape of output |
| `--load` | with a plain `--snapshot`, load the model first so the captured panel shows the running state |
| `--tab <models\|analytics>` | initial tab |
| `--port <n>` | HTTP port for `serve` (default 8712) |
| `SLAM_LM_PYTHON` | interpreter for the bridge (default: repo `.venv/bin/python`) |
| `SLAM_LM_BRIDGE_DIR` | directory containing `slam_lm_bridge/` (default: bundled `Contents/Resources/sidecar`) |

`--snapshot <path>` renders the picker with the real catalog and live memory, and
exits. `--snapshot-analytics <path>` must additionally *drive* the run itself,
because nothing else will: load the `--model` (or first catalog) model, start
`serve`, wait for the load to report ready, send one real prompt of `--tokens`
tokens over the bridge, wait for `request_end`, and only then render. Both modes
must fail loudly (nonzero exit, message on stderr) rather than render a panel with
fabricated numbers if real data never arrives.

## File ownership

| path | owner |
|---|---|
| `menubar/PROTOCOL.md`, `Protocol.swift`, `Design.swift` | frozen contract, do not edit without changing this file |
| `menubar/Sources/SlamLM/{SlamLMApp,BridgeClient,ModelStore,HeaderView,ModelRow,FooterView,ModelCatalog,HardwareProbe}.swift` | picker surface |
| `menubar/Sources/SlamLM/{AnalyticsView,MetricSeries}.swift` | analytics surface |
| `menubar/sidecar/slam_lm_bridge/**`, `menubar/sidecar/tests/**` | bridge |
| `menubar/build.sh` | bundling, signing, snapshot capture |
