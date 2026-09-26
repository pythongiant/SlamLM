import Combine
import Foundation

// MARK: - Wire types
//
// Mirrors `menubar/PROTOCOL.md` exactly; both sides must agree on these keys.
// Wire structs stay dumb (no logic, no optionals unless the protocol says a
// field is nullable) so a decode failure is always a real contract mismatch.

/// One locally available model. `id` is the repo id and the unique key.
struct ModelInfo: Codable, Identifiable, Hashable {
    let id: String
    let name: String
    let params: String
    let quant: String
    let bytes: Int
    let path: String
    let categories: [String]
    let architecture: String
    let contextLength: Int
    let hasChatTemplate: Bool
    let lastUsed: Double
}

/// Machine facts plus a live memory sample. Sourced in the bridge from
/// `sysctl`, `system_profiler`, Mach `host_statistics64` and `mx.device_info()`;
/// never estimated.
struct HardwareInfo: Codable, Hashable {
    let model: String
    let chip: String
    let gpuCores: Int
    let totalBytes: Int
    let recommendedBytes: Int
    let memoryActiveBytes: Int
    let memoryPeakBytes: Int
    let cacheBytes: Int
    let processRssBytes: Int
    let systemUsedBytes: Int
    let systemAppBytes: Int
    let systemWiredBytes: Int
    let systemCompressedBytes: Int
    let systemCachedBytes: Int
    let systemSwapBytes: Int
}

/// Runner status as it appears on the wire.
enum RunnerStatus: String, Codable, CaseIterable {
    case idle, loading, ready, generating, error
}

/// What the runner is spending time on right now.
enum RunnerPhase: String, Codable, CaseIterable {
    case idle, load, prefill, decode
}

/// One 5 Hz telemetry sample. Every field is a real measurement; a field with
/// no data yet is `0` and the UI must render an empty state, not a placeholder.
struct LiveMetrics: Codable, Identifiable, Hashable {
    let ts: Double
    let memoryActiveBytes: Int
    let memoryPeakBytes: Int
    let memoryCacheBytes: Int
    let memoryTotalBytes: Int
    let memoryRecommendedBytes: Int
    let processRssBytes: Int
    /// Machine-wide, Activity Monitor's definitions. Not this process's share.
    let systemUsedBytes: Int
    let systemAppBytes: Int
    let systemWiredBytes: Int
    let systemCompressedBytes: Int
    let systemCachedBytes: Int
    let systemSwapBytes: Int
    let decodeTps: Double
    let prefillTps: Double
    let ttftMs: Double
    let tokensGenerated: Int
    let requests: Int
    let status: String
    let phase: String
    let model: String?
    let loadMs: Double

    var id: Double { ts }
    var runnerStatus: RunnerStatus { RunnerStatus(rawValue: status) ?? .idle }
    var runnerPhase: RunnerPhase { RunnerPhase(rawValue: phase) ?? .idle }
}

/// One completed (or cancelled, or failed) request.
struct RequestRecord: Codable, Identifiable, Hashable {
    let request: Int
    let model: String
    let promptTokens: Int
    let genTokens: Int
    let ttftMs: Double
    let prefillTps: Double
    let decodeTps: Double
    let peakMemBytes: Int
    let startedAt: Double
    let totalMs: Double
    let toolCalls: Int
    let finishReason: String

    var id: Int { request }
    var failed: Bool { finishReason == "error" }
}

// MARK: - Result payloads

struct HelloResult: Codable {
    let protocolVersion: Int
    let bridge: String
    let hardware: HardwareInfo
    let categories: [String]

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol"
        case bridge, hardware, categories
    }
}

struct CatalogResult: Codable { let models: [ModelInfo] }
struct LoadResult: Codable { let model: String; let loadMs: Double; let memoryBytes: Int }
struct CancelResult: Codable { let cancelled: Bool }
struct ServeResult: Codable { let port: Int; let url: String }
struct StopServeResult: Codable { let stopped: Bool }
struct EmptyPayload: Codable {}

// MARK: - Event payloads

struct StatePayload: Codable {
    let status: String
    let phase: String
    let model: String?
    let message: String?
}

struct TokenPayload: Codable {
    let request: Int
    let index: Int
    let text: String
    let ttsMs: Double
}

struct LogPayload: Codable {
    let level: String
    let message: String
}

/// One tool call or its result. A call carries the arguments the model asked
/// with; a result carries what came back. `ok` is false for a refusal or a
/// failure, which the model is shown rather than swallowed.
struct ToolEvent: Codable, Identifiable, Hashable {
    let request: Int
    let round: Int
    let phase: String
    let name: String
    let arguments: ToolArguments?
    let ok: Bool
    let summary: String
    let detail: String?

    var id: String { "\(request)-\(round)-\(phase)-\(name)" }
    var isCall: Bool { phase == "call" }
    var failed: Bool { !ok }
}

/// Tool arguments arrive as an arbitrary JSON object. The panel only ever shows
/// them, so each value is flattened to display text here rather than pulling a
/// JSON value type through the whole app.
struct ToolArguments: Codable, Hashable {
    let pairs: [(String, String)]

    /// Only decoding constructs these in the app; this exists so tests and
    /// offscreen harnesses can build one.
    init(pairs: [(String, String)]) { self.pairs = pairs }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DynamicKey.self)
        pairs = try container.allKeys
            .map { ($0.stringValue, try container.decode(AnyJSONText.self, forKey: $0).text) }
            .sorted { $0.0 < $1.0 }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: DynamicKey.self)
        for (key, value) in pairs {
            try container.encode(value, forKey: DynamicKey(stringValue: key)!)
        }
    }

    /// `key: value, key: value`, for a one-line caption.
    var summary: String { pairs.map { "\($0.0): \($0.1)" }.joined(separator: ", ") }

    static func == (lhs: ToolArguments, rhs: ToolArguments) -> Bool {
        lhs.pairs.elementsEqual(rhs.pairs, by: { $0.0 == $1.0 && $0.1 == $1.1 })
    }

    func hash(into hasher: inout Hasher) {
        for pair in pairs { hasher.combine(pair.0); hasher.combine(pair.1) }
    }
}

/// Display text for any JSON value: strings as themselves, everything else in
/// compact JSON form. Recursive, so nested objects and arrays survive.
struct AnyJSONText: Decodable {
    let text: String

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) { text = value; return }
        if let value = try? container.decode(Bool.self) { text = value ? "true" : "false"; return }
        if let value = try? container.decode(Int.self) { text = String(value); return }
        if let value = try? container.decode(Double.self) { text = String(value); return }
        if container.decodeNil() { text = "null"; return }
        if let values = try? container.decode([AnyJSONText].self) {
            text = "[" + values.map(\.text).joined(separator: ", ") + "]"
            return
        }
        let keyed = try decoder.container(keyedBy: DynamicKey.self)
        let parts = try keyed.allKeys.map { key -> String in
            let value = try keyed.decode(AnyJSONText.self, forKey: key)
            return "\(key.stringValue): \(value.text)"
        }
        text = "{" + parts.joined(separator: ", ") + "}"
    }
}

struct DynamicKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
}

/// Reply envelope: `{"id":Int,"ok":Bool,"result":{…}}` or `{"id":Int,"ok":Bool,"error":String}`.
struct WireReply<Payload: Decodable>: Decodable {
    let id: Int
    let ok: Bool
    let error: String?
    let result: Payload?
}

typealias AckReply = WireReply<EmptyPayload>

struct WireEvent<P: Decodable>: Decodable {
    let event: String
    let data: P
}

/// Probe used to route a stdout line before decoding its payload.
struct WireProbe: Decodable {
    let id: Int?
    let ok: Bool?
    let error: String?
    let event: String?
}

// MARK: - The one state object both surfaces read
//
// State lives here, transport does not: `BridgeClient` (picker surface) decodes
// process output and pushes it in, and every view observes this single object.
// The analytics surface therefore depends only on this file, never on the
// transport.

final class MetricsBus: ObservableObject {
    /// Ring buffers. Sized for the panel's own history charts.
    static let historyLimit = 600
    static let requestLimit = 100
    static let toolEventLimit = 200

    @Published var status: RunnerStatus = .idle
    @Published var phase: RunnerPhase = .idle
    @Published var models: [ModelInfo] = []
    @Published var selected: ModelInfo?
    @Published var hardware: HardwareInfo?
    @Published var categories: [String] = []
    @Published var live: LiveMetrics?
    @Published var history: [LiveMetrics] = []
    @Published var requests: [RequestRecord] = []
    @Published var servingURL: String?
    @Published var logLine: String?
    @Published var busy = false
    @Published var errorText: String?
    @Published var bridgeReady = false
    /// Text streamed for the most recent app-initiated request, newest last.
    @Published var streamText = ""
    /// Tool calls and results for the most recent request, in order.
    @Published var toolEvents: [ToolEvent] = []
    /// Whether the model may call the read-only tools. Read by the store when it
    /// sends a prompt, so both surfaces agree on the setting.
    @Published var toolsEnabled = true
    /// Session token counter when the newest request started, so the panel can
    /// report that request's progress while it runs.
    @Published var requestStartTokens = 0

    /// Append a telemetry sample, keeping the buffer bounded.
    func ingest(_ sample: LiveMetrics) {
        live = sample
        status = sample.runnerStatus
        phase = sample.runnerPhase
        history.append(sample)
        if history.count > Self.historyLimit {
            history.removeFirst(history.count - Self.historyLimit)
        }
    }

    /// Record a finished request, newest first.
    func ingest(_ record: RequestRecord) {
        requests.insert(record, at: 0)
        if requests.count > Self.requestLimit {
            requests.removeLast(requests.count - Self.requestLimit)
        }
    }

    func apply(_ state: StatePayload) {
        status = RunnerStatus(rawValue: state.status) ?? status
        phase = RunnerPhase(rawValue: state.phase) ?? phase
    }

    /// Record a tool call or result, keeping the trace bounded.
    func ingest(_ event: ToolEvent) {
        toolEvents.append(event)
        if toolEvents.count > Self.toolEventLimit {
            toolEvents.removeFirst(toolEvents.count - Self.toolEventLimit)
        }
    }

    func clearStream() {
        streamText = ""
        toolEvents = []
        requestStartTokens = live?.tokensGenerated ?? 0
    }
}

/// Panel tabs. `models` is the picker, `analytics` the expanded metrics surface.
enum PanelTab: String, CaseIterable, Identifiable {
    case models, analytics
    var id: String { rawValue }
    var title: String { self == .models ? "Models" : "Analytics" }
    var symbol: String { self == .models ? "square.stack.3d.up" : "chart.xyaxis.line" }
}

/// Commands the analytics surface may invoke, injected by the picker surface so
/// the panel never depends on the transport or the store.
struct AnalyticsActions {
    var send: (String) -> Void
    var cancel: () -> Void
    /// Return to the model picker. The panel owns its tab state, so it supplies
    /// this; the analytics surface must never be a dead end.
    var showModels: () -> Void
}

// MARK: - Runtime configuration

/// Where the bridge lives and what it is launched with. Resolved once at
/// startup from the command line and environment; see `PROTOCOL.md`.
struct RunnerConfig {
    var pythonPath: String
    var bridgeDir: String
    /// Repo checkout holding `mlx_lm` and `.venv`. The app passes this to the
    /// bridge so it never has to guess where the package lives.
    var repoRoot: String
    var port: Int
    var tab: PanelTab
    var preview: Bool
    var snapshotPath: String?
    var snapshotAnalytics: String?
    var model: String?
    /// A token budget for a snapshot run; absent means no budget.
    var tokens: Int?
    /// Overrides the built-in snapshot prompt, so the capture can exercise a
    /// particular shape of output (markdown, code, a long list).
    var prompt: String?
    /// How many real runs `--snapshot-analytics` drives before rendering.
    var requests: Int
    /// Loads the model before a plain `--snapshot` too, so the captured panel
    /// shows the running state rather than an idle one.
    var loadForSnapshot: Bool

    static func resolve() -> RunnerConfig {
        var args = Array(CommandLine.arguments.dropFirst())
        func takeFlag(_ name: String) -> String? {
            guard let i = args.firstIndex(of: name) else { return nil }
            guard i + 1 < args.count else { return "" }
            let value = args[i + 1]
            args.removeSubrange(i...(i + 1))
            return value
        }
        let env = ProcessInfo.processInfo.environment
        let snapshot = takeFlag("--snapshot")
        let snapshotAnalytics = takeFlag("--snapshot-analytics")
        let tab = takeFlag("--tab").flatMap(PanelTab.init(rawValue:)) ?? .models
        let port = takeFlag("--port").flatMap(Int.init) ?? 8712
        let model = RunnerConfig.nonEmpty(takeFlag("--model"))
        let tokens = takeFlag("--tokens").flatMap(Int.init)
        let requests = max(1, takeFlag("--requests").flatMap(Int.init) ?? 1)
        let prompt = RunnerConfig.nonEmpty(takeFlag("--prompt"))
        let loadForSnapshot = args.contains("--load")
        let preview = args.contains("--preview") || snapshot != nil || snapshotAnalytics != nil

        let bridgeDir: String
        if let explicit = env["SLAM_LM_BRIDGE_DIR"], !explicit.isEmpty {
            bridgeDir = explicit
        } else if let res = Bundle.main.resourcePath,
                  FileManager.default.fileExists(atPath: res + "/sidecar/slam_lm_bridge") {
            bridgeDir = res + "/sidecar"
        } else {
            // Running straight out of `.build/debug` during development.
            let here = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
            bridgeDir = here.deletingLastPathComponent().path
        }

        // The bundle records the interpreter and checkout it was built against
        // (`build.sh` writes it), so a Finder launch never has to guess them.
        let runtime = RunnerConfig.bundledRuntime()

        let repoRoot: String
        if let explicit = env["SLAM_LM_REPO"], !explicit.isEmpty {
            repoRoot = explicit
        } else if let recorded = runtime.repo, RunnerConfig.looksLikeRepo(recorded) {
            repoRoot = recorded
        } else {
            repoRoot = RunnerConfig.findRepoRoot(bridgeDir: bridgeDir)
        }

        let pythonPath: String
        if let explicit = env["SLAM_LM_PYTHON"], !explicit.isEmpty {
            pythonPath = explicit
        } else if let recorded = runtime.python,
                  FileManager.default.isExecutableFile(atPath: recorded) {
            pythonPath = recorded
        } else {
            pythonPath = repoRoot + "/.venv/bin/python"
        }

        return RunnerConfig(
            pythonPath: pythonPath,
            bridgeDir: bridgeDir,
            repoRoot: repoRoot,
            port: port,
            tab: tab,
            preview: preview,
            snapshotPath: RunnerConfig.nonEmpty(snapshot),
            snapshotAnalytics: RunnerConfig.nonEmpty(snapshotAnalytics),
            model: model,
            tokens: tokens,
            prompt: prompt,
            requests: requests,
            loadForSnapshot: loadForSnapshot
        )
    }

    private struct BundledRuntime: Decodable {
        let python: String?
        let repo: String?
    }

    private static func bundledRuntime() -> BundledRuntime {
        let empty = BundledRuntime(python: nil, repo: nil)
        guard let resources = Bundle.main.resourcePath,
              let data = FileManager.default.contents(atPath: resources + "/runtime.json"),
              let parsed = try? JSONDecoder().decode(BundledRuntime.self, from: data)
        else { return empty }
        return parsed
    }

    /// A directory is the checkout when it holds `mlx_lm`, and in a development
    /// tree the virtualenv beside it.
    private static func looksLikeRepo(_ path: String) -> Bool {
        let fm = FileManager.default
        return fm.fileExists(atPath: path + "/mlx_lm/__init__.py")
            || fm.fileExists(atPath: path + "/.venv/bin/python")
    }

    /// Empty and absent flags mean the same thing to the caller.
    private static func nonEmpty(_ value: String?) -> String? {
        guard let value, !value.isEmpty else { return nil }
        return value
    }

    /// Walk up from the bridge directory looking for the checkout: the bridge
    /// sits at `menubar/sidecar` in a development tree and at
    /// `Contents/Resources/sidecar` inside the bundle.
    private static func findRepoRoot(bridgeDir: String) -> String {
        var dir = URL(fileURLWithPath: bridgeDir).standardizedFileURL
        for _ in 0..<9 {
            if looksLikeRepo(dir.path) { return dir.path }
            let parent = dir.deletingLastPathComponent()
            if parent.path == dir.path { break }
            dir = parent
        }
        return FileManager.default.currentDirectoryPath
    }
}
