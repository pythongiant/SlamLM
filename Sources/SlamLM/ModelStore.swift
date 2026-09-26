import Combine
import Foundation

/// `generate` acknowledges a request with its id; the tokens follow as events.
struct GenerateAck: Codable { let request: Int }

/// Owner of the bridge process, the bus, the catalog filter state and the
/// command surface from PROTOCOL.md:
///
///     refreshCatalog()  toggle(_:)  send(prompt:maxTokens:)  cancel()  toggleServe()
///
/// Every method must be called on the main queue (the views do), and every
/// completion arrives on the main queue from `BridgeClient`.
final class ModelStore: ObservableObject {
    let config: RunnerConfig
    let bus: MetricsBus
    let client: BridgeClient

    /// Catalog filter state; the header's search field and the footer's chips
    /// write here.
    @Published var query = ""
    @Published var category = ModelCatalog.allCategory

    /// Id of the most recent app-initiated `generate`, for callers that want to
    /// wait for that request's `request_end` (the snapshot runner does).
    private(set) var lastRequestID: Int?

    private var nextRequest = 0

    init(config: RunnerConfig) {
        self.config = config
        self.bus = MetricsBus()
        self.client = BridgeClient(config: config, bus: bus)
        self.client.onReady = { [weak self] in self?.refreshCatalog() }
        self.client.start()
    }

    /// Terminate the sidecar; called when the app quits.
    func shutdown() { client.shutdown() }

    // MARK: - Derived catalog state

    /// The model id the bridge reports as loaded, which is the only honest
    /// answer: `bus.selected` follows the picker, this follows the runner.
    var loadedModelID: String? { bus.live?.model }

    var loadedModel: ModelInfo? {
        guard let id = loadedModelID else { return nil }
        return bus.models.first { $0.id == id }
    }

    var visibleModels: [ModelInfo] {
        ModelCatalog.filter(bus.models, query: query, category: category)
    }

    var categoryChips: [String] {
        ModelCatalog.chips(discovered: bus.categories, models: bus.models)
    }

    /// Highlight the row in the panel without touching the runner.
    func select(_ model: ModelInfo) { bus.selected = model }

    // MARK: - Commands

    /// Rescan the local model stores.
    func refreshCatalog() {
        client.request("catalog", expecting: CatalogResult.self) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let catalog):
                self.bus.models = catalog.models
                let stillThere = self.bus.selected.map { selected in
                    catalog.models.contains { $0.id == selected.id }
                } ?? false
                if !stillThere { self.bus.selected = catalog.models.first }
            case .failure(let failure):
                self.bus.errorText = failure.message
            }
        }
    }

    /// Load the model and start the local endpoint, or — when it is already the
    /// loaded one — stop serving and unload it.
    func toggle(_ model: ModelInfo) {
        guard !bus.busy else { return }
        if loadedModelID == model.id {
            unload()
        } else {
            load(model)
        }
    }

    /// Send one prompt through the loaded model. Output arrives as `token`
    /// events into `bus.streamText`, then as a `request_end` record.
    /// Request fields for one generation. `max_tokens` is only included when one
    /// was asked for, so the bridge's own ceiling applies otherwise.
    private func generateFields(prompt: String, request: Int, maxTokens: Int?) -> [String: Any] {
        var fields: [String: Any] = [
            "prompt": prompt,
            "request": request,
            "chat": true,
            "tools": bus.toolsEnabled
        ]
        if let maxTokens { fields["max_tokens"] = maxTokens }
        return fields
    }

    /// `maxTokens` is only used by the snapshot tooling to keep a capture short;
    /// the panel sends none, so a request runs until the model stops.
    func send(prompt: String, maxTokens: Int? = nil) {
        guard !bus.busy else { return }
        guard loadedModelID != nil else {
            bus.errorText = "Load a model before sending a prompt."
            return
        }
        nextRequest += 1
        let request = nextRequest
        lastRequestID = request
        bus.clearStream()
        bus.errorText = nil
        client.request(
            // The panel's box is a chat turn, not a document to continue, so the
            // bridge renders it through the model's own chat template — the same
            // path the HTTP endpoint uses.
            "generate",
            fields: generateFields(prompt: prompt, request: request, maxTokens: maxTokens),
            expecting: GenerateAck.self
        ) { [weak self] result in
            guard let self else { return }
            if case .failure(let failure) = result {
                self.bus.errorText = "Generate failed: \(failure.message)"
            }
        }
    }

    /// Cancel the in-flight request, if any.
    func cancel() {
        client.request("cancel", expecting: CancelResult.self) { [weak self] result in
            guard let self else { return }
            if case .failure(let failure) = result {
                self.bus.errorText = "Cancel failed: \(failure.message)"
            }
        }
    }

    /// Start or stop the OpenAI-compatible endpoint on `config.port`.
    func toggleServe() {
        guard !bus.busy else { return }
        bus.busy = true
        bus.errorText = nil
        if bus.servingURL != nil {
            client.request("stop_serve", expecting: StopServeResult.self) { [weak self] result in
                guard let self else { return }
                switch result {
                case .success:
                    self.bus.servingURL = nil
                    self.finish(error: nil)
                case .failure(let failure):
                    self.finish(error: failure)
                }
            }
        } else {
            startServing()
        }
    }

    // MARK: - Load / unload / serve

    private func load(_ model: ModelInfo) {
        bus.busy = true
        bus.errorText = nil
        client.request("load", fields: ["model": model.id], expecting: LoadResult.self) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.bus.selected = model
                // "Running locally" is only honest once the endpoint is up.
                self.startServing()
            case .failure(let failure):
                self.finish(error: failure)
            }
        }
    }

    private func startServing() {
        client.request("serve", fields: ["port": config.port], expecting: ServeResult.self) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let served):
                self.bus.servingURL = served.url
                self.finish(error: nil)
            case .failure(let failure):
                self.finish(error: failure)
            }
        }
    }

    private func unload() {
        bus.busy = true
        bus.errorText = nil
        let finishUnload: () -> Void = { [weak self] in
            guard let self else { return }
            // `expecting:` takes the payload type; `AckReply` would be the
            // envelope and the reply's `{}` would be decoded twice.
            self.client.request("unload", expecting: EmptyPayload.self) { result in
                switch result {
                case .success: self.finish(error: nil)
                case .failure(let failure): self.finish(error: failure)
                }
            }
        }
        guard bus.servingURL != nil else {
            finishUnload()
            return
        }
        // Stop the endpoint first so no client can hit a model mid-unload.
        client.request("stop_serve", expecting: StopServeResult.self) { [weak self] result in
            guard let self else { return }
            if case .success = result { self.bus.servingURL = nil }
            finishUnload()
        }
    }

    /// End of a model-affecting command: the busy flag clears, and a successful
    /// run also clears a stale error.
    private func finish(error: BridgeClient.Failure?) {
        bus.busy = false
        if let error {
            bus.errorText = error.message
        } else {
            bus.errorText = nil
        }
    }
}
