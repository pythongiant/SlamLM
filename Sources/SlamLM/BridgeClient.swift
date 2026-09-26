import Combine
import Foundation

/// Transport for the bridge sidecar.
///
/// Spawns `python -m slam_lm_bridge.server` with the resolved interpreter and
/// bridge directory, writes newline-delimited JSON commands to its stdin and
/// routes stdout lines into `MetricsBus`. State lives in the bus, the process
/// lives here; `ModelStore` owns both.
///
/// Every call into the bus and every command completion happens on the main
/// queue, so neither `ModelStore` nor the views need locking.
final class BridgeClient {
    enum Failure: LocalizedError {
        case launch(String)
        case offline
        case bridge(String)
        case malformed(String)

        var message: String {
            switch self {
            case .launch(let detail): return "Launch failed: \(detail)"
            case .offline: return "The bridge process is not running."
            case .bridge(let detail): return detail
            case .malformed(let detail): return "Unreadable bridge reply (\(detail))."
            }
        }

        var errorDescription: String? { message }
    }

    /// How many stderr lines are kept for diagnostics. The newest one is also
    /// shown in the panel's log strip through `bus.logLine`.
    private static let stderrTailLimit = 40

    private let config: RunnerConfig
    private let bus: MetricsBus

    /// Called once, when the `hello` handshake resolves.
    var onReady: (() -> Void)?

    private var process: Process?
    private var stdin: FileHandle?
    private var stdoutHandle: FileHandle?
    private var stderrHandle: FileHandle?
    private var stdoutBuffer = Data()
    private var stderrBuffer = Data()
    private var stderrTail: [String] = []
    private var pending: [Int: (Result<Data, Failure>) -> Void] = [:]
    private var nextID = 1
    private var handshakeDone = false
    private var stopping = false
    /// Request id of the most recent app-initiated `generate`. The bridge also
    /// streams `token` events for HTTP clients hitting the local endpoint (their
    /// ids come from the bridge's HTTP range), and those are not this app's
    /// output: only this id may reach `bus.streamText`.
    private var appStreamRequest: Int?

    init(config: RunnerConfig, bus: MetricsBus) {
        self.config = config
        self.bus = bus
    }

    var isRunning: Bool { process?.isRunning == true }

    // MARK: - Lifecycle

    /// Launch the sidecar and run the `hello` handshake.
    func start() {
        guard process == nil else { return }

        let child = Process()
        child.executableURL = URL(fileURLWithPath: config.pythonPath)
        child.arguments = ["-m", "slam_lm_bridge.server"]
        var environment = ProcessInfo.processInfo.environment
        // The bridge needs the checkout (for `mlx_lm`) and its own package dir;
        // a Finder launch has no useful PYTHONPATH of its own, so both are set
        // explicitly rather than inherited.
        environment["PYTHONPATH"] = [config.bridgeDir, config.repoRoot].joined(separator: ":")
        environment["SLAM_LM_REPO"] = config.repoRoot
        // The protocol is line-delimited, so make sure the child never parks a
        // full line in a block buffer.
        environment["PYTHONUNBUFFERED"] = "1"
        child.environment = environment
        child.currentDirectoryURL = URL(fileURLWithPath: config.repoRoot)

        let input = Pipe()
        let output = Pipe()
        let diagnostics = Pipe()
        child.standardInput = input
        child.standardOutput = output
        child.standardError = diagnostics

        child.terminationHandler = { [weak self] finished in
            DispatchQueue.main.async { self?.didExit(status: finished.terminationStatus) }
        }

        do {
            try child.run()
        } catch {
            fail(message: "Could not launch \(config.pythonPath): \(error.localizedDescription)")
            return
        }

        process = child
        stdin = input.fileHandleForWriting
        stdoutHandle = output.fileHandleForReading
        stderrHandle = diagnostics.fileHandleForReading
        observe(stdoutHandle, channel: .stdout)
        observe(stderrHandle, channel: .stderr)

        request("hello", expecting: HelloResult.self) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let hello):
                self.apply(hello: hello)
            case .failure(let failure):
                self.fail(message: "Bridge handshake failed: \(failure.message)")
            }
        }
    }

    /// Stop the child. Called when the app quits; after this the client is dead.
    func shutdown() {
        stopping = true
        stdoutHandle?.readabilityHandler = nil
        stderrHandle?.readabilityHandler = nil
        stdoutHandle = nil
        stderrHandle = nil
        if let stdin {
            try? stdin.close()
        }
        stdin = nil
        if let process, process.isRunning {
            process.terminate()
        }
        process = nil
    }

    // MARK: - Commands

    /// Write one request and hand its single reply to `completion` on the main
    /// queue. Returns the request id, or `nil` when the bridge is not running.
    @discardableResult
    func request<T: Decodable>(
        _ cmd: String,
        fields: [String: Any] = [:],
        expecting: T.Type,
        completion: @escaping (Result<T, Failure>) -> Void
    ) -> Int? {
        guard let stdin, isRunning else {
            DispatchQueue.main.async { completion(.failure(.offline)) }
            return nil
        }

        let id = nextID
        nextID += 1

        // Every `generate` written here is app-initiated, so its request id is
        // the one whose tokens belong in the panel's stream.
        if cmd == "generate", let request = fields["request"] as? Int {
            appStreamRequest = request
        }

        // The reply is resolved exactly once: `resolve` removes the entry, and
        // `failAll` drains it, so a completion can never fire twice.
        pending[id] = { outcome in
            switch outcome {
            case .failure(let failure):
                completion(.failure(failure))
            case .success(let line):
                guard let reply = try? JSONDecoder().decode(WireReply<T>.self, from: line) else {
                    let raw = String(decoding: line.prefix(120), as: UTF8.self)
                    completion(.failure(.malformed("\(cmd) reply: \(raw)")))
                    return
                }
                if reply.ok, let payload = reply.result {
                    completion(.success(payload))
                } else {
                    completion(.failure(.bridge(reply.error ?? "\(cmd) failed without a message")))
                }
            }
        }

        do {
            try stdin.write(contentsOf: encode(id: id, cmd: cmd, fields: fields))
        } catch {
            pending.removeValue(forKey: id)
            DispatchQueue.main.async { completion(.failure(.launch(error.localizedDescription))) }
            return nil
        }
        return id
    }

    private func encode(id: Int, cmd: String, fields: [String: Any]) -> Data {
        var object: [String: Any] = ["id": id, "cmd": cmd]
        for (key, value) in fields { object[key] = value }
        // Sorted keys keep the wire bytes deterministic.
        var line = (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data()
        line.append(0x0A)
        return line
    }

    // MARK: - Reading

    private enum Channel { case stdout, stderr }

    private func observe(_ handle: FileHandle?, channel: Channel) {
        handle?.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else {
                // EOF: stop the handler so it does not spin.
                handle.readabilityHandler = nil
                return
            }
            self?.receive(chunk, from: channel)
        }
    }

    /// Called on the pipe's own queue. Each channel owns its own buffer, so the
    /// two handlers never share mutable state.
    private func receive(_ chunk: Data, from channel: Channel) {
        switch channel {
        case .stdout:
            for line in Self.lines(from: &stdoutBuffer, chunk: chunk) {
                DispatchQueue.main.async { self.handle(line: line) }
            }
        case .stderr:
            for line in Self.lines(from: &stderrBuffer, chunk: chunk) {
                let text = String(decoding: line, as: UTF8.self)
                DispatchQueue.main.async { self.note(stderr: text) }
            }
        }
    }

    /// Pull complete newline-terminated lines out of `buffer` plus `chunk`.
    /// A partial trailing chunk stays buffered until its newline arrives.
    private static func lines(from buffer: inout Data, chunk: Data) -> [Data] {
        buffer.append(chunk)
        var lines: [Data] = []
        while let newline = buffer.firstIndex(of: 0x0A) {
            let line = buffer.subdata(in: buffer.startIndex..<newline)
            buffer.removeSubrange(buffer.startIndex...newline)
            let trimmed = Data(line.drop(while: { $0 == 0x20 || $0 == 0x09 || $0 == 0x0D }))
            if !trimmed.isEmpty { lines.append(trimmed) }
        }
        return lines
    }

    // MARK: - Routing

    private func handle(line: Data) {
        guard let probe = try? JSONDecoder().decode(WireProbe.self, from: line) else {
            bus.logLine = "Unreadable bridge output: \(String(decoding: line.prefix(160), as: UTF8.self))"
            return
        }
        if let event = probe.event {
            route(event: event, line: line)
        } else if let id = probe.id {
            resolve(id: id, line: line)
        } else {
            bus.logLine = "Bridge sent a line with no id and no event."
        }
    }

    private func route(event: String, line: Data) {
        switch event {
        case "hello":
            // `hello` is also a command; apply it whichever envelope it lands in.
            guard let payload = decode(WireEvent<HelloResult>.self, from: line) else { return }
            apply(hello: payload.data)
        case "metrics":
            guard let payload = decode(WireEvent<LiveMetrics>.self, from: line) else { return }
            bus.ingest(payload.data)
        case "state":
            guard let payload = decode(WireEvent<StatePayload>.self, from: line) else { return }
            bus.apply(payload.data)
            if let id = payload.data.model, let model = bus.models.first(where: { $0.id == id }) {
                bus.selected = model
            }
        case "token":
            guard let payload = decode(WireEvent<TokenPayload>.self, from: line) else { return }
            // Another client streaming from the local endpoint shares this bridge;
            // its tokens must not show up as the app's own output.
            guard payload.data.request == appStreamRequest else { return }
            bus.streamText += payload.data.text
        case "tool":
            guard let payload = decode(WireEvent<ToolEvent>.self, from: line) else { return }
            bus.ingest(payload.data)
        case "request_end":
            guard let payload = decode(WireEvent<RequestRecord>.self, from: line) else { return }
            bus.ingest(payload.data)
        case "log":
            guard let payload = decode(WireEvent<LogPayload>.self, from: line) else { return }
            bus.logLine = payload.data.level == "info"
                ? payload.data.message
                : "\(payload.data.level): \(payload.data.message)"
        default:
            bus.logLine = "Unknown bridge event “\(event)”."
        }
    }

    private func resolve(id: Int, line: Data) {
        guard let completion = pending.removeValue(forKey: id) else {
            bus.logLine = "Reply for unknown request \(id)."
            return
        }
        completion(.success(line))
    }

    private func apply(hello: HelloResult) {
        guard !handshakeDone else { return }
        handshakeDone = true
        bus.hardware = hello.hardware
        bus.categories = hello.categories
        bus.bridgeReady = true
        onReady?()
    }

    private func decode<T: Decodable>(_ type: T.Type, from line: Data) -> T? {
        guard let payload = try? JSONDecoder().decode(T.self, from: line) else {
            bus.logLine = "Unreadable \(T.self) bridge payload."
            return nil
        }
        return payload
    }

    // MARK: - Failure

    private func note(stderr line: String) {
        stderrTail.append(line)
        if stderrTail.count > Self.stderrTailLimit {
            stderrTail.removeFirst(stderrTail.count - Self.stderrTailLimit)
        }
        bus.logLine = line
    }

    private func didExit(status: Int32) {
        process = nil
        stdin = nil
        bus.bridgeReady = false
        guard !stopping else { return }
        fail(message: "Bridge exited with status \(status).")
    }

    private func fail(message: String) {
        bus.bridgeReady = false
        bus.status = .error
        bus.errorText = diagnostic(message)
        failAll(.bridge(message))
    }

    private func failAll(_ failure: Failure) {
        let waiting = pending.values
        pending.removeAll()
        for completion in waiting { completion(.failure(failure)) }
    }

    /// A failure message with the newest stderr lines appended, when there are any.
    private func diagnostic(_ message: String) -> String {
        guard !stderrTail.isEmpty else { return message }
        return "\(message) — \(stderrTail.suffix(3).joined(separator: " · "))"
    }
}
