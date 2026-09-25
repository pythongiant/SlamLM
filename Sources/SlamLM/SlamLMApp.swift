import AppKit
import Combine
import SwiftUI

@main
struct SlamLMApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        MenuBarExtra("SlamLM", systemImage: "brain") { menuContent }
            .menuBarExtraStyle(.window)
    }

    /// Snapshot runs render the panel offscreen and exit, so they never present
    /// anything here.
    @ViewBuilder private var menuContent: some View {
        if !isSnapshot {
            let store = delegate.store()
            PanelView(bus: store.bus, store: store, initialTab: initialTab)
        }
    }

    /// The delegate resolves the flags once at startup; the scene reads the same
    /// values from it.
    private var config: RunnerConfig { delegate.config }

    private var isSnapshot: Bool {
        config.snapshotPath != nil || config.snapshotAnalytics != nil
    }

    /// `--snapshot-analytics` always opens on the analytics tab; otherwise the
    /// `--tab` flag decides.
    private var initialTab: PanelTab {
        config.snapshotAnalytics != nil ? .analytics : config.tab
    }
}

/// The panel: one owner of the tab state, sized by `PanelLayout` and rounded by
/// `PanelLayout.corner`. The picker body for `.models`, the analytics surface
/// for `.analytics`.
struct PanelView: View {
    @ObservedObject var bus: MetricsBus
    @ObservedObject var store: ModelStore
    /// Reports tab changes so the preview window can resize itself.
    var onTabChange: ((PanelTab) -> Void)?
    /// Swap the AppKit-backed controls (`ScrollView`, `TextField`, `Menu`) for
    /// static equivalents so `ImageRenderer` can draw the panel for a snapshot.
    var renderStatic = false

    @State private var tab: PanelTab

    init(
        bus: MetricsBus,
        store: ModelStore,
        initialTab: PanelTab = .models,
        onTabChange: ((PanelTab) -> Void)? = nil,
        renderStatic: Bool = false
    ) {
        self.bus = bus
        self.store = store
        self.onTabChange = onTabChange
        self.renderStatic = renderStatic
        self._tab = State(initialValue: initialTab)
    }

    var body: some View {
        let size = PanelLayout.size(for: tab)
        // The analytics tab scrolls, so its content is taller than the panel. A
        // static snapshot of it is a full-page capture: pin the width, let the
        // content report its ideal height, otherwise `ImageRenderer` centres
        // and clips it. The picker stays at its real panel size.
        let intrinsicHeight = renderStatic && tab == .analytics
        surface
            .frame(width: size.width, height: intrinsicHeight ? nil : size.height)
            .fixedSize(horizontal: false, vertical: intrinsicHeight)
            .background(Paper.bg)
            .clipShape(RoundedRectangle(cornerRadius: PanelLayout.corner, style: .continuous))
            .onChange(of: tab) { _, newTab in onTabChange?(newTab) }
    }

    @ViewBuilder private var surface: some View {
        switch tab {
        case .models:
            picker
        case .analytics:
            // `renderStaticSnapshot` is declared by the analytics surface
            // (`MetricSeries.swift`); the live panel never sets it.
            AnalyticsView(metrics: bus, actions: analyticsActions)
                .environment(\.renderStaticSnapshot, renderStatic)
        }
    }

    /// The panel owns the tab, so it supplies the way back to the picker.
    private var analyticsActions: AnalyticsActions {
        AnalyticsActions(
            send: { prompt, maxTokens in store.send(prompt: prompt, maxTokens: maxTokens) },
            cancel: { store.cancel() },
            showModels: { tab = .models }
        )
    }

    // MARK: - Picker

    private var picker: some View {
        VStack(spacing: 0) {
            HeaderView(bus: bus, store: store, query: $store.query, renderStatic: renderStatic)
            CategoryFilterRow(
                chips: store.categoryChips,
                selection: $store.category,
                renderStatic: renderStatic
            )
            PaperRule()
            modelList
            if let error = bus.errorText { errorStrip(error) }
            FooterView(bus: bus, store: store, tab: $tab, renderStatic: renderStatic)
        }
    }

    @ViewBuilder private var modelList: some View {
        if renderStatic {
            rows
        } else {
            ScrollView { rows }
                .frame(maxHeight: .infinity)
        }
    }

    @ViewBuilder private var rows: some View {
        VStack(spacing: 0) {
            let models = store.visibleModels
            if models.isEmpty {
                EmptyState(symbol: emptySymbol, title: emptyTitle, detail: emptyDetail)
            } else {
                ForEach(models.indices, id: \.self) { index in
                    let model = models[index]
                    ModelRow(
                        model: model,
                        loaded: store.loadedModelID == model.id,
                        busy: bus.busy,
                        isLast: index == models.count - 1,
                        onToggle: { store.toggle(model) },
                        onSelect: { store.select(model) }
                    )
                }
            }
        }
        .padding(.vertical, 4)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private var isUnfiltered: Bool {
        store.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && store.category == ModelCatalog.allCategory
    }

    private var emptySymbol: String {
        if !bus.bridgeReady { return "bolt.horizontal" }
        return isUnfiltered ? "tray" : "magnifyingglass"
    }

    private var emptyTitle: String {
        if !bus.bridgeReady { return "Starting the bridge…" }
        return isUnfiltered ? "No local models" : "No matches"
    }

    private var emptyDetail: String? {
        guard bus.bridgeReady else { return nil }
        return isUnfiltered
            ? "No model with weights was found in the local stores."
            : "Nothing matches the current search and filter."
    }

    private func errorStrip(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 10))
                .foregroundStyle(Paper.clayDeep)
            Text(message)
                .font(PaperFont.meta)
                .foregroundStyle(Paper.ink)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 4)
            Button {
                bus.errorText = nil
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(Paper.inkSoft)
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous).fill(Paper.washClay)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(Paper.clay.opacity(0.55), lineWidth: 1)
        )
        .padding(.horizontal, PanelLayout.gutter)
        .padding(.bottom, 4)
    }
}

/// Owns the process, the panel window used by `--preview`, and the offscreen
/// snapshot runs. It creates the store lazily so exactly one bridge exists per
/// launch, whether the store is first reached from the menu bar item, the
/// preview window or a snapshot run.
@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    /// Prompts used by `--snapshot-analytics`; short, and their tokens are real.
    /// Each round differs so the per-request throughput chart shows real
    /// variation instead of a repeated identical run.
    private func snapshotPrompt(round: Int) -> String {
        if let override = config.prompt { return override }
        return "List the numbers 1 through \(12 * round), one per line."
    }

    /// The command line, resolved once per launch; the scene reads it from here.
    let config = RunnerConfig.resolve()

    private var activeStore: ModelStore?
    private var previewWindow: NSWindow?
    private var snapshotTimer: Timer?

    /// The one bridge for this launch, first reached from the menu bar item, the
    /// preview window or a snapshot run.
    func store() -> ModelStore {
        if let activeStore { return activeStore }
        let store = ModelStore(config: config)
        activeStore = store
        return store
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        if let path = config.snapshotPath {
            startSnapshot(path: path, analytics: false)
        } else if let path = config.snapshotAnalytics {
            startSnapshot(path: path, analytics: true)
        } else if config.preview {
            openPreview()
        } else {
            // Bring the bridge up at launch so the panel is live on first open.
            _ = store()
            reportStatusItem()
        }
    }

    /// A `MenuBarExtra` is backed by a status item window in this process.
    /// Report what AppKit actually installed, once, so a packaged launch is
    /// diagnosable without the Accessibility permission that clicking the item
    /// would need.
    private func reportStatusItem() {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            let classes = NSApp.windows.map { String(describing: type(of: $0)) }
            let item = classes.filter { $0.contains("StatusBar") || $0.contains("StatusItem") }
            let line = "slamlm: launch=menubar statusItem="
                + (item.isEmpty ? "not-found" : item.joined(separator: ","))
                + " windows=\(classes.count)\n"
            FileHandle.standardError.write(Data(line.utf8))
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        snapshotTimer?.invalidate()
        activeStore?.shutdown()
    }

    // MARK: - Preview window

    private var initialTab: PanelTab {
        config.snapshotAnalytics != nil ? .analytics : config.tab
    }

    private func openPreview() {
        let store = store()
        let size = PanelLayout.size(for: initialTab)
        let panel = PanelView(
            bus: store.bus,
            store: store,
            initialTab: initialTab,
            onTabChange: { [weak self] tab in self?.resizePreview(to: tab) }
        )
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: size),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = windowTitle(for: initialTab)
        window.contentView = NSHostingView(rootView: panel)
        window.setContentSize(size)
        window.backgroundColor = NSColor(Paper.bg)
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate()
        previewWindow = window
    }

    private func resizePreview(to tab: PanelTab) {
        guard let window = previewWindow else { return }
        window.title = windowTitle(for: tab)
        let size = PanelLayout.size(for: tab)
        var frame = window.frame
        let content = window.contentRect(forFrameRect: frame)
        let deltaHeight = size.height - content.height
        frame.size.width += size.width - content.width
        frame.size.height += deltaHeight
        // Keep the top edge put while the panel grows or shrinks.
        frame.origin.y -= deltaHeight
        window.setFrame(frame, display: true, animate: false)
    }

    private func windowTitle(for tab: PanelTab) -> String {
        "SlamLM — \(tab.title)"
    }

    // MARK: - Snapshots

    /// Renders the panel offscreen once real data has arrived, then exits.
    ///
    /// `--snapshot` waits for the catalog plus a live memory sample.
    /// `--snapshot-analytics` additionally drives the run, because nothing else
    /// will: load `--model` (or the first catalog entry), start `serve`, send
    /// one real prompt of `--tokens` tokens and wait for its `request_end`.
    private func startSnapshot(path: String, analytics: Bool) {
        _ = store()
        let deadline = Date().addingTimeInterval(analytics ? 600 : 90)
        var sentRequests = 0

        snapshotTimer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] timer in
            // The timer is scheduled on the main run loop, so hopping back to
            // the main actor is a static fact, not a guess. Only `self` is
            // captured; the store and bus are non-Sendable.
            MainActor.assumeIsolated {
                guard let self else {
                    timer.invalidate()
                    return
                }
                guard let store = self.activeStore else {
                    self.fail("the bridge store was never created")
                }
                let bus = store.bus
                if Date() > deadline {
                    self.fail("timed out waiting for real data after \(analytics ? 600 : 90)s")
                }
                if let error = bus.errorText {
                    self.fail(error)
                }
                guard !bus.models.isEmpty else { return }
                guard bus.live != nil else { return }
                guard analytics || self.config.loadForSnapshot else {
                    self.finishSnapshot(path: path, store: store, analytics: false)
                    return
                }
                guard let target = self.snapshotModel(in: store) else {
                    self.fail("no catalog entry to load for the snapshot")
                }
                guard store.loadedModelID == target.id, !bus.busy else {
                    if !bus.busy { store.toggle(target) }
                    return
                }
                guard bus.status == .ready else { return }
                if !analytics {
                    self.finishSnapshot(path: path, store: store, analytics: false)
                    return
                }

                guard let target = self.snapshotModel(in: store) else {
                    self.fail("no catalog entry to drive the analytics snapshot")
                }
                guard store.loadedModelID == target.id else {
                    if !bus.busy { store.toggle(target) }
                    return
                }
                guard bus.servingURL != nil else {
                    if !bus.busy { store.toggleServe() }
                    return
                }
                guard bus.status == .ready, !bus.busy else { return }
                // One or more real runs: the second and later ones give the
                // per-request throughput chart more than a single point.
                if let pending = store.lastRequestID,
                   !bus.requests.contains(where: { $0.request == pending }) {
                    return
                }
                guard sentRequests < self.config.requests else {
                    self.finishSnapshot(path: path, store: store, analytics: true)
                    return
                }
                sentRequests += 1
                store.send(prompt: self.snapshotPrompt(round: sentRequests), maxTokens: self.config.tokens)
            }
        }
    }

    private func snapshotModel(in store: ModelStore) -> ModelInfo? {
        if let id = config.model, let model = store.bus.models.first(where: { $0.id == id }) {
            return model
        }
        return store.visibleModels.first ?? store.bus.models.first
    }

    private func finishSnapshot(path: String, store: ModelStore, analytics: Bool) {
        snapshotTimer?.invalidate()
        snapshotTimer = nil

        let tab: PanelTab = analytics ? .analytics : .models
        // `renderStatic` swaps the AppKit-backed controls for static
        // equivalents; `ImageRenderer` cannot draw those, and a snapshot has to
        // show the panel's real content, not an "unsupported view" glyph.
        // `PanelView` already pins its own size, so no outer frame here: a
        // fixed height would clamp the full-page analytics capture back to a crop.
        let panel = PanelView(bus: store.bus, store: store, initialTab: tab, renderStatic: true)

        let renderer = ImageRenderer(content: panel)
        renderer.scale = 2
        guard let image = renderer.cgImage else {
            fail("ImageRenderer produced no image")
        }
        guard let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else {
            fail("could not encode the panel as PNG")
        }
        do {
            try png.write(to: URL(fileURLWithPath: path))
        } catch {
            fail("could not write \(path): \(error.localizedDescription)")
        }
        store.shutdown()
        exit(0)
    }

    /// Never renders a fabricated panel: prints the real reason and leaves.
    private func fail(_ message: String) -> Never {
        snapshotTimer?.invalidate()
        snapshotTimer = nil
        FileHandle.standardError.write(Data("slamlm: \(message)\n".utf8))
        activeStore?.shutdown()
        exit(2)
    }
}
