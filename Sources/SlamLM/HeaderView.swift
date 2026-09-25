import AppKit
import SwiftUI

/// Round ghost button that pops a menu.
///
/// `Design.swift`'s `GlyphButton` is a plain action button; the header and the
/// footer need the same chrome around a menu, so the primitive lives here.
///
/// `renderStatic` draws the identical chrome without the `Menu`, because
/// `ImageRenderer` cannot draw AppKit-backed controls and a snapshot still has
/// to look like the panel it captures.
struct GhostMenu<Content: View>: View {
    let symbol: String
    var diameter: CGFloat = 26
    var glyphSize: CGFloat = 11.5
    var enabled = true
    var renderStatic = false
    /// Spoken label; an SF Symbol name is not one.
    var label: String = ""
    @ViewBuilder var content: Content

    @State private var hovering = false

    var body: some View {
        if renderStatic {
            glyph
        } else {
            Menu {
                content
            } label: {
                glyph
            }
            // `.borderlessButton` rasterises the label into a template image and
            // draws it as a white silhouette; the `.button` style with a plain
            // button style is the one that renders the label as drawn.
            .menuStyle(.button)
            .buttonStyle(.plain)
            .menuIndicator(.hidden)
            .fixedSize()
            .disabled(!enabled)
        }
    }

    private var glyph: some View {
        ZStack {
            Circle().fill(Paper.cardSunken)
            Circle().strokeBorder(Paper.stroke, lineWidth: 1)
            Image(systemName: symbol)
                .font(.system(size: glyphSize, weight: .semibold))
                .foregroundStyle(Paper.ink.opacity(0.8))
        }
        .frame(width: diameter, height: diameter)
        .opacity(hovering ? 0.82 : 1)
        .onHover { hovering = $0 }
        .accessibilityLabel(Text(label.isEmpty ? symbol : label))
        .help(label)
    }
}

/// Sunken search pill with a magnifier, a `⌘K` hint and a keyboard shortcut
/// that pulls focus into the field.
struct SearchField: View {
    @Binding var text: String
    /// Draws the pill without the `TextField`, which `ImageRenderer` cannot
    /// render; see `GhostMenu`.
    var renderStatic = false

    @FocusState private var focused: Bool

    var body: some View {
        field
            .padding(.horizontal, 9)
            .frame(height: 28)
            .background(
                RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Paper.field)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(Paper.stroke, lineWidth: 1)
            )
            .contentShape(Rectangle())
            .onTapGesture { focused = true }
            // A zero-sized button is the reliable way to own a window-level key
            // equivalent from inside a `MenuBarExtra` panel.
            .background(
                Button("Search") { focused = true }
                    .keyboardShortcut("k", modifiers: .command)
                    .buttonStyle(.plain)
                    .frame(width: 0, height: 0)
                    .opacity(0)
            )
    }

    @ViewBuilder private var field: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(Paper.fieldPlaceholder)
            if renderStatic {
                Text(text.isEmpty ? "Search models…" : text)
                    .font(PaperFont.body)
                    .foregroundStyle(text.isEmpty ? Paper.fieldPlaceholder : Paper.fieldInk)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                // The system placeholder tint is far too faint on this fill, so
                // the placeholder is drawn here in a colour that clears 4.5:1.
                TextField("", text: $text)
                    .textFieldStyle(.plain)
                    .font(PaperFont.body)
                    .foregroundStyle(Paper.fieldInk)
                    .focused($focused)
                    .overlay(alignment: .leading) {
                        if text.isEmpty {
                            Text("Search models…")
                                .font(PaperFont.body)
                                .foregroundStyle(Paper.fieldPlaceholder)
                                .allowsHitTesting(false)
                        }
                    }
            }
            KeyHint(text: "⌘K")
        }
    }
}

/// Panel header: brain tile, title, honest status line, the search field and the
/// settings / reveal-in-Finder buttons.
struct HeaderView: View {
    @ObservedObject var bus: MetricsBus
    @ObservedObject var store: ModelStore
    @Binding var query: String
    var renderStatic = false

    var body: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                GlyphTile(symbol: "brain", tint: Paper.card, size: 34, glyphSize: 17)
                VStack(alignment: .leading, spacing: 2) {
                    Text("SlamLM")
                        .font(PaperFont.panelTitle)
                        .foregroundStyle(Paper.ink)
                    statusLine
                }
                Spacer(minLength: 8)
                GhostMenu(symbol: "gearshape", renderStatic: renderStatic, label: "Settings") { settingsMenu }
                GlyphButton(symbol: "folder", label: "Reveal model in Finder", action: revealSelected)
                    .opacity(selectedPath == nil ? 0.45 : 1)
                    .allowsHitTesting(selectedPath != nil)
            }
            SearchField(text: $query, renderStatic: renderStatic)
        }
        .padding(.horizontal, PanelLayout.gutter)
        .padding(.top, 12)
        .padding(.bottom, 10)
    }

    // MARK: - Status

    private var statusLine: some View {
        HStack(spacing: 5) {
            StatusDot(color: statusColor)
            Text(statusText)
                .font(PaperFont.meta)
                .foregroundStyle(Paper.inkSoft)
                .lineLimit(1)
            if let name = runningModelName {
                Text(name)
                    .font(PaperFont.meta)
                    .foregroundStyle(Paper.inkFaint)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
    }

    private var statusText: String {
        if bus.status == .error { return "Bridge failed" }
        if !bus.bridgeReady { return "Starting bridge…" }
        switch bus.status {
        case .ready, .generating: return "Running locally"
        case .loading: return "Loading \(loadingModelName)…"
        case .idle, .error: return "Idle"
        }
    }

    private var statusColor: Color {
        switch bus.status {
        case .ready, .generating: return bus.bridgeReady ? Paper.running : Paper.inkFaint
        case .loading: return Paper.olive
        case .error: return Paper.clayDeep
        case .idle: return Paper.inkFaint
        }
    }

    /// The loaded model's display name, resolved through the catalog when it is
    /// there and falling back to the raw id.
    private func displayName(for id: String?) -> String? {
        guard let id else { return nil }
        return bus.models.first { $0.id == id }?.name ?? id
    }

    private var runningModelName: String? {
        guard bus.status == .ready || bus.status == .generating else { return nil }
        return displayName(for: bus.live?.model)
    }

    private var loadingModelName: String {
        displayName(for: bus.live?.model ?? bus.selected?.id) ?? "model"
    }

    // MARK: - Actions

    private var selectedPath: String? {
        store.loadedModel?.path ?? bus.selected?.path
    }

    private func revealSelected() {
        guard let path = selectedPath else { return }
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }

    @ViewBuilder private var settingsMenu: some View {
        Button("Refresh catalog") { store.refreshCatalog() }
        Button(bus.servingURL == nil ? "Start endpoint" : "Stop endpoint") { store.toggleServe() }
            .disabled(store.loadedModel == nil && bus.servingURL == nil)
        Button("Copy endpoint URL") { HeaderView.copyToPasteboard(bus.servingURL) }
            .disabled(bus.servingURL == nil)
        Divider()
        Button("Quit SlamLM") { NSApplication.shared.terminate(nil) }
    }

    static func copyToPasteboard(_ text: String?) {
        guard let text else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }
}
