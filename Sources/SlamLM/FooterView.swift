import SwiftUI

/// Category chips: `All` first, then the bridge's canonical order. Lives above
/// the model list (the reference board's pill row) and writes `store.category`.
struct CategoryFilterRow: View {
    let chips: [String]
    @Binding var selection: String
    /// Drops the `ScrollView` for static rendering; see `GhostMenu`.
    var renderStatic = false

    var body: some View {
        Group {
            if renderStatic {
                chipsRow.padding(.horizontal, PanelLayout.gutter)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    chipsRow.padding(.horizontal, PanelLayout.gutter)
                }
            }
        }
        .frame(height: 30)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var chipsRow: some View {
        HStack(spacing: 6) {
            ForEach(chips, id: \.self) { chip in
                CategoryChip(title: chip, selected: selection == chip) {
                    selection = chip
                }
            }
        }
        .padding(.vertical, 1)
    }
}

/// The footer card: hardware, real memory bars, the endpoint indicator, the live
/// throughput readout and the two round ghost buttons.
struct FooterView: View {
    @ObservedObject var bus: MetricsBus
    @ObservedObject var store: ModelStore
    @Binding var tab: PanelTab
    var renderStatic = false

    var body: some View {
        VStack(spacing: 8) {
            PaperCard(padding: 11) {
                VStack(spacing: 8) {
                    hardwareLine
                    if let memory {
                        usageRow(
                            label: "Memory",
                            fraction: fraction(memory.systemUsedBytes, of: memory.totalBytes),
                            value: PaperFormat.bytesPair(memory.systemUsedBytes, memory.totalBytes)
                        )
                        .help(memory.help)
                        usageRow(
                            label: "Model",
                            fraction: fraction(memory.modelBytes, of: memory.totalBytes),
                            value: PaperFormat.bytes(memory.modelBytes),
                            fill: Paper.olive
                        )
                        .help("The loaded model's MLX allocation, inside the same unified memory as the bar above")
                        Text(memory.breakdown)
                            .font(.system(size: 9.5))
                            .foregroundStyle(Paper.inkFaint)
                            .lineLimit(1)
                            .truncationMode(.tail)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        EmptyState(symbol: "memorychip", title: "No memory sample yet")
                    }
                    PaperRule()
                    endpointRow
                    if let logLine = bus.logLine {
                        Text(logLine)
                            .font(.system(size: 9, weight: .regular))
                            .foregroundStyle(Paper.inkFaint)
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        }
        .padding(.horizontal, PanelLayout.gutter)
        .padding(.top, 6)
        .padding(.bottom, 10)
    }

    // MARK: - Hardware and memory

    @ViewBuilder private var hardwareLine: some View {
        HStack(spacing: 8) {
            GlyphTile(symbol: "memorychip", tint: Paper.cardSunken, size: 28, glyphSize: 14)
            VStack(alignment: .leading, spacing: 1) {
                Text(bus.hardware?.chip ?? "Hardware unknown")
                    .font(PaperFont.rowTitle)
                    .foregroundStyle(bus.hardware == nil ? Paper.inkSoft : Paper.ink)
                    .lineLimit(1)
                if let gpu = gpuLine {
                    Text(gpu)
                        .font(PaperFont.meta)
                        .foregroundStyle(Paper.inkSoft)
                }
            }
            Spacer(minLength: 6)
            throughput
        }
    }

    /// `10-core GPU`, omitted when the probe could not read the core count.
    private var gpuLine: String? {
        guard let hardware = bus.hardware, hardware.gpuCores > 0 else { return nil }
        return "\(hardware.gpuCores)-core GPU"
    }

    /// What the two bars report.
    ///
    /// "Memory" is machine-wide in Activity Monitor's terms. "Model" is what this
    /// bridge holds in MLX. On Apple silicon those are the same physical pool —
    /// there is no separate GPU memory to show — so both are drawn against
    /// physical RAM and the model's share is labelled as a share.
    private struct MemoryBars {
        let systemUsedBytes: Int
        let modelBytes: Int
        let totalBytes: Int
        let appBytes: Int
        let wiredBytes: Int
        let compressedBytes: Int
        let cachedBytes: Int
        let swapBytes: Int

        /// The machine's split, shown under the bars.
        var breakdown: String {
            var parts = [
                "app \(PaperFormat.bytes(appBytes))",
                "wired \(PaperFormat.bytes(wiredBytes))",
                "compressed \(PaperFormat.bytes(compressedBytes))"
            ]
            if swapBytes > 0 { parts.append("swap \(PaperFormat.bytes(swapBytes))") }
            return parts.joined(separator: " · ")
        }

        var help: String {
            "System \(PaperFormat.bytes(systemUsedBytes)) used of "
                + "\(PaperFormat.bytes(totalBytes)) "
                + "(cached \(PaperFormat.bytes(cachedBytes))) · "
                + "model holds \(PaperFormat.bytes(modelBytes)) of the same unified memory"
        }
    }

    /// Live sample first — it is the 5 Hz measurement — falling back to the
    /// handshake snapshot until the first `metrics` event lands.
    private var memory: MemoryBars? {
        let used: Int
        let active: Int
        let cache: Int
        let total: Int
        let app: Int
        let wired: Int
        let compressed: Int
        let cached: Int
        let swap: Int
        if let live = bus.live {
            used = live.systemUsedBytes
            active = live.memoryActiveBytes
            cache = live.memoryCacheBytes
            total = live.memoryTotalBytes
            app = live.systemAppBytes
            wired = live.systemWiredBytes
            compressed = live.systemCompressedBytes
            cached = live.systemCachedBytes
            swap = live.systemSwapBytes
        } else if let hardware = bus.hardware {
            used = hardware.systemUsedBytes
            active = hardware.memoryActiveBytes
            cache = hardware.cacheBytes
            total = hardware.totalBytes
            app = hardware.systemAppBytes
            wired = hardware.systemWiredBytes
            compressed = hardware.systemCompressedBytes
            cached = hardware.systemCachedBytes
            swap = hardware.systemSwapBytes
        } else {
            return nil
        }
        return MemoryBars(
            systemUsedBytes: used,
            modelBytes: active + cache,
            totalBytes: total,
            appBytes: app,
            wiredBytes: wired,
            compressedBytes: compressed,
            cachedBytes: cached,
            swapBytes: swap
        )
    }

    private func fraction(_ used: Int, of total: Int) -> Double {
        guard total > 0 else { return 0 }
        return Double(used) / Double(total)
    }

    private func usageRow(label: String, fraction: Double, value: String, fill: Color = Paper.running) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(PaperFont.meta)
                .foregroundStyle(Paper.inkSoft)
                .frame(width: 46, alignment: .leading)
            StatBar(fraction: fraction, fill: fill)
            Text(value)
                .font(PaperFont.meta)
                .foregroundStyle(Paper.inkSoft)
                .monospacedDigit()
                .lineLimit(1)
                .frame(width: 84, alignment: .trailing)
        }
    }

    // MARK: - Throughput

    private var throughput: some View {
        VStack(alignment: .trailing, spacing: 1) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(PaperFormat.tps(bus.live?.decodeTps ?? 0))
                    .font(PaperFont.numeral(13))
                    .foregroundStyle(Paper.ink)
                Text("tok/s")
                    .font(.system(size: 9, weight: .medium))
                    .foregroundStyle(Paper.inkFaint)
            }
            Text(contextLine)
                .font(.system(size: 9, weight: .regular))
                .foregroundStyle(Paper.inkFaint)
        }
    }

    private var contextLine: String {
        guard let model = store.loadedModel else { return "no model" }
        return model.contextLength > 0 ? "\(PaperFormat.tokens(model.contextLength)) ctx" : "ctx unknown"
    }

    // MARK: - Endpoint and buttons

    private var endpointRow: some View {
        HStack(spacing: 6) {
            StatusDot(color: bus.servingURL == nil ? Paper.inkFaint : Paper.running)
            if let url = bus.servingURL {
                Text(url)
                    .font(PaperFont.meta)
                    .foregroundStyle(Paper.ink)
                    .lineLimit(1)
                    .truncationMode(.middle)
                GlyphButton(symbol: "doc.on.doc", diameter: 20, glyphSize: 9.5,
                            label: "Copy endpoint URL") {
                    HeaderView.copyToPasteboard(url)
                }
            } else {
                Text("Endpoint stopped")
                    .font(PaperFont.meta)
                    .foregroundStyle(Paper.inkFaint)
            }
            Spacer(minLength: 6)
            GlyphButton(symbol: "chart.xyaxis.line", label: "Show analytics") { tab = .analytics }
            GhostMenu(symbol: "slider.horizontal.3", renderStatic: renderStatic, label: "Endpoint settings") { endpointMenu }
        }
    }

    @ViewBuilder private var endpointMenu: some View {
        Button("Refresh catalog") { store.refreshCatalog() }
        Button(bus.servingURL == nil ? "Start endpoint" : "Stop endpoint") { store.toggleServe() }
            .disabled(bus.servingURL == nil && store.loadedModel == nil)
        Button("Copy endpoint URL") { HeaderView.copyToPasteboard(bus.servingURL) }
            .disabled(bus.servingURL == nil)
    }
}
