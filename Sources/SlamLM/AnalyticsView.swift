import SwiftUI

// MARK: - Layout

/// The analytics board's local geometry. The panel itself is `PanelLayout`-sized;
/// the grid uses flexible columns so the cards reflow at any width.
private enum AnalyticsLayout {
    static let cardMinHeight: CGFloat = 150
    static let contentPadding: CGFloat = 14
    static let cardSpacing: CGFloat = PanelLayout.gutter
}

// MARK: - Analytics surface

/// The expanded analytics tab: a scrolling board of metric cards, every one of
/// them bound to a real measurement from the shared `MetricsBus`. A card with no
/// data renders an empty state — never a zero dressed up as a measurement.
struct AnalyticsView: View {
    @ObservedObject var metrics: MetricsBus
    let actions: AnalyticsActions

    @State private var prompt = ""
    /// Set only by the picker surface while it rasterises a snapshot offscreen.
    @Environment(\.renderStaticSnapshot) private var renderStatic

    init(metrics: MetricsBus, actions: AnalyticsActions) {
        self._metrics = ObservedObject(wrappedValue: metrics)
        self.actions = actions
    }

    var body: some View {
        if renderStatic {
            // `ImageRenderer` draws nothing of a `ScrollView`, so the snapshot
            // path lays the same board out directly.
            board.background(Paper.bg)
        } else {
            ScrollView { board }
                .background(Paper.bg)
        }
    }

    private var board: some View {
        VStack(alignment: .leading, spacing: AnalyticsLayout.cardSpacing) {
            HStack(spacing: 8) {
                PaperButton(title: "Models", symbol: "chevron.left", style: .ghost,
                            action: actions.showModels)
                Spacer(minLength: 4)
                if let model = metrics.selected {
                    Text(model.name)
                        .font(PaperFont.meta)
                        .foregroundStyle(Paper.inkSoft)
                        .lineLimit(1)
                }
            }
            MetricsStrip(live: live)
            LazyVGrid(columns: columns, spacing: AnalyticsLayout.cardSpacing) {
                DecodeCard(liveTps: live?.decodeTps ?? 0, stats: stats, delta: delta)
                RuntimeCard(split: split)
                PerformanceCard(totalTokens: sessionTokens > 0 ? sessionTokens : nil, buckets: buckets)
                HistoryCard(liveTps: live?.decodeTps ?? 0, liveSamples: historyTps, requestRates: requestRates)
                MemoryCard(
                    headroom: headroom,
                    mlxActiveBytes: live?.memoryActiveBytes ?? metrics.hardware?.memoryActiveBytes ?? 0,
                    mlxRecommendedBytes: live?.memoryRecommendedBytes ?? metrics.hardware?.recommendedBytes ?? 0,
                    swapBytes: live?.systemSwapBytes ?? metrics.hardware?.systemSwapBytes ?? 0
                )
                TTFTCard(liveTtftMs: live?.ttftMs ?? 0, records: metrics.requests)
            }
            PromptComposer(
                prompt: $prompt,
                modelName: metrics.selected?.name,
                canRun: canRun,
                canCancel: canCancel,
                toolsEnabled: metrics.toolsEnabled,
                onRun: run,
                onCancel: { actions.cancel() },
                onToggleTools: { metrics.toolsEnabled.toggle() }
            )
            if let url = metrics.servingURL {
                EndpointCard(url: url)
            }
            LastOutputCard(text: metrics.streamText, toolEvents: metrics.toolEvents)
        }
        .padding(.horizontal, AnalyticsLayout.contentPadding)
        .padding(.vertical, AnalyticsLayout.contentPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: Derived inputs

    private var columns: [GridItem] {
        [GridItem(.flexible(), spacing: AnalyticsLayout.cardSpacing),
         GridItem(.flexible(), spacing: AnalyticsLayout.cardSpacing)]
    }

    /// The newest telemetry sample, from the live slot or the head of the ring buffer.
    private var live: LiveMetrics? { metrics.live ?? metrics.history.last }
    private var stats: SessionStats { MetricSeries.sessionStats(metrics.requests) }
    private var buckets: [TokenBucket] { MetricSeries.tokenBuckets(metrics.history) }
    private var historyTps: [Double] { MetricSeries.decodeTps(metrics.history) }
    /// Decode rate of each finished request, oldest first.
    private var requestRates: [Double] { MetricSeries.requestDecodeTps(metrics.requests) }
    private var delta: Double? {
        MetricSeries.deltaVsMedian(currentTps: live?.decodeTps ?? 0, requests: metrics.requests)
    }
    private var split: RuntimeSplit? {
        MetricSeries.runtimeSplit(for: metrics.requests.first, loadMs: live?.loadMs ?? 0)
    }
    private var headroom: MemoryHeadroom {
        MetricSeries.systemHeadroom(
            used: live?.systemUsedBytes ?? metrics.hardware?.systemUsedBytes ?? 0,
            total: live?.memoryTotalBytes ?? metrics.hardware?.totalBytes ?? 0
        )
    }
    /// Tokens this session has generated. Telemetry is sampled at 5 Hz, so a
    /// request that has just finished can still be missing from the counter;
    /// the finished records are final and win when they are ahead of it.
    private var sessionTokens: Int {
        let sampled = live?.tokensGenerated ?? 0
        let recorded = metrics.requests.reduce(0) { $0 + max(0, $1.genTokens) }
        let total = max(sampled, recorded)
        return total > 0 ? total : 0
    }


    // MARK: Commands

    private var canRun: Bool {
        metrics.selected != nil && (metrics.status == .ready || metrics.status == .generating)
    }
    private var canCancel: Bool { metrics.status == .generating }

    private func run() {
        actions.send(prompt)
    }
}

// MARK: - Metrics strip

/// Compact strip of the four memory numbers the bridge measures directly.
private struct MetricsStrip: View {
    let live: LiveMetrics?

    var body: some View {
        PaperCard(fill: Paper.card, radius: PanelLayout.corner, padding: 11) {
            VStack(alignment: .leading, spacing: 7) {
                HStack(spacing: 0) {
                    // Machine-wide first: this is the number a person checks
                    // against Activity Monitor.
                    tile("System", live.map { PaperFormat.bytesPair($0.systemUsedBytes, $0.memoryTotalBytes) })
                    divider
                    tile("MLX active", live.map { PaperFormat.bytes($0.memoryActiveBytes) })
                    divider
                    tile("MLX peak", live.map { PaperFormat.bytes($0.memoryPeakBytes) })
                    divider
                    tile("MLX cache", live.map { PaperFormat.bytes($0.memoryCacheBytes) })
                    divider
                    tile("RSS", live.map { PaperFormat.bytes($0.processRssBytes) })
                }
                if let live {
                    Text(systemBreakdown(live))
                        .font(.system(size: 9.5))
                        .foregroundStyle(Paper.inkFaint)
                        .lineLimit(1)
                }
            }
        }
    }

    /// The machine-wide split, in Activity Monitor's terms. Swap is omitted when
    /// there is none rather than shown as a zero.
    private func systemBreakdown(_ live: LiveMetrics) -> String {
        var parts = [
            "app \(PaperFormat.bytes(live.systemAppBytes))",
            "wired \(PaperFormat.bytes(live.systemWiredBytes))",
            "compressed \(PaperFormat.bytes(live.systemCompressedBytes))",
            "cached \(PaperFormat.bytes(live.systemCachedBytes))"
        ]
        if live.systemSwapBytes > 0 {
            parts.append("swap \(PaperFormat.bytes(live.systemSwapBytes))")
        }
        return "system — " + parts.joined(separator: " · ")
    }

    private func tile(_ label: String, _ value: String?) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            SectionLabel(text: label, color: Paper.inkFaint, size: 8.5)
            Text(value ?? "—")
                .font(PaperFont.numeral(16))
                .monospacedDigit()
                .foregroundStyle(value == nil ? Paper.inkFaint : Paper.ink)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var divider: some View {
        Rectangle()
            .fill(Paper.hairline)
            .frame(width: 1, height: 26)
            .padding(.horizontal, 6)
    }
}

// MARK: - Decode (lime)

/// Live decode rate, falling back to the session median only when there is a
/// real median to fall back to, and saying so in the caption.
private struct DecodeCard: View {
    let liveTps: Double
    let stats: SessionStats
    let delta: Double?

    private var usingMedian: Bool { liveTps <= 0 && stats.medianDecodeTps > 0 }
    private var value: Double { usingMedian ? stats.medianDecodeTps : liveTps }
    private var caption: String {
        guard usingMedian else { return "tokens/s · 2 s window" }
        let noun = stats.measuredCount == 1 ? "request" : "requests"
        return "tokens/s · session median of \(stats.measuredCount) \(noun)"
    }

    var body: some View {
        PaperCard(fill: Paper.washOlive, stroke: nil) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Decode", color: Paper.ink.opacity(0.8))
                    Spacer(minLength: 4)
                    if let delta {
                        DeltaPill(text: MetricSeries.deltaText(delta), positive: delta >= 0)
                    }
                }
                if value > 0 {
                    Text(PaperFormat.tps(value))
                        .font(PaperFont.numeral(42))
                        .monospacedDigit()
                        .foregroundStyle(Paper.ink)
                    Spacer(minLength: 2)
                    Text(caption)
                        .font(PaperFont.meta)
                        .foregroundStyle(Paper.ink.opacity(0.75))
                } else {
                    Spacer(minLength: 0)
                    CardEmptyState(symbol: "speedometer", title: "No decode rate yet",
                                   detail: "Load a model and run a prompt to measure tokens/s.",
                                   ink: Paper.ink, faint: Paper.ink.opacity(0.7))
                    Spacer(minLength: 0)
                }
            }
            .frame(minHeight: AnalyticsLayout.cardMinHeight, alignment: .topLeading)
        }
    }
}

// MARK: - Runtime (obsidian)

/// Where the most recent request's wall-clock time went.
private struct RuntimeCard: View {
    let split: RuntimeSplit?

    var body: some View {
        PaperCard(fill: Paper.accent, stroke: nil) {
            VStack(alignment: .leading, spacing: 9) {
                SectionLabel(text: "Runtime", color: Color.white.opacity(0.5))
                if let split {
                    ForEach(split.slices) { slice in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 6) {
                                Text(slice.label)
                                    .font(.system(size: 10.5, weight: .medium))
                                    .foregroundStyle(Color.white.opacity(0.82))
                                Spacer(minLength: 4)
                                Text(slice.valueText)
                                    .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                                    .monospacedDigit()
                                    .foregroundStyle(Color.white)
                            }
                            StatBar(fraction: slice.share, fill: tint(slice.id),
                                    track: Color.white.opacity(0.12), height: 3)
                        }
                    }
                    Spacer(minLength: 2)
                    Text("request #\(split.request) · total \(PaperFormat.ms(split.totalMs)) ms")
                        .font(PaperFont.meta)
                        .monospacedDigit()
                        .foregroundStyle(Color.white.opacity(0.45))
                } else {
                    Spacer(minLength: 0)
                    CardEmptyState(symbol: "timer", title: "No finished request",
                                   detail: "Prefill, decode, load and overhead appear once a generation completes.")
                    Spacer(minLength: 0)
                }
            }
            .frame(minHeight: AnalyticsLayout.cardMinHeight, alignment: .topLeading)
        }
    }

    private func tint(_ id: String) -> Color {
        switch id {
        case "prefill": return Paper.washOlive
        case "decode": return Paper.olive
        case "load": return Paper.washButter
        default: return Color.white.opacity(0.45)
        }
    }
}

// MARK: - Performance (white)

/// Session token total with one bar per measured second of the last minute.
private struct PerformanceCard: View {
    let totalTokens: Int?
    let buckets: [TokenBucket]

    private var peak: Int { buckets.map(\.tokens).max() ?? 0 }

    /// One slot per second of the trailing window, oldest first, so a bar's
    /// horizontal position is its second and the newest second sits at the
    /// right edge. Seconds that were not observed are empty slots rather than
    /// phantom bars, which is what keeps a single active second thin instead of
    /// stretching it across the card.
    private var slots: [TokenBucket] {
        let blank = max(0, AnalyticsThresholds.bucketSeconds - buckets.count)
        guard blank > 0 else { return buckets }
        let blanks = (0..<blank).map { TokenBucket(id: -($0 + 1), startTs: 0, tokens: 0) }
        return blanks + buckets
    }

    var body: some View {
        PaperCard(fill: Paper.card) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Performance · last 60 s")
                    Spacer(minLength: 4)
                    if peak > 0 {
                        TagPill(text: "peak \(peak)/s")
                    }
                }
                if let total = totalTokens, total > 0 {
                    HStack(alignment: .firstTextBaseline, spacing: 5) {
                        Text(PaperFormat.tokens(total))
                            .font(PaperFont.numeral(34))
                            .monospacedDigit()
                            .foregroundStyle(Paper.ink)
                        Text("tokens generated")
                            .font(PaperFont.meta)
                            .foregroundStyle(Paper.inkSoft)
                    }
                    if buckets.isEmpty {
                        CardEmptyState(symbol: "chart.bar.xaxis", title: "No tokens in the last 60 s",
                                       detail: "Bars are per-second counts from the session token counter.",
                                       ink: Paper.inkSoft, faint: Paper.inkFaint)
                    } else {
                        HStack(alignment: .bottom, spacing: 1.5) {
                            ForEach(slots) { bucket in
                                TallBarMark(
                                    fraction: peak > 0 ? Double(bucket.tokens) / Double(peak) : 0,
                                    fill: bucket.tokens == peak ? Paper.accent : Paper.olive,
                                    radius: 2,
                                    minHeight: bucket.tokens > 0 ? 2 : 0,
                                    emphasized: bucket.tokens == peak
                                )
                            }
                        }
                        .frame(height: 64)
                    }
                } else {
                    Spacer(minLength: 0)
                    CardEmptyState(symbol: "chart.bar.xaxis", title: "No tokens generated yet",
                                   detail: "The session total and its per-second bars appear once a request streams.",
                                   ink: Paper.inkSoft, faint: Paper.inkFaint)
                    Spacer(minLength: 0)
                }
                Spacer(minLength: 0)
            }
            .frame(minHeight: AnalyticsLayout.cardMinHeight, alignment: .topLeading)
        }
    }
}

// MARK: - History (obsidian)

/// Decode rate over the telemetry window.
private struct HistoryCard: View {
    /// Live 2 s-window rate, 5 Hz — the fallback before two requests exist.
    let liveTps: Double
    let liveSamples: [Double]
    /// Decode rate of each finished request, oldest first.
    let requestRates: [Double]

    /// Two finished requests make a trend, which is the point of this card: it
    /// answers "is this model faster than the last one". Until then the live
    /// series is the only real measurement available.
    private var perRequest: Bool { requestRates.count >= 2 }
    private var series: [Double] { perRequest ? requestRates : liveSamples }
    private var range: SeriesRange { MetricSeries.range(series) }

    private var subtitle: String {
        perRequest ? "per request · last \(requestRates.count)" : "live · 5 Hz"
    }

    private var headline: Double {
        perRequest ? MetricSeries.mean(requestRates) : liveTps
    }

    private var headlineCaption: String {
        if perRequest { return "mean tok/s" }
        return liveTps > 0 ? "tok/s now" : "idle"
    }

    /// Enough data for whichever series is being drawn.
    private var hasEnough: Bool {
        perRequest ? true : liveSamples.count >= AnalyticsThresholds.lineChartSamples
    }

    var body: some View {
        PaperCard(fill: Paper.accent, stroke: nil) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Throughput", color: Color.white.opacity(0.5))
                    Spacer(minLength: 4)
                    Text(subtitle)
                        .font(PaperFont.meta)
                        .foregroundStyle(Color.white.opacity(0.55))
                }
                if !range.hasSignal {
                    Spacer(minLength: 0)
                    CardEmptyState(
                        symbol: "chart.xyaxis.line",
                        title: "No decode rate yet",
                        detail: "Telemetry arrives at 5 Hz; the rate is measured while a prompt is generating."
                    )
                    Spacer(minLength: 0)
                } else if !hasEnough {
                    Spacer(minLength: 0)
                    CardEmptyState(
                        symbol: "chart.xyaxis.line",
                        title: "Not enough history",
                        detail: "\(liveSamples.count) of \(AnalyticsThresholds.lineChartSamples) telemetry samples received."
                    )
                    Spacer(minLength: 0)
                } else {
                    HStack(alignment: .firstTextBaseline, spacing: 5) {
                        Text(PaperFormat.tps(headline))
                            .font(PaperFont.numeral(34))
                            .monospacedDigit()
                            .foregroundStyle(Color.white)
                        Text(headlineCaption)
                            .font(PaperFont.meta)
                            .foregroundStyle(Color.white.opacity(0.5))
                    }
                    ZStack {
                        GridLines(count: 4, color: Color.white.opacity(0.07))
                        LineAreaMark(values: series, line: Paper.olive,
                                     fill: Paper.olive.opacity(0.18), node: Paper.olive,
                                     nodes: max(2, min(series.count, 12)))
                    }
                    .frame(height: 60)
                    HStack(spacing: 6) {
                        // With a per-request series the x axis is request order,
                        // so the ends are named rather than left to the reader.
                        Text(perRequest ? "min \(PaperFormat.tps(range.min))" : "min \(range.min > 0 ? PaperFormat.tps(range.min) : "0.0")")
                        Spacer(minLength: 4)
                        Text("oldest → newest")
                            .font(.system(size: 9, weight: .medium, design: .rounded))
                            .foregroundStyle(Color.white.opacity(0.3))
                        Spacer(minLength: 4)
                        Text("max \(PaperFormat.tps(range.max))")
                    }
                    .font(.system(size: 9.5, weight: .medium, design: .rounded))
                    .monospacedDigit()
                    .foregroundStyle(Color.white.opacity(0.5))
                }
                Spacer(minLength: 0)
            }
            .frame(minHeight: AnalyticsLayout.cardMinHeight, alignment: .topLeading)
        }
    }
}

// MARK: - Memory headroom (butter)

/// Active bytes against the recommended working set.
private struct MemoryCard: View {
    let headroom: MemoryHeadroom
    /// MLX's own accounting and ceiling, reported separately: they are not free
    /// memory, and mixing them into the "free" figure is what made this card lie.
    let mlxActiveBytes: Int
    let mlxRecommendedBytes: Int
    let swapBytes: Int

    private var mlxLine: String? {
        guard mlxActiveBytes > 0 || mlxRecommendedBytes > 0 else { return nil }
        var parts = ["MLX \(PaperFormat.bytes(mlxActiveBytes))"]
        if mlxRecommendedBytes > 0 {
            parts.append("\(PaperFormat.bytes(mlxRecommendedBytes)) ceiling")
        }
        if swapBytes > 0 { parts.append("swap \(PaperFormat.bytes(swapBytes))") }
        return parts.joined(separator: " / ")
    }

    var body: some View {
        PaperCard(fill: Paper.washButter, stroke: nil) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Memory headroom", color: Paper.butterInk.opacity(0.7))
                    Spacer(minLength: 4)
                    if headroom.measured {
                        TagPill(text: "\(headroom.freeText) free",
                                fill: Paper.washButter, ink: Paper.butterInk)
                    }
                }
                if headroom.measured {
                    // A nearly empty fan is the honest picture of a machine with
                    // little left; the gauge fills with the free share, not with
                    // a budget the model has not spent.
                    ZStack {
                        TickFanGauge(fraction: headroom.fraction, ink: Paper.butterInk,
                                     track: Paper.butterInk.opacity(0.18))
                        Text(headroom.percentText)
                            .font(PaperFont.numeral(28))
                            .monospacedDigit()
                            .foregroundStyle(Paper.butterInk)
                    }
                    .frame(height: 94)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(headroom.caption)
                        if let mlxLine {
                            Text(mlxLine).foregroundStyle(Paper.butterInk.opacity(0.6))
                        }
                    }
                    .font(PaperFont.meta)
                    .monospacedDigit()
                    .foregroundStyle(Paper.butterInk.opacity(0.8))
                } else {
                    Spacer(minLength: 0)
                    CardEmptyState(symbol: "memorychip", title: "No memory sample yet",
                                   detail: "The bridge reports the machine's used and free memory at 5 Hz.",
                                   ink: Paper.butterInk.opacity(0.85), faint: Paper.butterInk.opacity(0.65))
                    Spacer(minLength: 0)
                }
                Spacer(minLength: 0)
            }
            .frame(minHeight: AnalyticsLayout.cardMinHeight, alignment: .topLeading)
        }
    }
}

// MARK: - TTFT (orange)

/// Time to first token of recent requests, with the prefill share of each TTFT
/// as the deep-tone base of the bar.
private struct TTFTCard: View {
    let liveTtftMs: Double
    /// Newest first, as the bus stores them.
    let records: [RequestRecord]

    private var recent: [RequestRecord] { Array(records.prefix(12).reversed()) }
    private var peakTtft: Double { recent.map(\.ttftMs).max() ?? 0 }
    private var latestTtft: Double {
        if liveTtftMs > 0 { return liveTtftMs }
        return records.first { $0.ttftMs > 0 }?.ttftMs ?? 0
    }

    var body: some View {
        PaperCard(fill: Paper.washClay, stroke: nil) {
            VStack(alignment: .leading, spacing: 8) {
                SectionLabel(text: recent.isEmpty
                             ? "TTFT"
                             : "TTFT · last \(recent.count) request\(recent.count == 1 ? "" : "s")",
                             color: Paper.clayInk.opacity(0.85))
                if recent.isEmpty {
                    Spacer(minLength: 0)
                    CardEmptyState(symbol: "bolt.horizontal", title: "No finished request",
                                   detail: "Time to first token appears here after the first generation.",
                                   ink: Paper.clayInk, faint: Paper.clayInk.opacity(0.75))
                    Spacer(minLength: 0)
                } else {
                    HStack(alignment: .firstTextBaseline, spacing: 5) {
                        Text(PaperFormat.ms(latestTtft))
                            .font(PaperFont.numeral(32))
                            .monospacedDigit()
                            .foregroundStyle(Paper.clayInk)
                        Text("ms to first token")
                            .font(PaperFont.meta)
                            .foregroundStyle(Paper.clayInk.opacity(0.85))
                    }
                    HStack(alignment: .bottom, spacing: 3) {
                        ForEach(Array(recent.enumerated()), id: \.element.id) { position, record in
                            VStack(spacing: 3) {
                                StackedBarMark(
                                    fraction: peakTtft > 0 ? record.ttftMs / peakTtft : 0,
                                    deepFraction: prefillShare(record),
                                    deep: Paper.clayDeep,
                                    light: Paper.clay,
                                    radius: 2,
                                    minHeight: record.ttftMs > 0 ? 2 : 0
                                )
                                // The axis is the window, not the request id: ids
                                // are internal and run to seven digits for HTTP
                                // traffic, which cannot fit in a column. The
                                // exact id for the newest request is in the
                                // runtime card's caption.
                                Text("\(position + 1)")
                                    .font(.system(size: 8, weight: .medium, design: .rounded))
                                    .monospacedDigit()
                                    .lineLimit(1)
                                    .foregroundStyle(Paper.clayInk.opacity(0.75))
                            }
                            .frame(maxWidth: 40)
                        }
                    }
                    .frame(height: 70)
                }
                Spacer(minLength: 0)
            }
            .frame(minHeight: AnalyticsLayout.cardMinHeight, alignment: .topLeading)
        }
    }

    /// Prefill's share of this request's TTFT; the deep base of its bar.
    private func prefillShare(_ record: RequestRecord) -> Double {
        guard record.ttftMs > 0, record.prefillTps > 0 else { return 0 }
        let prefillMs = Double(record.promptTokens) / record.prefillTps * 1000
        return min(1, max(0, prefillMs / record.ttftMs))
    }
}

// MARK: - Prompt composer

private struct PromptComposer: View {
    @Binding var prompt: String
    let modelName: String?
    let canRun: Bool
    let canCancel: Bool
    let toolsEnabled: Bool
    let onRun: () -> Void
    let onCancel: () -> Void
    let onToggleTools: () -> Void

    @Environment(\.renderStaticSnapshot) private var renderStatic

    var body: some View {
        PaperCard(fill: Paper.card) {
            VStack(alignment: .leading, spacing: 9) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Prompt")
                    Spacer(minLength: 4)
                    Text(modelName ?? "no model loaded")
                        .font(PaperFont.meta)
                        .foregroundStyle(Paper.inkSoft)
                        .lineLimit(1)
                }
                HStack(spacing: 8) {
                    promptField
                    ToolsSwitch(enabled: toolsEnabled, action: onToggleTools)
                    PaperButton(title: "Run", symbol: "play.fill", style: .primary,
                                enabled: canRun, action: onRun)
                    PaperButton(title: "Cancel", symbol: "stop.fill", style: .ghost,
                                enabled: canCancel, action: onCancel)
                }
            }
        }
    }

    /// `ImageRenderer` cannot draw a `TextField`, so a snapshot gets the same
    /// chrome around plain text.
    @ViewBuilder private var promptField: some View {
        Group {
            if renderStatic {
                Text(prompt.isEmpty ? "Ask the loaded model…" : prompt)
                    .foregroundStyle(prompt.isEmpty ? Paper.fieldPlaceholder : Paper.fieldInk)
            } else {
                TextField("", text: $prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .foregroundStyle(Paper.fieldInk)
                    .onSubmit(onRun)
                    .overlay(alignment: .topLeading) {
                        if prompt.isEmpty {
                            Text("Ask the loaded model…")
                                .foregroundStyle(Paper.fieldPlaceholder)
                                .allowsHitTesting(false)
                        }
                    }
            }
        }
        .font(PaperFont.body)
        .lineLimit(1...3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Paper.field))
        .overlay(RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(Paper.stroke, lineWidth: 1))
    }
}

/// Lets the model call the read-only tools. A switch rather than a checkbox so
/// it reads at a glance, and it carries the state rather than hiding it in a menu.
private struct ToolsSwitch: View {
    let enabled: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: enabled ? "wrench.and.screwdriver.fill" : "wrench.and.screwdriver")
                    .font(.system(size: 9, weight: .semibold))
                Text("Tools")
                    .font(.system(size: 11, weight: .semibold))
            }
            .foregroundStyle(enabled ? Paper.accentInk : Paper.inkSoft)
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(Capsule().fill(enabled ? Paper.accent : Paper.card))
            .overlay(Capsule().strokeBorder(enabled ? .clear : Paper.stroke, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .help(enabled
              ? "The model may search the web and read files (read-only) before answering"
              : "Tools off: the model answers from its own weights")
        .accessibilityLabel(Text(enabled ? "Disable tools" : "Enable tools"))
    }
}

private struct EndpointCard: View {
    let url: String

    var body: some View {
        PaperCard(fill: Paper.card) {
            HStack(spacing: 10) {
                GlyphTile(symbol: "network", tint: Paper.cardSunken, size: 30, glyphSize: 14)
                VStack(alignment: .leading, spacing: 3) {
                    SectionLabel(text: "Serving locally")
                    Text(url)
                        .font(.system(size: 11.5, weight: .medium, design: .monospaced))
                        .foregroundStyle(Paper.ink)
                        .textSelection(.enabled)
                    Text("/v1/chat/completions · /v1/completions · /metrics")
                        .font(PaperFont.meta)
                        .foregroundStyle(Paper.inkFaint)
                }
                Spacer(minLength: 0)
            }
        }
    }
}

// MARK: - Last output

private struct LastOutputCard: View {
    let text: String
    /// Tool calls and results for the request that produced this output.
    let toolEvents: [ToolEvent]

    var body: some View {
        PaperCard(fill: Paper.card) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Last output")
                    Spacer(minLength: 4)
                    if !text.isEmpty {
                        TagPill(text: "\(text.count) chars")
                    }
                }
                if text.isEmpty {
                    EmptyState(symbol: "text.alignleft", title: "No output yet",
                               detail: "Tokens streamed for the most recent request appear here.")
                } else {
                    ToolTrace(events: toolEvents)
                    MarkdownText(raw: text)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }
}
