import SwiftUI

// MARK: - Thresholds
//
// Minimum data a card needs before it shows a number at all. Cards and
// derivations share these, so an empty state and an empty derivation always
// agree on why nothing is shown.

enum AnalyticsThresholds {
    /// Width of the per-second token bucket window.
    static let bucketSeconds = 60
    /// History samples needed before the line chart is drawn.
    static let lineChartSamples = 8
    /// Finished requests needed before a delta against the median is honest.
    static let deltaRequests = 3
}

// MARK: - Series extraction
//
// Everything here is a pure function of `MetricsBus` state. Nothing is cached,
// nothing is estimated: a metric with no measurement returns an empty series or
// `nil`, and the card renders an empty state instead of a fabricated zero.

/// Summary of one numeric series, used for the history card's axis labels.
struct SeriesRange: Hashable {
    let min: Double
    let max: Double
    let mean: Double
    let last: Double

    static let empty = SeriesRange(min: 0, max: 0, mean: 0, last: 0)

    /// True when the series carries at least one real measurement.
    var hasSignal: Bool { max > 0 }
}

/// One second of generated tokens, derived from `tokensGenerated` deltas.
struct TokenBucket: Identifiable, Hashable {
    /// Bucket index inside the returned window; oldest is 0.
    let id: Int
    /// Absolute wall-clock second the bucket starts at (same clock as `ts`).
    let startTs: Double
    let tokens: Int
}

/// Aggregates over `MetricsBus.requests`, which arrives newest first.
struct SessionStats: Hashable {
    /// Every request the session recorded, including cancelled and failed ones.
    let count: Int
    /// Requests that carry a real decode measurement (`decodeTps > 0`).
    let measuredCount: Int
    let meanDecodeTps: Double
    let medianDecodeTps: Double
    let maxDecodeTps: Double
    let meanTtftMs: Double
    let lastTtftMs: Double
    let totalGenTokens: Int
    let peakMemBytes: Int
    let meanPrefillTps: Double

    static let empty = SessionStats(
        count: 0, measuredCount: 0,
        meanDecodeTps: 0, medianDecodeTps: 0, maxDecodeTps: 0,
        meanTtftMs: 0, lastTtftMs: 0,
        totalGenTokens: 0, peakMemBytes: 0, meanPrefillTps: 0
    )

    var hasMeasurements: Bool { measuredCount > 0 }
}

/// How much memory the machine actually has left.
///
/// This used to measure the model against MLX's recommended working set and call
/// the remainder "free", which reported `11.7 GB free` on a machine that was
/// 14 GB into swap: the recommended set is a ceiling for MLX allocations, not
/// available memory. Real headroom comes from the bridge's system sample.
struct MemoryHeadroom: Hashable {
    let freeBytes: Int
    let usedBytes: Int
    let totalBytes: Int
    /// `freeBytes / totalBytes`, clamped to 0…1.
    let fraction: Double

    static let empty = MemoryHeadroom(freeBytes: 0, usedBytes: 0, totalBytes: 0, fraction: 0)

    /// False when the machine has not reported a system memory sample.
    var measured: Bool { totalBytes > 0 }
    var freeText: String { PaperFormat.bytes(freeBytes) }
    var percentText: String { PaperFormat.percent(fraction) }
    var caption: String { "used \(PaperFormat.bytes(usedBytes)) of \(PaperFormat.bytes(totalBytes))" }
}

/// One row of the runtime breakdown.
struct RuntimeSlice: Identifiable, Hashable {
    /// `"prefill"`, `"decode"`, `"load"` or `"overhead"`.
    let id: String
    let label: String
    let ms: Double
    /// Share of the request's `totalMs`, clamped to 0…1 for the bar.
    let share: Double
    /// False when the measurement itself is missing (a zero that means "no data").
    let measured: Bool

    var valueText: String {
        if ms > 0 { return "\(PaperFormat.ms(ms)) ms" }
        return measured ? "0 ms" : "—"
    }
}

/// Where the most recent request's wall-clock time went.
struct RuntimeSplit: Hashable {
    let request: Int
    let totalMs: Double
    /// Model load time from the live telemetry; not part of `totalMs`.
    let loadMs: Double
    let prefillMs: Double
    let decodeMs: Double
    let overheadMs: Double
    /// prefill, decode, load, overhead — in that order.
    let slices: [RuntimeSlice]
}

enum MetricSeries {
    // MARK: History series (newest last)

    /// Decode tokens/s for every sample in the ring buffer.
    static func decodeTps(_ history: [LiveMetrics]) -> [Double] {
        history.map(\.decodeTps)
    }

    /// Decode rate of each finished request, oldest first — the series that
    /// shows whether this model is actually faster than the last one. Requests
    /// that never measured a rate (a failure or a cancel before the first token)
    /// are left out rather than plotted as a zero.
    static func requestDecodeTps(_ requests: [RequestRecord]) -> [Double] {
        Array(requests.compactMap { $0.decodeTps > 0 ? $0.decodeTps : nil }.reversed())
    }

    /// min/max/mean/last of a series; `.empty` for an empty series.
    static func range(_ values: [Double]) -> SeriesRange {
        guard let maxValue = values.max(), let minValue = values.min() else { return .empty }
        let total = values.reduce(0, +)
        return SeriesRange(
            min: minValue,
            max: maxValue,
            mean: total / Double(values.count),
            last: values[values.count - 1]
        )
    }

    // MARK: Per-second token buckets

    /// Generated tokens per second over the trailing window.
    ///
    /// Buckets come from the deltas of the session counter between consecutive
    /// samples, so only seconds that were actually observed are returned — an
    /// empty array means "nothing measured", never a row of fabricated zeros.
    /// A negative delta means the counter reset (model reload) and is dropped.
    static func tokenBuckets(_ history: [LiveMetrics], seconds: Int = AnalyticsThresholds.bucketSeconds) -> [TokenBucket] {
        guard history.count >= 2 else { return [] }
        let window = max(2, min(seconds, 600))
        let lastSecond = Int(history[history.count - 1].ts.rounded(.down))
        let firstSecond = max(Int(history[0].ts.rounded(.down)), lastSecond - window + 1)
        guard lastSecond >= firstSecond else { return [] }

        var counts = [Int](repeating: 0, count: lastSecond - firstSecond + 1)
        var sawTokens = false
        for index in 0..<(history.count - 1) {
            let delta = history[index + 1].tokensGenerated - history[index].tokensGenerated
            guard delta > 0 else { continue }        // 0 = idle second, < 0 = counter reset
            let bucket = Int(history[index + 1].ts.rounded(.down)) - firstSecond
            if bucket >= 0 && bucket < counts.count {
                counts[bucket] += delta
                sawTokens = true
            }
        }
        guard sawTokens else { return [] }

        return counts.enumerated().map { offset, tokens in
            TokenBucket(id: offset, startTs: Double(firstSecond + offset), tokens: tokens)
        }
    }

    // MARK: Session aggregates (requests arrive newest first)

    /// Mean/median/max decode rate, TTFT and token totals over the session.
    ///
    /// `count` covers every recorded request; the rate aggregates cover only
    /// requests that carry a measurement, so a failed or unmeasured request
    /// cannot drag a mean toward zero.
    static func sessionStats(_ requests: [RequestRecord]) -> SessionStats {
        guard !requests.isEmpty else { return .empty }
        let decodeValues = requests.map(\.decodeTps).filter { $0 > 0 }
        let prefillValues = requests.map(\.prefillTps).filter { $0 > 0 }
        let ttftValues = requests.map(\.ttftMs).filter { $0 > 0 }
        let lastTtft = requests.first { $0.ttftMs > 0 }?.ttftMs ?? 0
        let generated = requests.reduce(0) { $0 + max(0, $1.genTokens) }
        let peak = requests.map(\.peakMemBytes).max() ?? 0
        return SessionStats(
            count: requests.count,
            measuredCount: decodeValues.count,
            meanDecodeTps: mean(decodeValues),
            medianDecodeTps: median(decodeValues),
            maxDecodeTps: decodeValues.max() ?? 0,
            meanTtftMs: mean(ttftValues),
            lastTtftMs: lastTtft,
            totalGenTokens: generated,
            peakMemBytes: peak,
            meanPrefillTps: mean(prefillValues)
        )
    }

    /// Current decode rate against the session median decode rate.
    ///
    /// `nil` — never a made-up baseline — when the live sample has no rate, when
    /// fewer than `deltaRequests` requests were measured, or when the median is 0.
    static func deltaVsMedian(currentTps: Double, requests: [RequestRecord]) -> Double? {
        guard currentTps > 0 else { return nil }
        let stats = sessionStats(requests)
        guard stats.measuredCount >= AnalyticsThresholds.deltaRequests, stats.medianDecodeTps > 0 else { return nil }
        return (currentTps - stats.medianDecodeTps) / stats.medianDecodeTps
    }

    /// `+12.3%` / `-4.0%` for a `deltaVsMedian` fraction.
    static func deltaText(_ fraction: Double) -> String {
        String(format: "%+.1f%%", fraction * 100)
    }

    // MARK: Memory

    /// Real headroom from the machine's own numbers: how much memory is free.
    static func systemHeadroom(used: Int, total: Int) -> MemoryHeadroom {
        guard total > 0 else { return .empty }
        let clampedUsed = min(total, max(0, used))
        let free = total - clampedUsed
        return MemoryHeadroom(
            freeBytes: free,
            usedBytes: clampedUsed,
            totalBytes: total,
            fraction: min(1, max(0, Double(free) / Double(total)))
        )
    }

    // MARK: Runtime split

    /// Wall-clock split of the most recent request: prefill from `prefillTps`,
    /// decode from `decodeTps`, load from the live telemetry, and whatever is
    /// left of `totalMs` as overhead. `nil` without a finished request.
    static func runtimeSplit(for record: RequestRecord?, loadMs: Double) -> RuntimeSplit? {
        guard let record, record.totalMs > 0 else { return nil }
        let total = record.totalMs
        let prefill = record.prefillTps > 0 ? Double(record.promptTokens) / record.prefillTps * 1000 : 0
        let decode = record.decodeTps > 0 ? Double(max(0, record.genTokens - 1)) / record.decodeTps * 1000 : 0
        let prefillMs = min(total, max(0, prefill))
        let decodeMs = min(total - prefillMs, max(0, decode))
        let overheadMs = max(0, total - prefillMs - decodeMs)
        let load = max(0, loadMs)

        // `loadMs` is the model load, which happens outside the request's own
        // `totalMs`, so its bar is capped at a full width rather than scaling past it.
        let slices = [
            RuntimeSlice(id: "prefill", label: "Prefill", ms: prefillMs,
                         share: min(1, prefillMs / total), measured: record.prefillTps > 0),
            RuntimeSlice(id: "decode", label: "Decode", ms: decodeMs,
                         share: min(1, decodeMs / total), measured: record.decodeTps > 0),
            RuntimeSlice(id: "load", label: "Load", ms: load,
                         share: min(1, load / total), measured: load > 0),
            RuntimeSlice(id: "overhead", label: "Overhead", ms: overheadMs,
                         share: min(1, overheadMs / total),
                         measured: prefillMs > 0 || decodeMs > 0),
        ]
        return RuntimeSplit(
            request: record.request,
            totalMs: total,
            loadMs: load,
            prefillMs: prefillMs,
            decodeMs: decodeMs,
            overheadMs: overheadMs,
            slices: slices
        )
    }

    // MARK: Small math

    static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let middle = sorted.count / 2
        if sorted.count % 2 == 1 { return sorted[middle] }
        return (sorted[middle - 1] + sorted[middle]) / 2
    }
}

// MARK: - Chart primitives
//
// Every mark sizes itself from the container geometry it is handed, so a card
// reads correctly at any panel width.

/// Bar with rounded top corners only — the reference board's tall bar mark.
struct RoundedTopBar: Shape {
    var radius: CGFloat = 4

    func path(in rect: CGRect) -> Path {
        let r = min(max(0, radius), min(rect.width, rect.height) / 2)
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY))
        path.addLine(to: CGPoint(x: rect.minX, y: rect.minY + r))
        path.addQuadCurve(to: CGPoint(x: rect.minX + r, y: rect.minY),
                          control: CGPoint(x: rect.minX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX - r, y: rect.minY))
        path.addQuadCurve(to: CGPoint(x: rect.maxX, y: rect.minY + r),
                          control: CGPoint(x: rect.maxX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
        path.closeSubpath()
        return path
    }
}

/// One tall rounded bar. `fraction` is 0…1 of the container height.
struct TallBarMark: View {
    let fraction: Double
    var fill: Color = Paper.accent
    var radius: CGFloat = 3
    var minHeight: CGFloat = 2
    var emphasized: Bool = false

    var body: some View {
        GeometryReader { proxy in
            let height = max(minHeight, CGFloat(clampUnit(fraction)) * proxy.size.height)
            RoundedTopBar(radius: radius)
                .fill(fill)
                .frame(width: proxy.size.width, height: height)
                .frame(maxHeight: .infinity, alignment: .bottom)
                .opacity(emphasized ? 1 : 0.9)
        }
    }
}

/// Two-tone bar: a deep tone at the base under a lighter tone, as on the
/// reference board's orange card. `deepFraction` is the deep portion of the
/// bar's own height (0…1).
struct StackedBarMark: View {
    let fraction: Double
    var deepFraction: Double = 0
    var deep: Color = Paper.clayDeep
    var light: Color = Paper.clay
    var radius: CGFloat = 3
    var minHeight: CGFloat = 2

    var body: some View {
        GeometryReader { proxy in
            let height = max(minHeight, CGFloat(clampUnit(fraction)) * proxy.size.height)
            let deepHeight = min(height, max(0, CGFloat(clampUnit(deepFraction)) * height))
            ZStack(alignment: .bottom) {
                RoundedTopBar(radius: radius).fill(light)
                Rectangle().fill(deep).frame(width: proxy.size.width, height: deepHeight)
            }
            .frame(width: proxy.size.width, height: height)
            .frame(maxHeight: .infinity, alignment: .bottom)
            .clipShape(RoundedTopBar(radius: radius).size(width: proxy.size.width, height: height))
        }
    }
}

/// Fan of thin radial ticks over a 200° arc: dark up to the filled fraction,
/// faint beyond it, with an end-cap marker at the boundary.
struct TickFanGauge: View {
    let fraction: Double
    var tickCount: Int = 40
    var sweep: Double = 200
    var startAngle: Double = 170
    var ink: Color = Paper.butterInk
    var track: Color = Paper.butterInk.opacity(0.22)
    /// Marker colour; defaults to `ink`.
    var cap: Color?
    var tickWidth: CGFloat = 1.6
    var innerRatio: CGFloat = 0.68
    var capSize: CGFloat = 5

    var body: some View {
        GeometryReader { proxy in
            let size = proxy.size
            let angles = tickAngles
            let radians = angles.map { $0 * .pi / 180 }
            let sinMin = radians.map { sin($0) }.min() ?? -1
            let sinMax = radians.map { sin($0) }.max() ?? 1
            let cosMin = radians.map { cos($0) }.min() ?? -1
            let cosMax = radians.map { cos($0) }.max() ?? 1
            let outer = min(size.width / max(0.001, cosMax - cosMin),
                            size.height / max(0.001, sinMax - sinMin)) * 0.94
            let inner = outer * min(0.95, max(0, innerRatio))
            let center = CGPoint(
                x: size.width / 2 - CGFloat((cosMin + cosMax) / 2) * outer,
                y: size.height / 2 - CGFloat((sinMin + sinMax) / 2) * outer
            )
            let unit = clampUnit(fraction)
            let filled = Int((Double(tickCount) * unit).rounded())
            let capAngle = startAngle + sweep * unit

            ZStack {
                tickLine(from: 0, to: tickCount, center: center, inner: inner, outer: outer, angles: angles)
                    .stroke(track, style: StrokeStyle(lineWidth: tickWidth, lineCap: .round))
                tickLine(from: 0, to: min(filled, tickCount), center: center, inner: inner, outer: outer, angles: angles)
                    .stroke(ink, style: StrokeStyle(lineWidth: tickWidth, lineCap: .round))
                if unit > 0 {
                    Circle()
                        .fill(cap ?? ink)
                        .frame(width: capSize, height: capSize)
                        .position(point(at: capAngle, radius: (inner + outer) / 2, center: center))
                }
            }
            .frame(width: size.width, height: size.height)
        }
    }

    private var tickAngles: [Double] {
        let count = max(2, tickCount)
        return (0..<count).map { startAngle + sweep * Double($0) / Double(count - 1) }
    }

    private func tickLine(from: Int, to: Int, center: CGPoint, inner: CGFloat, outer: CGFloat, angles: [Double]) -> Path {
        var path = Path()
        guard to > from else { return path }
        for index in from..<min(to, angles.count) {
            let angle = angles[index]
            path.move(to: point(at: angle, radius: inner, center: center))
            path.addLine(to: point(at: angle, radius: outer, center: center))
        }
        return path
    }

    private func point(at degrees: Double, radius: CGFloat, center: CGPoint) -> CGPoint {
        let radians = degrees * .pi / 180
        return CGPoint(x: center.x + CGFloat(cos(radians)) * radius,
                       y: center.y + CGFloat(sin(radians)) * radius)
    }
}

/// Smooth polyline with a soft fill underneath and small circular node markers
/// at sampled points. Scaled from `baseline` (0 by default) so the line's height
/// reads as a rate, not as a shape.
struct LineAreaMark: View {
    let values: [Double]
    var line: Color = Paper.olive
    var fill: Color = Paper.olive.opacity(0.18)
    var node: Color? = Paper.olive
    var lineWidth: CGFloat = 1.7
    var nodeSize: CGFloat = 4
    var nodes: Int = 5
    var baseline: Double = 0

    var body: some View {
        GeometryReader { proxy in
            let points = points(in: proxy.size)
            ZStack {
                if points.count >= 2 {
                    areaPath(points, in: proxy.size).fill(fill)
                    smoothPath(points)
                        .stroke(line, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round, lineJoin: .round))
                    if let node {
                        ForEach(markerIndices, id: \.self) { index in
                            Circle()
                                .fill(node)
                                .frame(width: nodeSize, height: nodeSize)
                                .position(points[index])
                        }
                    }
                }
            }
            .clipShape(Rectangle())
        }
    }

    private var markerIndices: [Int] {
        guard values.count >= 2 else { return [] }
        let wanted = max(2, min(nodes, values.count))
        if wanted >= values.count { return Array(values.indices) }
        return (0..<wanted).map { step in
            Int((Double(step) * Double(values.count - 1) / Double(wanted - 1)).rounded())
        }
    }

    private func points(in size: CGSize) -> [CGPoint] {
        guard values.count >= 2, size.width > 1, size.height > 1 else { return [] }
        let inset = (lineWidth + nodeSize) / 2 + 1
        let top = inset
        let bottom = max(top + 1, size.height - inset)
        let usable = bottom - top
        let high = values.max() ?? 0
        let low = min(baseline, high)
        let span = high - low

        var result: [CGPoint] = []
        result.reserveCapacity(values.count)
        for index in values.indices {
            // The same inset is applied on x, otherwise the first and last node
            // markers sit on the edge and `clipShape` cuts them in half.
            let x = inset + (size.width - 2 * inset) * CGFloat(index) / CGFloat(values.count - 1)
            let y: CGFloat = span <= 0 ? bottom : bottom - CGFloat((values[index] - low) / span) * usable
            result.append(CGPoint(x: x, y: min(bottom, max(top, y))))
        }
        return result
    }

    /// Monotone cubic Hermite (Fritsch–Carlson), which cannot overshoot.
    ///
    /// Catmull–Rom was drawing values that were never measured on this app's
    /// spiky series: on a run whose measured peak was 83.5 tok/s the curve
    /// reached 87.9 and dipped to −6.1, i.e. below zero. The tangent limiter
    /// keeps every point of the curve inside the range of its two endpoints.
    private func smoothPath(_ points: [CGPoint]) -> Path {
        var path = Path()
        guard points.count >= 2 else { return path }
        let count = points.count

        var slopes = [CGFloat](repeating: 0, count: count - 1)
        for index in 0..<(count - 1) {
            let dx = points[index + 1].x - points[index].x
            slopes[index] = dx == 0 ? 0 : (points[index + 1].y - points[index].y) / dx
        }

        var tangents = [CGFloat](repeating: 0, count: count)
        tangents[0] = slopes[0]
        tangents[count - 1] = slopes[count - 2]
        for index in 1..<(count - 1) {
            if slopes[index - 1] * slopes[index] <= 0 {
                tangents[index] = 0          // local extremum: a flat tangent cannot overshoot
            } else {
                let average = (slopes[index - 1] + slopes[index]) / 2
                let limit = 3 * min(abs(slopes[index - 1]), abs(slopes[index]))
                tangents[index] = abs(average) > limit
                    ? (average < 0 ? -limit : limit)
                    : average
            }
        }

        path.move(to: points[0])
        for index in 0..<(count - 1) {
            let dx = (points[index + 1].x - points[index].x) / 3
            let start = CGPoint(x: points[index].x + dx, y: points[index].y + tangents[index] * dx)
            let end = CGPoint(x: points[index + 1].x - dx, y: points[index + 1].y - tangents[index + 1] * dx)
            path.addCurve(to: points[index + 1], control1: start, control2: end)
        }
        return path
    }

    private func areaPath(_ points: [CGPoint], in size: CGSize) -> Path {
        var path = smoothPath(points)
        guard let first = points.first, let last = points.last else { return path }
        path.addLine(to: CGPoint(x: last.x, y: size.height))
        path.addLine(to: CGPoint(x: first.x, y: size.height))
        path.closeSubpath()
        return path
    }
}

/// Faint horizontal rules behind a chart.
struct GridLines: View {
    var count: Int = 4
    var color: Color = Paper.hairline

    var body: some View {
        let lines = max(2, count)
        VStack(spacing: 0) {
            ForEach(0..<lines, id: \.self) { index in
                Rectangle().fill(color).frame(height: 1)
                if index < lines - 1 { Spacer(minLength: 0) }
            }
        }
    }
}

/// Dark-surface twin of `EmptyState`: the frozen primitive hardcodes the paper
/// palette's ink, which is unreadable on the obsidian cards.
struct CardEmptyState: View {
    let symbol: String
    let title: String
    var detail: String?
    var ink: Color = .white.opacity(0.78)
    var faint: Color = .white.opacity(0.42)

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(faint)
            Text(title)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(ink)
            if let detail {
                Text(detail)
                    .font(PaperFont.meta)
                    .foregroundStyle(faint)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
    }
}

/// Capsule button used by the analytics composer.
struct PaperButton: View {
    enum Style { case primary, ghost }

    let title: String
    var symbol: String?
    var style: Style = .primary
    var enabled: Bool = true
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                if let symbol {
                    Image(systemName: symbol).font(.system(size: 9, weight: .semibold))
                }
                Text(title).font(.system(size: 11, weight: .semibold))
            }
            .foregroundStyle(style == .primary ? Paper.accentInk : Paper.ink.opacity(0.8))
            .padding(.horizontal, 11)
            .padding(.vertical, 6)
            .background(Capsule().fill(style == .primary ? Paper.accent : Paper.card))
            .overlay(Capsule().strokeBorder(style == .primary ? .clear : Paper.stroke, lineWidth: 1))
            .opacity(enabled ? 1 : 0.34)
        }
        .buttonStyle(.plain)
        .disabled(!enabled)
    }
}

/// Clamp helper shared by the marks above.
private func clampUnit(_ value: Double) -> Double {
    if value.isNaN { return 0 }
    return min(1, max(0, value))
}

// MARK: - Snapshot rendering
//
// `--snapshot` / `--snapshot-analytics` draw the panel offscreen with
// `ImageRenderer`, which cannot draw AppKit-backed controls: a `ScrollView` or
// `TextField` in the tree makes the whole render come out blank. The picker
// surface sets this flag while it renders a snapshot, so the analytics board can
// swap those two controls for plain SwiftUI equivalents without touching the
// live panel, which never sees the flag.

private struct RenderStaticSnapshotKey: EnvironmentKey {
    static let defaultValue = false
}

extension EnvironmentValues {
    /// True only while the panel is being rasterised offscreen.
    var renderStaticSnapshot: Bool {
        get { self[RenderStaticSnapshotKey.self] }
        set { self[RenderStaticSnapshotKey.self] = newValue }
    }
}
