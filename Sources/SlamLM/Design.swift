import SwiftUI

// MARK: - Palette
//
// Warm paper base with white cards, a dark forest accent and muted tag tints
// (picker surface), plus the saturated card set used by the analytics surface.

enum Paper {
    // Surfaces
    static let bg = Color(hex: 0xF1EDE4)
    static let card = Color(hex: 0xFCFAF6)
    static let cardSunken = Color(hex: 0xEAE5DA)
    static let stroke = Color(hex: 0xE1DACB)
    static let hairline = Color(hex: 0xEDE7DA)

    // Ink
    static let ink = Color(hex: 0x1C1B17)
    static let inkSoft = Color(hex: 0x6E695E)
    static let inkFaint = Color(hex: 0x9C968A)

    // Accents
    static let accent = Color(hex: 0x2A3423)      // selected chip, primary button
    static let accentInk = Color(hex: 0xF6F4EC)   // text on accent
    static let running = Color(hex: 0x3F8A44)
    static let track = Color(hex: 0xE5DFD1)

    /// Text fields. Deliberately darker than the panel and than a card, and
    /// paired with dark ink, so typed text and the placeholder both clear 4.5:1
    /// instead of the washed-out grey the system placeholder tint gave.
    static let field = Color(hex: 0xDCD5C4)
    static let fieldInk = Color(hex: 0x2E2A24)
    static let fieldPlaceholder = Color(hex: 0x5E584C)

    // Analytics surfaces. The expanded board is built from the same material as
    // the panel — paper cards, one deep-forest card for contrast, and the tag
    // tints used as washes — because the saturated board colours that came from
    // the metric-card reference read as a different product at panel width.
    static let washOlive = Color(hex: 0xE9EEDA)
    static let washClay = Color(hex: 0xF0DACE)
    static let washButter = Color(hex: 0xF2EAC9)
    /// Marks: strong enough to read on a wash or on the deep card.
    static let olive = Color(hex: 0x6F7F53)
    static let oliveDeep = Color(hex: 0x3F4A2E)
    static let clay = Color(hex: 0xC97A55)
    static let clayDeep = Color(hex: 0x8C4A2E)

    // Ink for text on the washes: dark, so the small labels stay readable.
    static let clayInk = Color(hex: 0x3A2A20)
    static let butterInk = Color(hex: 0x45402C)

    /// Tag tints, keyed by the category strings the bridge emits.
    static func tagFill(_ category: String) -> Color {
        switch category {
        case "Chat": return Color(hex: 0xDCE7F6)
        case "Instruct": return Color(hex: 0xDFEBDC)
        case "Code": return Color(hex: 0xF7DEDE)
        case "Vision": return Color(hex: 0xE6DEF5)
        case "Embedding": return Color(hex: 0xE3E9E4)
        case "Audio": return Color(hex: 0xF6E7CF)
        case "Multilingual": return Color(hex: 0xF2EAC9)
        case "Reasoning": return Color(hex: 0xEADFF3)
        case "General": return Color(hex: 0xEFE9D8)
        default: return Color(hex: 0xE9E5DA)
        }
    }

    static func tagInk(_ category: String) -> Color {
        switch category {
        case "Chat": return Color(hex: 0x2C4A75)
        case "Instruct": return Color(hex: 0x2F5A33)
        case "Code": return Color(hex: 0x7A2F2F)
        case "Vision": return Color(hex: 0x4B3576)
        case "Embedding": return Color(hex: 0x36443A)
        case "Audio": return Color(hex: 0x7A5320)
        case "Multilingual": return Color(hex: 0x6E5C1C)
        case "Reasoning": return Color(hex: 0x53356E)
        case "General": return Color(hex: 0x5F5637)
        default: return inkSoft
        }
    }

    /// Icon tile tint per model family, derived from the architecture string.
    static func familyTint(_ architecture: String) -> Color {
        switch architecture.lowercased() {
        case let a where a.contains("llama"): return Color(hex: 0xE7DCC6)
        case let a where a.contains("qwen"): return Color(hex: 0xE4E2DD)
        case let a where a.contains("mistral"): return Color(hex: 0xF6DCCB)
        case let a where a.contains("deepseek"): return Color(hex: 0xDED9EA)
        case let a where a.contains("gemma"): return Color(hex: 0xD8E3F2)
        case let a where a.contains("phi"): return Color(hex: 0xE2E6DA)
        case let a where a.contains("bert") || a.contains("embed"): return Color(hex: 0xDCE6DE)
        default: return Color(hex: 0xE7E3D9)
        }
    }

    static func familyGlyph(_ architecture: String) -> String {
        switch architecture.lowercased() {
        case let a where a.contains("llama"): return "hare"
        case let a where a.contains("qwen"): return "circle.hexagongrid"
        case let a where a.contains("mistral"): return "wind"
        case let a where a.contains("deepseek"): return "brain.head.profile"
        case let a where a.contains("gemma"): return "diamond"
        case let a where a.contains("phi"): return "p.square"
        case let a where a.contains("bert") || a.contains("embed"): return "square.grid.3x3"
        default: return "cube"
        }
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255.0,
            green: Double((hex >> 8) & 0xFF) / 255.0,
            blue: Double(hex & 0xFF) / 255.0,
            opacity: 1.0
        )
    }
}

// MARK: - Type scale

enum PaperFont {
    /// Large numerals on metric cards.
    static func numeral(_ size: CGFloat) -> Font {
        .system(size: size, weight: .semibold, design: .rounded)
    }
    static func numeralLight(_ size: CGFloat) -> Font {
        .system(size: size, weight: .regular, design: .rounded)
    }
    static let panelTitle = Font.system(size: 19, weight: .bold)
    static let rowTitle = Font.system(size: 12.5, weight: .semibold)
    static let body = Font.system(size: 12)
    static let meta = Font.system(size: 10.5)
    static let cardTitle = Font.system(size: 10.5, weight: .semibold)
    static let micro = Font.system(size: 9, weight: .semibold)
}

/// Letterspaced uppercase micro-label, the reference board's label style.
struct SectionLabel: View {
    let text: String
    var color: Color = Paper.inkSoft
    var size: CGFloat = 9

    var body: some View {
        Text(text.uppercased())
            .font(.system(size: size, weight: .semibold))
            .tracking(0.9)
            .foregroundStyle(color)
    }
}

// MARK: - Panel geometry
//
// The analytics tab expands the panel, per the reference board's "opens up and
// expands" behaviour.

enum PanelLayout {
    static let corner: CGFloat = 16
    static let gutter: CGFloat = 12

    static func size(for tab: PanelTab) -> CGSize {
        switch tab {
        case .models: return CGSize(width: 392, height: 566)
        case .analytics: return CGSize(width: 648, height: 648)
        }
    }

    static func bodyHeight(for tab: PanelTab) -> CGFloat {
        size(for: tab).height - 132
    }
}

// MARK: - Formatting
//
// One place for every number the UI shows, so both surfaces agree.

enum PaperFormat {
    /// Memory sizes, binary like Activity Monitor's memory tab: 17179869184
    /// prints as `16 GB`. One base for every memory figure is what makes a
    /// gauge's numbers add up — used + free = total — and Activity Monitor's own
    /// three figures only reconcile in GiB.
    static func bytes(_ value: Int) -> String {
        let gb = Double(max(0, value)) / 1_073_741_824
        if gb >= 100 { return String(format: "%.0f GB", gb) }
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(max(0, value)) / 1_048_576
        if mb >= 1 { return String(format: "%.0f MB", mb) }
        return "0 MB"
    }

    /// `used / total` sharing one unit, for the bars: `12.7/16 GB`.
    static func bytesPair(_ used: Int, _ total: Int) -> String {
        let totalGb = Double(max(0, total)) / 1_073_741_824
        let usedGb = Double(max(0, used)) / 1_073_741_824
        if usedGb < 1 {
            let mb = Double(max(0, used)) / 1_048_576
            return String(format: "%.0f MB", mb) + String(format: "/%.0f GB", totalGb)
        }
        return String(format: "%.1f/%.0f GB", usedGb, totalGb)
    }

    static func tps(_ value: Double) -> String {
        value <= 0 ? "—" : String(format: "%.1f", value)
    }

    static func ms(_ value: Double) -> String {
        value <= 0 ? "—" : String(format: "%.0f", value)
    }

    static func tokens(_ value: Int) -> String {
        if value >= 1_000_000 { return String(format: "%.1fM", Double(value) / 1_000_000) }
        if value >= 10_000 { return String(format: "%.0fk", Double(value) / 1_000) }
        if value >= 1_000 { return String(format: "%.1fk", Double(value) / 1_000) }
        return "\(value)"
    }

    static func percent(_ fraction: Double) -> String {
        String(format: "%.1f%%", fraction * 100)
    }

    static func relative(_ epoch: Double) -> String {
        guard epoch > 0 else { return "" }
        let delta = Date().timeIntervalSince1970 - epoch
        if delta < 60 { return "just now" }
        if delta < 3600 { return "\(Int(delta / 60))m ago" }
        if delta < 86_400 { return "\(Int(delta / 3600))h ago" }
        return "\(Int(delta / 86_400))d ago"
    }
}

// MARK: - Chrome

/// Rounded card with the paper-style fill, hairline border and shallow shadow.
struct PaperCard<Content: View>: View {
    var fill: Color = Paper.card
    var radius: CGFloat = 14
    var padding: CGFloat = 12
    var stroke: Color? = Paper.stroke
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .fill(fill)
            )
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .strokeBorder(stroke ?? .clear, lineWidth: 1)
            )
            .shadow(color: Paper.ink.opacity(0.05), radius: 6, x: 0, y: 2)
    }
}

struct PaperRule: View {
    var color: Color = Paper.hairline
    var body: some View {
        Rectangle().fill(color).frame(height: 1)
    }
}

/// Category chip used by the model filter row.
struct CategoryChip: View {
    let title: String
    let selected: Bool
    var action: () -> Void = {}

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.system(size: 11.5, weight: selected ? .semibold : .regular))
                .foregroundStyle(selected ? Paper.accentInk : Paper.ink.opacity(0.75))
                .padding(.horizontal, 11)
                .padding(.vertical, 5)
                .background(
                    Capsule().fill(selected ? Paper.accent : Paper.card)
                )
                .overlay(
                    Capsule().strokeBorder(selected ? .clear : Paper.stroke, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
}

/// Small label pill, e.g. `Chat`, `Instruct`, `4-bit`.
struct TagPill: View {
    let text: String
    var fill: Color = Paper.cardSunken
    var ink: Color = Paper.inkSoft

    var body: some View {
        Text(text)
            .font(.system(size: 9.5, weight: .medium))
            .foregroundStyle(ink)
            .padding(.horizontal, 6)
            .padding(.vertical, 2.5)
            .background(RoundedRectangle(cornerRadius: 5, style: .continuous).fill(fill))
    }
}

/// Numeric delta badge, e.g. `+0.2%` on the reference board.
struct DeltaPill: View {
    let text: String
    var positive: Bool = true

    var body: some View {
        Text(text)
            .font(.system(size: 9.5, weight: .semibold))
            .foregroundStyle(positive ? Color(hex: 0x1F2A14) : Color(hex: 0x5A1E14))
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Capsule().fill(positive ? Paper.washOlive : Color(hex: 0xF3D3C7)))
    }
}

/// Rounded icon tile, the model row's leading glyph.
struct GlyphTile: View {
    let symbol: String
    var tint: Color = Paper.cardSunken
    var size: CGFloat = 40
    var glyphSize: CGFloat = 19
    var ink: Color = Paper.ink.opacity(0.72)

    var body: some View {
        RoundedRectangle(cornerRadius: 11, style: .continuous)
            .fill(tint)
            .frame(width: size, height: size)
            .overlay(
                Image(systemName: symbol)
                    .font(.system(size: glyphSize, weight: .medium))
                    .foregroundStyle(ink)
            )
    }
}

/// Circular icon button used for settings, reveal-in-Finder, play and stop.
struct GlyphButton: View {
    let symbol: String
    var diameter: CGFloat = 26
    var glyphSize: CGFloat = 11.5
    var fill: Color = Paper.cardSunken
    var ink: Color = Paper.ink.opacity(0.8)
    /// Spoken and asserted label. A bare SF Symbol name is not one, so callers
    /// pass what the button actually does.
    var label: String = ""
    var action: () -> Void = {}
    @State private var hovering = false

    var body: some View {
        Button(action: action) {
            ZStack {
                Circle().fill(fill)
                Circle().strokeBorder(Paper.stroke, lineWidth: fill == Paper.cardSunken ? 1 : 0)
                Image(systemName: symbol)
                    .font(.system(size: glyphSize, weight: .semibold))
                    .foregroundStyle(ink)
            }
            .frame(width: diameter, height: diameter)
            .opacity(hovering ? 0.82 : 1)
        }
        .buttonStyle(.plain)
        .onHover { hovering = $0 }
        .accessibilityLabel(Text(label.isEmpty ? symbol : label))
        .help(label)
    }
}

/// Horizontal usage bar: filled portion over a soft track.
struct StatBar: View {
    let fraction: Double
    var fill: Color = Paper.running
    var track: Color = Paper.track
    var height: CGFloat = 5

    var body: some View {
        GeometryReader { proxy in
            ZStack(alignment: .leading) {
                Capsule().fill(track)
                Capsule()
                    .fill(fill)
                    .frame(width: max(0, min(1, fraction)) * proxy.size.width)
            }
        }
        .frame(height: height)
    }
}

struct StatusDot: View {
    var color: Color = Paper.running
    var diameter: CGFloat = 6
    var body: some View {
        Circle().fill(color).frame(width: diameter, height: diameter)
    }
}

/// `⌘K`-style key hint shown inside the search field.
struct KeyHint: View {
    let text: String
    var body: some View {
        Text(text)
            .font(.system(size: 10, weight: .medium))
            .foregroundStyle(Paper.inkFaint)
            .padding(.horizontal, 5)
            .padding(.vertical, 2)
            .background(RoundedRectangle(cornerRadius: 4).fill(Paper.card))
            .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(Paper.stroke, lineWidth: 1))
    }
}

/// Honest empty state: no fabricated zeros, just what is missing.
struct EmptyState: View {
    let symbol: String
    let title: String
    var detail: String?

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: symbol)
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(Paper.inkFaint)
            Text(title)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(Paper.inkSoft)
            if let detail {
                Text(detail)
                    .font(PaperFont.meta)
                    .foregroundStyle(Paper.inkFaint)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 18)
    }
}
