import SwiftUI

/// One catalog row: family glyph, name, meta line, category tags and the
/// play/stop control for the loaded model. Rows are separated by `PaperRule`.
struct ModelRow: View {
    let model: ModelInfo
    /// This is the model the bridge reports as loaded.
    let loaded: Bool
    /// A model command is in flight; the toggle is inert until it lands.
    let busy: Bool
    let isLast: Bool
    let onToggle: () -> Void
    let onSelect: () -> Void

    @State private var hovering = false

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                GlyphTile(
                    symbol: Paper.familyGlyph(model.architecture),
                    tint: Paper.familyTint(model.architecture),
                    size: 40,
                    glyphSize: 19
                )
                VStack(alignment: .leading, spacing: 3) {
                    Text(model.name)
                        .font(PaperFont.rowTitle)
                        .foregroundStyle(Paper.ink)
                        .lineLimit(1)
                    Text(meta)
                        .font(PaperFont.meta)
                        .foregroundStyle(Paper.inkSoft)
                        .lineLimit(1)
                    if !model.categories.isEmpty { tags }
                }
                Spacer(minLength: 6)
                if loaded {
                    HStack(spacing: 4) {
                        StatusDot()
                        Text("Running")
                            .font(PaperFont.meta)
                            .foregroundStyle(Paper.inkSoft)
                    }
                }
                GlyphButton(
                    symbol: loaded ? "stop.fill" : "play.fill",
                    diameter: 24,
                    glyphSize: 9.5,
                    fill: loaded ? Paper.cardSunken : Paper.accent,
                    ink: loaded ? Paper.ink.opacity(0.8) : Paper.accentInk,
                    label: loaded ? "Stop \(model.name)" : "Run \(model.name)",
                    action: onToggle
                )
                .opacity(busy ? 0.45 : 1)
                Image(systemName: "chevron.right")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(Paper.inkFaint)
            }
            .padding(.horizontal, PanelLayout.gutter)
            .padding(.vertical, 8)
            .background(highlight)
            .contentShape(Rectangle())
            .onTapGesture(perform: onSelect)
            .onHover { hovering = $0 }

            if !isLast {
                PaperRule().padding(.leading, PanelLayout.gutter + 50)
            }
        }
    }

    /// `params · quant · size`, skipping whatever the config did not provide.
    private var meta: String {
        [model.params, model.quant, PaperFormat.bytes(model.bytes)]
            .filter { !$0.isEmpty }
            .joined(separator: " · ")
    }

    private var tags: some View {
        HStack(spacing: 4) {
            ForEach(Array(model.categories.prefix(3)), id: \.self) { category in
                TagPill(text: category, fill: Paper.tagFill(category), ink: Paper.tagInk(category))
            }
            if model.categories.count > 3 {
                TagPill(text: "+\(model.categories.count - 3)")
            }
        }
    }

    private var highlight: some View {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
            .fill(loaded ? Paper.cardSunken.opacity(0.75) : (hovering ? Paper.card.opacity(0.9) : .clear))
            .padding(.horizontal, PanelLayout.gutter - 6)
    }
}
