import SwiftUI

/// What the model did with its tools before answering.
///
/// A tool call is never invisible: each one shows what was asked, whether it
/// worked, and what came back. This is the panel's evidence that an answer which
/// cites the web or a file actually looked at one.
struct ToolTrace: View {
    let events: [ToolEvent]

    /// Calls and results, paired by round, in the order they happened.
    private var entries: [Entry] {
        var calls: [Int: ToolEvent] = [:]
        var built: [Entry] = []
        for event in events {
            if event.isCall {
                calls[event.round] = event
            } else if let call = calls[event.round], call.name == event.name {
                built.append(Entry(call: call, result: event))
            } else {
                built.append(Entry(call: event, result: nil))
            }
        }
        // A call whose result never arrived (a cancel mid-tool) still shows.
        let answered = Set(built.map(\.call.id))
        for call in events where call.isCall && !answered.contains(call.id) {
            built.append(Entry(call: call, result: nil))
        }
        return built
    }

    struct Entry: Identifiable {
        let call: ToolEvent
        let result: ToolEvent?
        var id: String { call.id }
    }

    var body: some View {
        if !events.isEmpty {
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 6) {
                    SectionLabel(text: "Tool calls · \(entries.count)")
                    Spacer(minLength: 4)
                    if entries.contains(where: { $0.result?.ok == false }) {
                        TagPill(text: "some failed", fill: Paper.washClay, ink: Paper.clayInk)
                    }
                }
                ForEach(entries) { entry in
                    row(entry)
                }
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 7)
            .background(RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Paper.washOlive))
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(Paper.olive.opacity(0.25), lineWidth: 1)
            )
        }
    }

    private func row(_ entry: Entry) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: Self.symbol(for: entry.call.name))
                .font(.system(size: 9, weight: .semibold))
                .foregroundStyle(Paper.oliveDeep)
                .frame(width: 12)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 5) {
                    Text(entry.call.name)
                        .font(.system(size: 10.5, weight: .semibold, design: .monospaced))
                        .foregroundStyle(Paper.ink)
                    Text(argumentSummary(entry.call))
                        .font(.system(size: 10.5))
                        .foregroundStyle(Paper.inkSoft)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer(minLength: 4)
                    outcome(entry)
                }
                if let detail = entry.result?.detail, !detail.isEmpty {
                    Text(detail)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Paper.inkSoft)
                        .lineLimit(3)
                        .truncationMode(.tail)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                } else if let result = entry.result {
                    Text(result.summary)
                        .font(PaperFont.meta)
                        .foregroundStyle(Paper.inkFaint)
                        .lineLimit(1)
                }
            }
        }
    }

    @ViewBuilder private func outcome(_ entry: Entry) -> some View {
        if let result = entry.result {
            HStack(spacing: 4) {
                StatusDot(color: result.ok ? Paper.running : Paper.clayDeep, diameter: 5)
                Text(result.ok ? result.summary : "failed")
                    .font(.system(size: 9.5, weight: .medium))
                    .foregroundStyle(result.ok ? Paper.inkSoft : Paper.clayDeep)
                    .lineLimit(1)
            }
        } else {
            Text("running…")
                .font(.system(size: 9.5, weight: .medium))
                .foregroundStyle(Paper.inkFaint)
        }
    }

    private func argumentSummary(_ event: ToolEvent) -> String {
        guard let arguments = event.arguments, !arguments.pairs.isEmpty else { return "" }
        return "(" + arguments.summary + ")"
    }

    private static func symbol(for tool: String) -> String {
        switch tool {
        case "web_search": return "globe"
        case "read_file": return "doc.text"
        case "list_directory": return "folder"
        case "search_files": return "magnifyingglass"
        case "file_info": return "info.circle"
        default: return "wrench.and.screwdriver"
        }
    }
}
