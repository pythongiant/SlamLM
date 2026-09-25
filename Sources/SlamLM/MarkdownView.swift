import SwiftUI

// MARK: - Markdown rendering
//
// The bridge streams a model's raw text, so the parser has to survive partial
// input: an unterminated `**`, a fence that has not closed, or a ` thinking`
// block that is still arriving all render as something sane on every frame.
// Styling uses the panel's own palette so the output reads as the same material
// as the cards around it.

/// One block of a parsed document.
enum MarkdownBlock: Equatable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case bullets([String])
    case numbered([String])
    case code(language: String?, text: String)
    case quote(String)
    case rule
    case table(header: [String], rows: [[String]])
}

/// A model's output split into its reasoning and its answer, then into blocks.
struct MarkdownDocument {
    /// Reasoning carried in the model's own think tags, if any.
    let thinking: String?
    /// False while the closing tag has not arrived yet.
    let thinkingComplete: Bool
    let blocks: [MarkdownBlock]

    init(_ raw: String) {
        let (reasoning, answer, complete) = Self.splitThinking(raw)
        thinking = reasoning
        thinkingComplete = complete
        blocks = Self.parse(answer)
    }

    /// Qwen3 and friends wrap reasoning in ` thinking …  response`, and the first
    /// tag may be omitted by some templates. Either way the answer is what
    /// follows the closing tag.
    static func splitThinking(_ raw: String) -> (reasoning: String?, answer: String, complete: Bool) {
        let lower = raw.lowercased()
        let openRange = lower.range(of: "<think>")
        let closeRange = lower.range(of: "</think>")

        guard let open = openRange, let close = closeRange, close.lowerBound > open.upperBound else {
            if let close = closeRange {
                // No opening tag: everything before the closer is reasoning.
                let reasoning = String(raw[raw.startIndex..<close.lowerBound])
                let answer = String(raw[close.upperBound...])
                return (trimmed(reasoning), answer, true)
            }
            if let open = openRange {
                // Still streaming: the closer has not arrived.
                return (trimmed(String(raw[open.upperBound...])), "", false)
            }
            return (nil, raw, true)
        }

        let reasoning = String(raw[open.upperBound..<close.lowerBound])
        let answer = String(raw[close.upperBound...])
        return (trimmed(reasoning), answer, true)
    }

    private static func trimmed(_ text: String) -> String? {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }

    /// Line-oriented block parse; enough of CommonMark for model output.
    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = []
        var bullets: [String] = []
        var numbers: [String] = []
        var quote: [String] = []

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            blocks.append(.paragraph(paragraph.joined(separator: " ")))
            paragraph = []
        }
        func flushBullets() {
            guard !bullets.isEmpty else { return }
            blocks.append(.bullets(bullets))
            bullets = []
        }
        func flushNumbers() {
            guard !numbers.isEmpty else { return }
            blocks.append(.numbered(numbers))
            numbers = []
        }
        func flushQuote() {
            guard !quote.isEmpty else { return }
            blocks.append(.quote(quote.joined(separator: " ")))
            quote = []
        }
        func flushAll() {
            flushParagraph(); flushBullets(); flushNumbers(); flushQuote()
        }

        var lines = text.components(separatedBy: .newlines)
        var index = 0
        while index < lines.count {
            let line = lines[index]
            let trimmedLine = line.trimmingCharacters(in: .whitespaces)

            // Fenced code, complete or still streaming.
            if trimmedLine.hasPrefix("```") {
                flushAll()
                let language = String(trimmedLine.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var body: [String] = []
                index += 1
                while index < lines.count, !lines[index].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                    body.append(lines[index])
                    index += 1
                }
                index += 1     // past the closing fence, or past the end
                blocks.append(.code(language: language.isEmpty ? nil : language,
                                    text: body.joined(separator: "\n")))
                continue
            }

            // Table: a pipe row followed by a separator row.
            if trimmedLine.contains("|"), index + 1 < lines.count,
               Self.isTableSeparator(lines[index + 1]) {
                flushAll()
                let header = Self.cells(trimmedLine)
                var rows: [[String]] = []
                index += 2
                while index < lines.count, lines[index].contains("|"),
                      !lines[index].trimmingCharacters(in: .whitespaces).isEmpty {
                    rows.append(Self.cells(lines[index]))
                    index += 1
                }
                blocks.append(.table(header: header, rows: rows))
                continue
            }

            if trimmedLine.isEmpty {
                flushAll()
                index += 1
                continue
            }

            if let heading = Self.heading(trimmedLine) {
                flushAll()
                blocks.append(heading)
                index += 1
                continue
            }

            if Self.isRule(trimmedLine) {
                flushAll()
                blocks.append(.rule)
                index += 1
                continue
            }

            if let item = Self.bulletItem(line) {
                flushParagraph(); flushNumbers(); flushQuote()
                bullets.append(item)
                index += 1
                continue
            }

            if let item = Self.numberedItem(trimmedLine) {
                flushParagraph(); flushBullets(); flushQuote()
                numbers.append(item)
                index += 1
                continue
            }

            if trimmedLine.hasPrefix(">") {
                flushParagraph(); flushBullets(); flushNumbers()
                quote.append(String(trimmedLine.dropFirst()).trimmingCharacters(in: .whitespaces))
                index += 1
                continue
            }

            // Indented continuation of whatever list is open, else a paragraph.
            if line.hasPrefix("  "), !bullets.isEmpty || !numbers.isEmpty {
                if !bullets.isEmpty { bullets[bullets.count - 1] += " " + trimmedLine }
                else { numbers[numbers.count - 1] += " " + trimmedLine }
                index += 1
                continue
            }

            flushBullets(); flushNumbers(); flushQuote()
            paragraph.append(trimmedLine)
            index += 1
        }

        flushAll()
        return blocks
    }

    private static func heading(_ line: String) -> MarkdownBlock? {
        var level = 0
        for character in line {
            if character == "#" { level += 1 } else { break }
        }
        guard level > 0, level <= 6 else { return nil }
        let rest = line.dropFirst(level)
        guard rest.hasPrefix(" ") else { return nil }
        return .heading(level: level, text: rest.trimmingCharacters(in: .whitespaces))
    }

    private static func isRule(_ line: String) -> Bool {
        let stripped = line.replacingOccurrences(of: " ", with: "")
        guard stripped.count >= 3 else { return false }
        return stripped.allSatisfy { $0 == "-" } || stripped.allSatisfy { $0 == "*" } || stripped.allSatisfy { $0 == "_" }
    }

    private static func bulletItem(_ line: String) -> String? {
        for marker in ["- ", "* ", "+ ", "• "] where line.hasPrefix(marker) {
            return String(line.dropFirst(marker.count)).trimmingCharacters(in: .whitespaces)
        }
        return nil
    }

    private static func numberedItem(_ line: String) -> String? {
        var digits = ""
        var rest = Substring(line)
        while let first = rest.first, first.isNumber {
            digits.append(first)
            rest = rest.dropFirst()
        }
        guard !digits.isEmpty, rest.hasPrefix(". ") || rest.hasPrefix(") ") else { return nil }
        return String(rest.dropFirst(2)).trimmingCharacters(in: .whitespaces)
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let trimmedLine = line.trimmingCharacters(in: .whitespaces)
        guard trimmedLine.contains("-"), trimmedLine.contains("|") else { return false }
        let allowed = CharacterSet(charactersIn: "-:| ")
        return trimmedLine.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    private static func cells(_ line: String) -> [String] {
        var parts = line.trimmingCharacters(in: .whitespaces)
            .split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        if parts.first?.isEmpty == true { parts.removeFirst() }
        if parts.last?.isEmpty == true { parts.removeLast() }
        return parts
    }
}

// MARK: - Inline styling

enum MarkdownInline {
    /// Parses inline markdown and paints it in the panel's palette. Inline-only
    /// so block structure stays with the block parser, and partially-parsed text
    /// is preferred so a half-typed `**` still renders.
    static func attributed(_ text: String, baseSize: CGFloat = 12) -> AttributedString {
        var options = AttributedString.MarkdownParsingOptions()
        options.interpretedSyntax = .inlineOnlyPreservingWhitespace
        options.failurePolicy = .returnPartiallyParsedIfPossible
        var attributed = (try? AttributedString(markdown: text, options: options)) ?? AttributedString(text)

        for run in attributed.runs {
            var container = AttributeContainer()
            let intent = run.inlinePresentationIntent
            if intent?.contains(.code) == true {
                container.font = .system(size: baseSize - 1, design: .monospaced)
                container.foregroundColor = Paper.clayDeep
                container.backgroundColor = Paper.cardSunken
            } else {
                var font = Font.system(size: baseSize)
                if intent?.contains(.stronglyEmphasized) == true { font = font.weight(.semibold) }
                if intent?.contains(.emphasized) == true { font = font.italic() }
                container.font = font
                container.foregroundColor = Paper.ink
            }
            if intent?.contains(.strikethrough) == true {
                container.strikethroughStyle = .single
                container.foregroundColor = Paper.inkFaint
            }
            if run.link != nil {
                container.foregroundColor = Paper.accent
                container.underlineStyle = .single
            }
            attributed[run.range].mergeAttributes(container)
        }
        return attributed
    }
}

// MARK: - Views

/// A model's output as markdown, in the panel's palette.
struct MarkdownText: View {
    let raw: String

    var body: some View {
        let document = MarkdownDocument(raw)
        VStack(alignment: .leading, spacing: 7) {
            if let thinking = document.thinking {
                ThinkingBlock(text: thinking, complete: document.thinkingComplete)
            }
            ForEach(Array(document.blocks.enumerated()), id: \.offset) { _, block in
                BlockView(block: block)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Reasoning, recessed and collapsed by default: it is the model's working, not
/// its answer, and on a long think block it otherwise buries the reply.
private struct ThinkingBlock: View {
    let text: String
    let complete: Bool

    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Button {
                expanded.toggle()
            } label: {
                HStack(spacing: 5) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.system(size: 8, weight: .bold))
                    SectionLabel(text: complete ? "Thinking · \(text.count) chars" : "Thinking… · \(text.count) chars",
                                 color: Paper.butterInk.opacity(0.8))
                    Spacer(minLength: 0)
                }
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(Text(expanded ? "Hide the model's reasoning" : "Show the model's reasoning"))

            if expanded {
                Text(text)
                    .font(.system(size: 11))
                    .foregroundStyle(Paper.inkSoft)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 9)
        .padding(.vertical, 7)
        .background(RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Paper.washButter))
        .overlay(
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .strokeBorder(Paper.butterInk.opacity(0.18), lineWidth: 1)
        )
    }
}

private struct BlockView: View {
    let block: MarkdownBlock

    var body: some View {
        switch block {
        case let .heading(level, text):
            Text(MarkdownInline.attributed(text, baseSize: level <= 2 ? 14 : 12.5))
                .font(.system(size: level <= 2 ? 14 : 12.5, weight: .semibold))
                .foregroundStyle(Paper.ink)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 1)

        case let .paragraph(text):
            Text(MarkdownInline.attributed(text))
                .lineSpacing(2)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)

        case let .bullets(items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    row(marker: "•", markerColor: Paper.olive, text: item)
                }
            }

        case let .numbered(items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    row(marker: "\(index + 1).", markerColor: Paper.inkSoft, text: item)
                }
            }

        case let .code(language, text):
            VStack(alignment: .leading, spacing: 4) {
                if let language {
                    SectionLabel(text: language, color: Paper.inkFaint, size: 8.5)
                }
                Text(text)
                    .font(.system(size: 10.5, design: .monospaced))
                    .foregroundStyle(Paper.fieldInk)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(9)
            .background(RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Paper.cardSunken))

        case let .quote(text):
            HStack(alignment: .top, spacing: 8) {
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(Paper.olive)
                    .frame(width: 2.5)
                Text(MarkdownInline.attributed(text))
                    .italic()
                    .foregroundStyle(Paper.inkSoft)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .fixedSize(horizontal: false, vertical: true)

        case .rule:
            PaperRule()

        case let .table(header, rows):
            VStack(spacing: 0) {
                tableRow(header, isHeader: true)
                ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                    PaperRule(color: index.isMultiple(of: 2) ? Paper.hairline : Paper.stroke)
                    tableRow(row, isHeader: false)
                }
            }
            .background(RoundedRectangle(cornerRadius: 9, style: .continuous).fill(Paper.card))
            .overlay(
                RoundedRectangle(cornerRadius: 9, style: .continuous)
                    .strokeBorder(Paper.stroke, lineWidth: 1)
            )
        }
    }

    private func row(marker: String, markerColor: Color, text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text(marker)
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(markerColor)
                .frame(minWidth: 12, alignment: .trailing)
            Text(MarkdownInline.attributed(text))
                .lineSpacing(1.5)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func tableRow(_ cells: [String], isHeader: Bool) -> some View {
        HStack(alignment: .top, spacing: 8) {
            ForEach(Array(cells.enumerated()), id: \.offset) { _, cell in
                Text(MarkdownInline.attributed(cell, baseSize: 11))
                    .font(.system(size: 11, weight: isHeader ? .semibold : .regular))
                    .foregroundStyle(isHeader ? Paper.ink : Paper.inkSoft)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
    }
}
