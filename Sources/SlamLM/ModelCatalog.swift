import Foundation

/// Search, filter and sort over the bridge's catalog.
///
/// A pure value layer: `ModelStore` owns the filter state and calls in here, so
/// the rules stay testable without a running bridge.
enum ModelCatalog {
    /// The chip that clears the category filter.
    static let allCategory = "All"

    /// PROTOCOL.md's canonical display order for the derived categories, used
    /// when the bridge has not told us its own order yet.
    static let canonicalOrder = ["Chat", "Code", "Vision", "Embedding", "Audio"]

    /// Chips to show: `All` first, then the canonical categories that are
    /// actually present, then any other real category the models carry so no
    /// discovered signal is hidden.
    static func chips(discovered: [String], models: [ModelInfo]) -> [String] {
        let present = Set(models.flatMap(\.categories))
        guard !present.isEmpty else { return [allCategory] }
        let canonical = discovered.isEmpty ? canonicalOrder : discovered
        var ordered = canonical.filter { present.contains($0) }
        ordered += present.subtracting(canonical).sorted()
        return [allCategory] + ordered
    }

    /// Case-insensitive substring match on display name or repo id.
    static func matches(_ model: ModelInfo, query: String) -> Bool {
        let needle = query.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty else { return true }
        return model.name.lowercased().contains(needle) || model.id.lowercased().contains(needle)
    }

    /// Filter by category and query, then sort by `lastUsed` descending with the
    /// name (and finally the id) breaking ties.
    static func filter(_ models: [ModelInfo], query: String, category: String) -> [ModelInfo] {
        models
            .filter { model in
                guard category == allCategory || model.categories.contains(category) else { return false }
                return matches(model, query: query)
            }
            .sorted { lhs, rhs in
                if lhs.lastUsed != rhs.lastUsed { return lhs.lastUsed > rhs.lastUsed }
                let byName = lhs.name.localizedCaseInsensitiveCompare(rhs.name)
                if byName != .orderedSame { return byName == .orderedAscending }
                return lhs.id < rhs.id
            }
    }
}
