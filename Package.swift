// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "SlamLM",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "SlamLM",
            path: "Sources/SlamLM"
        )
    ]
)
