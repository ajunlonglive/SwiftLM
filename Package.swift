// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftLM",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "MLXInferenceCore", targets: ["MLXInferenceCore"]),
        .executable(name: "SwiftLM", targets: ["SwiftLM"]),
        .executable(name: "SwiftBuddy", targets: ["SwiftBuddy"])
    ],
    dependencies: [
        // ── Dependency Update Flow ────────────────────────────────────────────────
        // ml-explore/mlx-swift  →  SharpAI/mlx-swift (sync bot PR + CI)  →  tag here
        // ml-explore/mlx-swift-lm → SharpAI/mlx-swift-lm (sync bot + Omni/SSD patches) → submodule SHA
        //
        // When a new SharpAI/mlx-swift tag is released, update_dependencies.yml
        // opens an automated PR bumping the version below. Do NOT float on branch: "main"
        // or you will inherit Apple upstream regressions immediately.
        //
        // ── Local Debug Override ──────────────────────────────────────────────────
        // To debug mlx-swift locally, comment the URL line and uncomment the path line:
        //   .package(path: "./mlx-swift"),
        // To debug mlx-swift-lm locally, it is already a submodule at ./mlx-swift-lm
        // ─────────────────────────────────────────────────────────────────────────

        // SharpAI fork of Apple MLX Swift — version-locked to a validated tag
        .package(url: "https://github.com/SharpAI/mlx-swift.git", revision: "b457"),
        // SharpAI fork of mlx-swift-lm — pinned via git submodule SHA (see .gitmodules)
        // Submodule tag is bumped automatically by update_dependencies.yml on new releases
        .package(path: "./mlx-swift-lm"),
        // HuggingFace tokenizers + model download
        .package(url: "https://github.com/huggingface/swift-transformers", .upToNextMinor(from: "1.2.0")),
        // Lightweight HTTP server (Apple-backed Swift server project)
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),
        // Async argument parser (for CLI flags: --model, --port)
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        // SwiftSoup for HTML parsing
        .package(url: "https://github.com/scinfu/SwiftSoup.git", from: "2.7.0"),
    ],
    targets: [
        // ── CLI HTTP server (macOS only) ──────────────────────────────
        .executableTarget(
            name: "SwiftLM",
            dependencies: [
                "MLXInferenceCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/SwiftLM"
        ),
        // ── STFT Audio Profiling Testing Script (macOS only) ───────────
        .executableTarget(
            name: "SwiftLMTestSTFT",
            dependencies: [
                "MLXInferenceCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/SwiftLMTestSTFT",
            exclude: ["ground_truth.py"]
        ),

        // ── macOS GUI App (SwiftBuddy) ──────────────────────────────
        .executableTarget(
            name: "SwiftBuddy",
            dependencies: [
                "MLXInferenceCore",
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "SwiftSoup", package: "SwiftSoup"),
            ],
            path: "SwiftBuddy/SwiftBuddy",
            exclude: [
                "Assets.xcassets",
                "SwiftBuddy.entitlements",
                "Personas/Lumina.json"
            ]
        ),
        // ── Shared inference library for SwiftLM Chat (iOS + macOS) ──
        .target(
            name: "MLXInferenceCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXInferenceCore",
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        // ── Automated Test Harness ──────────────────────────────────
        .testTarget(
            name: "SwiftBuddyTests",
            dependencies: ["SwiftBuddy", "MLXInferenceCore"]
        )
    ]
)
