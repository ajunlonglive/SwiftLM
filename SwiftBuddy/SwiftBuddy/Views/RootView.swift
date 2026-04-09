// RootView.swift — Adaptive root layout: tab bar on iOS, sidebar on macOS
import SwiftUI
import SwiftData
#if canImport(MLXInferenceCore)
import MLXInferenceCore
#endif

struct RootView: View {
    @EnvironmentObject private var engine: InferenceEngine
    @EnvironmentObject private var appearance: AppearanceStore
    @Environment(\.modelContext) private var modelContext
    @StateObject private var viewModel = ChatViewModel()
    @StateObject private var registry = RegistryService.shared
    @Query(sort: \PalaceWing.createdDate) var wings: [PalaceWing]

    // iOS: tab selection
    @State private var selectedTab: Tab = .chat

    // macOS sheets
    @State private var showModelPicker = false
    @State private var showSettings = false
    @State private var showMap = false
    @State private var showTextIngestion = false
    @State private var showInspector = true

    enum Tab { case chat, models, palace, miner, settings }

    var body: some View {
        Group {
            #if os(macOS)
            macOSLayout
                .sheet(isPresented: $showModelPicker) {
                    ModelPickerView(onSelect: { modelId in
                        showModelPicker = false
                        Task { await engine.load(modelId: modelId) }
                    })
                    .environmentObject(engine)
                }
                .sheet(isPresented: $showSettings) {
                    SettingsView(viewModel: viewModel)
                        .environmentObject(appearance)
                }
                .sheet(isPresented: $showMap) {
                    PalaceVisualizerView()
                        .frame(width: 800, height: 600)
                }
                .sheet(isPresented: $showTextIngestion) {
                    TextIngestionView()
                        .environmentObject(engine)
                }
                .onReceive(NotificationCenter.default.publisher(for: .showModelPicker)) { _ in
                    showModelPicker = true
                }
                .onAppear {
                    viewModel.engine = engine
                    viewModel.modelContext = modelContext
                }
                .onChange(of: engine.state) { _, state in
                }
                .overlay {
                    if registry.isSyncing {
                        PersonaExtractionOverlay(registry: registry)
                    }
                }
            #else
            iOSTabView
                .onAppear { 
                    viewModel.engine = engine 
                    viewModel.modelContext = modelContext
                }
            #endif
        }
    }

    // MARK: — iOS Tab View

    #if os(iOS)
    private var iOSTabView: some View {
        TabView(selection: $selectedTab) {
            // ── Chat Tab ──────────────────────────────────────────────────
            NavigationStack {
                List {
                    Section("Conversations") {
                        NavigationLink {
                            ChatView(viewModel: viewModel)
                                .environmentObject(engine)
                                .onAppear { 
                                    viewModel.currentWing = nil
                                    viewModel.newConversation()
                                }
                        } label: {
                            Label("Core System Chat", systemImage: "sparkles")
                        }
                    }
                    
                    Section("Friends (Personas)") {
                        ForEach(wings) { wing in
                            NavigationLink {
                                ChatView(viewModel: viewModel)
                                    .environmentObject(engine)
                                    .onAppear { 
                                        viewModel.currentWing = wing.name 
                                        viewModel.newConversation()
                                    }
                            } label: {
                                Label(wing.name, systemImage: "person.crop.circle")
                            }
                            .swipeActions {
                                Button(role: .destructive) {
                                    modelContext.delete(wing)
                                    try? modelContext.save()
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                        }
                    }
                }
                .navigationTitle("Connections")
            }
            .tabItem {
                Label("Chat", systemImage: selectedTab == .chat
                      ? "bubble.left.and.bubble.right.fill"
                      : "bubble.left.and.bubble.right")
            }
            .tag(Tab.chat)

            // ── Models Tab ────────────────────────────────────────────────
            NavigationStack {
                ModelsView(viewModel: viewModel)
                    .environmentObject(engine)
            }
            .tabItem {
                Label("Models", systemImage: selectedTab == .models ? "cpu.fill" : "cpu")
            }
            .tag(Tab.models)
            .badge(engine.downloadManager.activeDownloads.isEmpty
                   ? 0
                   : engine.downloadManager.activeDownloads.count)

            // ── Palace Tab ──────────────────────────────────────────────
            NavigationStack {
                PalaceVisualizerView()
            }
            .tabItem {
                Label("Palace", systemImage: selectedTab == .palace ? "brain.head.profile" : "brain")
            }
            .tag(Tab.palace)
            
            // ── Miner Tab ──────────────────────────────────────────────
            NavigationStack {
                TextIngestionView()
                    .environmentObject(engine)
                    .navigationTitle("Memory Miner")
            }
            .tabItem {
                Label("Miner", systemImage: selectedTab == .miner ? "hammer.fill" : "hammer")
            }
            .tag(Tab.miner)

            // ── Settings Tab ──────────────────────────────────────────────
            NavigationStack {
                SettingsView(viewModel: viewModel, isTab: true)
                    .environmentObject(appearance)
            }
            .tabItem {
                Label("Settings", systemImage: selectedTab == .settings ? "gearshape.fill" : "gearshape")
            }
            .tag(Tab.settings)
        }
        .tint(SwiftBuddyTheme.accent)
        // Navigate to Models tab when a model load is requested from chat
        .onReceive(NotificationCenter.default.publisher(for: .showModelPicker)) { _ in
            selectedTab = .models
        }
    }
    #endif

    // MARK: — macOS Split View

    #if os(macOS)
    private var macOSLayout: some View {
        NavigationSplitView {
            VStack(alignment: .leading, spacing: 0) {
                // ── Branded sidebar header ────────────────────────────────
                sidebarHeader
                Divider()
                    .background(SwiftBuddyTheme.divider)

                // ── Engine status ─────────────────────────────────────────
                engineStatusSection
                Divider()
                    .background(SwiftBuddyTheme.divider)

                // ── Actions list ──────────────────────────────────────────
                List {
                    Section("Conversations") {
                        Button {
                            viewModel.currentWing = nil
                            viewModel.newConversation()
                        } label: {
                            Label("Core Chat", systemImage: "sparkles")
                                .foregroundStyle(SwiftBuddyTheme.accent)
                        }
                        .buttonStyle(.plain)
                        
                        Button {
                            showMap = true
                        } label: {
                            Label("Memory Map", systemImage: "map.fill")
                                .foregroundStyle(.orange)
                        }
                        .buttonStyle(.plain)
                    }
                    
                    Section("Tools") {
                        Button {
                            showTextIngestion = true
                        } label: {
                            Label("Text Ingestion", systemImage: "hammer.fill")
                                .foregroundStyle(SwiftBuddyTheme.cyan)
                        }
                        .buttonStyle(.plain)
                    }
                    
                    Section("Friends (Personas)") {
                        ForEach(wings) { wing in
                            Button {
                                viewModel.currentWing = wing.name
                                viewModel.newConversation()
                            } label: {
                                Label(wing.name, systemImage: "person.crop.circle")
                            }
                            .buttonStyle(.plain)
                            .contextMenu {
                                Button(role: .destructive) {
                                    modelContext.delete(wing)
                                    try? modelContext.save()
                                } label: {
                                    Label("Delete Persona", systemImage: "trash")
                                }
                            }
                        }
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
                .background(SwiftBuddyTheme.background)
            }
            .frame(minWidth: 220)
            .background(SwiftBuddyTheme.background)
        } detail: {
            ChatView(
                viewModel: viewModel,
                showSettings: $showSettings,
                showModelPicker: $showModelPicker,
                showInspector: $showInspector
            )
            .frame(minWidth: 400)
            .background(SwiftBuddyTheme.background)
            .navigationTitle("Chat")
            .inspector(isPresented: $showInspector) {
                InspectorView(
                    showModelPicker: $showModelPicker
                )
                .inspectorColumnWidth(min: 250, ideal: 275, max: 350)
                .background(SwiftBuddyTheme.background)
            }
        }
    }

    // Branded header — bolt icon + SwiftBuddy wordmark + version chip
    private var sidebarHeader: some View {
        HStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(SwiftBuddyTheme.heroGradient)
                    .frame(width: 32, height: 32)
                Image(systemName: "bolt.fill")
                    .font(.system(size: 14, weight: .bold))
                    .foregroundStyle(.white)
            }
            .shadow(color: SwiftBuddyTheme.accent.opacity(0.40), radius: 6)

            VStack(alignment: .leading, spacing: 1) {
                Text("SwiftBuddy")
                    .font(.system(.subheadline, weight: .bold))
                    .foregroundStyle(SwiftBuddyTheme.textPrimary)
                Text("Chat")
                    .font(.caption2)
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }

            Spacer()

            Text("v1.0")
                .font(.system(size: 9, weight: .bold))
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(SwiftBuddyTheme.accent.opacity(0.18), in: Capsule())
                .foregroundStyle(SwiftBuddyTheme.accent)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
    }

    // Engine status row in sidebar
    private var engineStatusSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Engine")
                .font(.caption.weight(.semibold))
                .foregroundStyle(SwiftBuddyTheme.textTertiary)
                .textCase(.uppercase)
                .padding(.horizontal, 14)
                .padding(.top, 10)

            engineStateView
                .padding(.horizontal, 14)
                .padding(.bottom, 10)
        }
    }

    @ViewBuilder
    private var engineStateView: some View {
        switch engine.state {
        case .idle:
            Button("Load Model") { showModelPicker = true }
                .buttonStyle(.borderedProminent)
                .tint(SwiftBuddyTheme.accent)
                .controlSize(.small)

        case .loading:
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini).tint(SwiftBuddyTheme.accent)
                Text("Loading…")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }

        case .downloading(let progress, let speed):
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: progress).tint(SwiftBuddyTheme.accent)
                Text("\(Int(progress * 100))% · \(speed)")
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(SwiftBuddyTheme.textTertiary)
            }

        case .ready(let modelId):
            HStack(spacing: 6) {
                Circle()
                    .fill(SwiftBuddyTheme.success)
                    .frame(width: 7, height: 7)
                Text(modelId.components(separatedBy: "/").last ?? modelId)
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
                    .lineLimit(1)
            }

        case .generating:
            HStack(spacing: 6) {
                GeneratingDots()
                Text("Generating…")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.textSecondary)
            }

        case .error(let msg):
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.error)
                Text(msg)
                    .font(.caption)
                    .foregroundStyle(SwiftBuddyTheme.error)
                    .lineLimit(2)
            }
        }
    }
    #endif
}

struct PersonaExtractionOverlay: View {
    @ObservedObject var registry: RegistryService
    @State private var isBlinking = false
    
    var body: some View {
        ZStack {
            // Dark transparent backing
            Color.black.opacity(0.85)
                .edgesIgnoringSafeArea(.all)
                
            VStack(alignment: .leading, spacing: 20) {
                // Header
                HStack {
                    Image(systemName: "cpu")
                        .font(.system(size: 24))
                        .foregroundColor(.green)
                        .symbolEffect(.pulse)
                    
                    Text("CONSCIOUSNESS SYNTHESIS")
                        .font(.system(size: 24, weight: .bold, design: .monospaced))
                        .foregroundColor(.green)
                    
                    Spacer()
                    
                    Text(isBlinking ? "_" : "")
                        .font(.system(size: 24, weight: .bold, design: .monospaced))
                        .foregroundColor(.green)
                        .onAppear {
                            withAnimation(Animation.easeInOut(duration: 0.5).repeatForever()) {
                                isBlinking.toggle()
                            }
                        }
                }
                
                Divider().background(Color.green.opacity(0.5))
                
                // Active Extraction Telemetry 
                VStack(alignment: .leading, spacing: 10) {
                    Text("> \(registry.lastSyncLog.uppercased())")
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundColor(.green)
                    
                    if registry.extractionTotal > 0 {
                        HStack {
                            Text("TARGET SECTOR: [\(registry.extractionPhase.uppercased())]")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.green.opacity(0.8))
                            Spacer()
                            Text("\(registry.extractionProcessed)/\(registry.extractionTotal) VECTORS")
                                .font(.system(size: 12, design: .monospaced))
                                .foregroundColor(.green.opacity(0.8))
                        }
                        
                        // Cyberpunk Progress Bar 
                        GeometryReader { proxy in
                            ZStack(alignment: .leading) {
                                Rectangle()
                                    .fill(Color.green.opacity(0.2))
                                    .frame(height: 12)
                                    .border(Color.green, width: 1)
                                
                                Rectangle()
                                    .fill(Color.green)
                                    .frame(width: proxy.size.width * CGFloat(registry.extractionProcessed) / CGFloat(max(1, registry.extractionTotal)), height: 12)
                                    .animation(.spring(), value: registry.extractionProcessed)
                            }
                        }
                        .frame(height: 12)
                        
                        // Scroll Matrix Text Preview
                        ScrollViewReader { scrollProxy in
                            ScrollView {
                                Text(registry.currentChunkText)
                                    .font(.system(size: 10, design: .monospaced))
                                    .foregroundColor(.green.opacity(0.6))
                                    .multilineTextAlignment(.leading)
                                    .lineSpacing(4)
                                    .id("bottom")
                            }
                            .frame(height: 120)
                            .padding()
                            .background(Color.black)
                            .border(Color.green.opacity(0.5), width: 1)
                            .onChange(of: registry.currentChunkText) { _ in
                                scrollProxy.scrollTo("bottom")
                            }
                        }
                    } else {
                        // Downloading Phase Waiter
                        HStack {
                            Text("ESTABLISHING MANIFOLD UPLINK...")
                                .font(.system(size: 14, design: .monospaced))
                                .foregroundColor(.green.opacity(0.6))
                            ProgressView()
                                .controlSize(.small)
                                .tint(.green)
                        }
                        .padding(.top, 20)
                    }
                }
            }
            .padding(30)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.black.opacity(0.9))
                    .border(Color.green.opacity(0.6), width: 2)
            )
            .shadow(color: .green.opacity(0.3), radius: 20, x: 0, y: 0)
            .frame(maxWidth: 600)
        }
        .zIndex(100)
    }
}
