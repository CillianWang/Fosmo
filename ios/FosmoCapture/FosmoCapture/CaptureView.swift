import Foundation
import SwiftUI

struct CaptureView: View {
    @StateObject private var controller = ScanCaptureController()
    @State private var showsModelPreview = ProcessInfo.processInfo.arguments.contains("--open-model-preview")

    var body: some View {
        ZStack {
            ARCameraView(session: controller.session)
                .ignoresSafeArea()

            LinearGradient(
                colors: [.black.opacity(0.65), .clear, .black.opacity(0.75)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()
            .allowsHitTesting(false)

            VStack(spacing: 18) {
                header
                Spacer()
                guidanceCard
                controls
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
        }
        .preferredColorScheme(.dark)
        .onAppear { controller.startPreview() }
        .onDisappear { controller.pausePreview() }
        .fullScreenCover(isPresented: $showsModelPreview) {
            ModelPreviewView(backendURLString: controller.backendURLString)
                .onAppear { controller.pausePreview() }
                .onDisappear { controller.startPreview() }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text("FOSMO")
                    .font(.headline.monospaced().weight(.bold))
                Text(phaseLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            HStack(spacing: 8) {
                Circle()
                    .fill(controller.trackingIsNormal ? .green : .orange)
                    .frame(width: 8, height: 8)
                Text(controller.trackingIsNormal ? "跟踪正常" : "正在定位")
                    .font(.caption.weight(.semibold))
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
        }
    }

    private var guidanceCard: some View {
        HStack(spacing: 14) {
            ZStack {
                Circle()
                    .stroke(.white.opacity(0.18), lineWidth: 5)
                Circle()
                    .trim(
                        from: 0,
                        to: Double(controller.coverageSectors) / Double(max(1, controller.requiredCoverageSectors))
                    )
                    .stroke(.green, style: StrokeStyle(lineWidth: 5, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Text("\(controller.coverageSectors)")
                    .font(.title2.monospacedDigit().weight(.bold))
            }
            .frame(width: 58, height: 58)

            VStack(alignment: .leading, spacing: 5) {
                Text("后端覆盖 \(controller.coverageSectors) / \(controller.requiredCoverageSectors)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text(controller.guidance)
                    .font(.body.weight(.semibold))
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .padding(16)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    @ViewBuilder
    private var controls: some View {
        switch controller.phase {
        case .idle:
            VStack(spacing: 12) {
                TextField("http://Mac地址:8765", text: $controller.backendURLString)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                    .padding(12)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                primaryButton("连接后端并开始", systemImage: "viewfinder") {
                    controller.startScan()
                }
                Button {
                    showsModelPreview = true
                } label: {
                    Label("查看最近重建", systemImage: "cube.transparent")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 13)
                }
                .buttonStyle(.bordered)
            }
        case .connecting:
            HStack(spacing: 12) {
                ProgressView()
                Text("正在连接覆盖后端")
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
        case .scanning:
            Button {
                controller.finishScan()
            } label: {
                Label(
                    controller.coverageComplete ? "结束并导出" : "等待后端确认完整一圈",
                    systemImage: "stop.fill"
                )
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
            .disabled(!controller.coverageComplete)
        case .readyToShare:
            if let url = controller.exportedBundleURL {
                ShareLink(item: url) {
                    Label("分享 ScanBundle", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 15)
                }
                .buttonStyle(.borderedProminent)
                .tint(.green)
            }
            Button("重新扫描") { controller.reset() }
                .buttonStyle(.bordered)
        case .failed(let message):
            VStack(spacing: 10) {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                primaryButton("重试", systemImage: "arrow.clockwise") {
                    controller.reset()
                    controller.startPreview()
                }
                Button {
                    showsModelPreview = true
                } label: {
                    Label("查看最近重建", systemImage: "cube.transparent")
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 13)
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private func primaryButton(
        _ title: String,
        systemImage: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 15)
        }
        .buttonStyle(.borderedProminent)
        .tint(.green)
    }

    private var phaseLabel: String {
        switch controller.phase {
        case .idle: "室内 RGB 扫描"
        case .connecting: "连接覆盖服务"
        case .scanning: "正在采集 ScanBundle 1.0"
        case .readyToShare: "扫描完成"
        case .failed: "需要处理"
        }
    }
}
