@preconcurrency import ARKit
import Combine
import Foundation
import ScanBundleKit
import UIKit

@MainActor
final class ScanCaptureController: NSObject, ObservableObject {
    enum Phase: Equatable {
        case idle
        case connecting
        case scanning
        case readyToShare
        case failed(String)
    }

    let session = ARSession()

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var keyframeCount = 0
    @Published private(set) var guidance = "点击开始，缓慢环绕房间"
    @Published private(set) var trackingIsNormal = false
    @Published private(set) var exportedBundleURL: URL?
    @Published var backendURLString: String
    @Published private(set) var coverageSectors = 0
    @Published private(set) var requiredCoverageSectors = 12
    @Published private(set) var coverageComplete = false

    let maximumFrameCount = 120

    private var selector = KeyframeSelector(
        policy: KeyframePolicy(minimumSharpness: 0.015)
    )
    private var writer: ScanBundleWriter?
    private var scanID: UUID?
    private var createdAt: String?
    private var isProcessingFrame = false
    private var coverageClient: CoverageBackendClient?
    private var coverageSessionID: String?

    override init() {
        backendURLString = ProcessInfo.processInfo.environment["FOSMO_BACKEND_URL"]
            ?? UserDefaults.standard.string(forKey: "coverageBackendURL")
            ?? "http://MoonShapedPool.local:8765"
        super.init()
        session.delegate = self
        session.delegateQueue = .main
    }

    func startPreview() {
        guard ARWorldTrackingConfiguration.isSupported else {
            phase = .failed("此设备不支持 ARKit 世界跟踪")
            return
        }
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        configuration.environmentTexturing = .none
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func pausePreview() {
        session.pause()
    }

    func startScan() {
        guard phase == .idle || isFailure else { return }
        phase = .connecting
        guidance = "正在连接覆盖后端"
        Task { await connectAndStartScan() }
    }

    private var isFailure: Bool {
        if case .failed = phase { return true }
        return false
    }

    private func connectAndStartScan() async {
        do {
            guard let backendURL = normalizedBackendURL() else {
                throw CoverageBackendError.invalidURL
            }
            let client = CoverageBackendClient(baseURL: backendURL)
            let backendSession = try await client.createSession()
            guard let backendSessionID = backendSession.sessionID else {
                throw CoverageBackendError.invalidResponse
            }
            let id = UUID()
            let scansDirectory = try Self.scansDirectory()
            writer = try ScanBundleWriter(outputDirectory: scansDirectory, scanID: id)
            coverageClient = client
            coverageSessionID = backendSessionID
            scanID = id
            createdAt = Self.iso8601Now()
            selector = KeyframeSelector(policy: KeyframePolicy(minimumSharpness: 0.015))
            keyframeCount = 0
            coverageSectors = backendSession.sectorsCovered
            requiredCoverageSectors = backendSession.sectorsRequired
            coverageComplete = false
            exportedBundleURL = nil
            UserDefaults.standard.set(backendURL.absoluteString, forKey: "coverageBackendURL")
            guidance = "后端已连接；请稳定录制完整一圈"
            phase = .scanning
        } catch {
            phase = .failed("无法开始扫描：\(error.localizedDescription)")
        }
    }

    func finishScan() {
        guard phase == .scanning, coverageComplete else { return }
        Task { await finalizeBundle() }
    }

    func reset() {
        writer = nil
        scanID = nil
        createdAt = nil
        exportedBundleURL = nil
        keyframeCount = 0
        coverageSectors = 0
        requiredCoverageSectors = 12
        coverageComplete = false
        coverageClient = nil
        coverageSessionID = nil
        guidance = "点击开始，缓慢环绕房间"
        phase = .idle
    }

    private func process(_ arFrame: ARFrame) async {
        defer { isProcessingFrame = false }
        guard phase == .scanning, keyframeCount < maximumFrameCount else { return }

        do {
            let quality = FrameQualityAnalyzer.analyze(arFrame.capturedImage)
            let candidate = try ARKitFrameAdapter.candidate(
                from: arFrame,
                sharpness: quality.sharpness,
                meanLuminance: quality.meanLuminance
            )
            switch selector.evaluate(candidate) {
            case .accept:
                let captured = try ARKitFrameAdapter.capture(arFrame, id: keyframeCount + 1)
                guard let writer, let coverageClient, let coverageSessionID else { return }
                try await writer.append(jpegData: captured.jpegData, frame: captured.frame)
                keyframeCount += 1
                let thumbnail = try Self.thumbnailJPEG(from: captured.jpegData)
                let status = try await coverageClient.submit(
                    sessionID: coverageSessionID,
                    frame: captured.frame,
                    quality: quality,
                    thumbnailJPEG: thumbnail
                )
                coverageSectors = status.sectorsCovered
                requiredCoverageSectors = status.sectorsRequired
                coverageComplete = status.complete
                guidance = status.complete
                    ? "后端已确认完整一圈；可以结束并导出"
                    : "后端已确认 \(status.sectorsCovered)/\(status.sectorsRequired) 个方向；继续同一圈录制"
            case .reject(let reason):
                if reason == .trackingNotNormal {
                    guidance = "ARKit 跟踪不稳定；后端未收到当前帧"
                }
            }
        } catch {
            guidance = "后端未确认当前帧：\(error.localizedDescription)"
        }
    }

    private func finalizeBundle() async {
        guard let writer, let createdAt else { return }
        do {
            let device = ScanDevice(
                model: UIDevice.current.model,
                osVersion: "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"
            )
            exportedBundleURL = try await writer.finalize(createdAt: createdAt, device: device)
            self.writer = nil
            coverageClient = nil
            coverageSessionID = nil
            guidance = "ScanBundle 已生成，可通过隔空投送或文件导出"
            phase = .readyToShare
        } catch {
            phase = .failed("导出失败：\(error.localizedDescription)")
        }
    }

    private func normalizedBackendURL() -> URL? {
        let trimmed = backendURLString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard var components = URLComponents(string: trimmed),
              components.scheme == "http" || components.scheme == "https",
              components.host != nil else {
            return nil
        }
        components.path = components.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return components.url
    }

    private static func thumbnailJPEG(from jpegData: Data) throws -> Data {
        guard let image = UIImage(data: jpegData),
              let thumbnail = image.preparingThumbnail(of: CGSize(width: 320, height: 240)),
              let encoded = thumbnail.jpegData(compressionQuality: 0.65) else {
            throw CoverageBackendError.invalidResponse
        }
        return encoded
    }

    private static func scansDirectory() throws -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let scans = documents.appendingPathComponent("Scans", isDirectory: true)
        try FileManager.default.createDirectory(at: scans, withIntermediateDirectories: true)
        return scans
    }

    private static func iso8601Now() -> String {
        ISO8601DateFormatter.string(
            from: Date(),
            timeZone: .current,
            formatOptions: [.withInternetDateTime, .withFractionalSeconds]
        )
    }
}

extension ScanCaptureController: @preconcurrency ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        trackingIsNormal = {
            if case .normal = frame.camera.trackingState { return true }
            return false
        }()
        guard phase == .scanning, !isProcessingFrame else { return }
        isProcessingFrame = true
        Task { await process(frame) }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        trackingIsNormal = {
            if case .normal = camera.trackingState { return true }
            return false
        }()
        if !trackingIsNormal, phase == .scanning {
            guidance = "ARKit 跟踪不稳定；后端不会确认这些帧"
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        phase = .failed("ARSession 失败：\(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        if phase == .scanning {
            guidance = "扫描已暂停，请保持当前位置"
        }
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        startPreview()
        if phase == .scanning {
            guidance = "请回看刚才扫过的区域，等待跟踪恢复"
        }
    }
}
