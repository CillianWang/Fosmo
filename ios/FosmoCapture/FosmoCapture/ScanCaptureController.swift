@preconcurrency import ARKit
import Combine
import Foundation
import ScanBundleKit
import UIKit

@MainActor
final class ScanCaptureController: NSObject, ObservableObject {
    enum Phase: Equatable {
        case idle
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

    let minimumFrameCount = 3
    let maximumFrameCount = 5

    private var selector = KeyframeSelector(
        policy: KeyframePolicy(minimumSharpness: 0.015)
    )
    private var writer: ScanBundleWriter?
    private var scanID: UUID?
    private var createdAt: String?
    private var isProcessingFrame = false

    override init() {
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
        do {
            let id = UUID()
            let scansDirectory = try Self.scansDirectory()
            writer = try ScanBundleWriter(outputDirectory: scansDirectory, scanID: id)
            scanID = id
            createdAt = Self.iso8601Now()
            selector = KeyframeSelector(policy: KeyframePolicy(minimumSharpness: 0.015))
            keyframeCount = 0
            exportedBundleURL = nil
            guidance = "缓慢移动手机，保持墙面清晰可见"
            phase = .scanning
        } catch {
            phase = .failed("无法创建扫描目录：\(error.localizedDescription)")
        }
    }

    func finishScan() {
        guard phase == .scanning, keyframeCount >= minimumFrameCount else { return }
        Task { await finalizeBundle() }
    }

    func reset() {
        writer = nil
        scanID = nil
        createdAt = nil
        exportedBundleURL = nil
        keyframeCount = 0
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
                guard let writer else { return }
                try await writer.append(jpegData: captured.jpegData, frame: captured.frame)
                keyframeCount += 1
                if keyframeCount == maximumFrameCount {
                    guidance = "已采集 5 帧，正在生成 ScanBundle"
                    await finalizeBundle()
                } else if keyframeCount >= minimumFrameCount {
                    guidance = "已保存 \(keyframeCount) 帧；可结束，或继续补扫"
                } else {
                    guidance = "已保存 \(keyframeCount) 帧；继续缓慢移动"
                }
            case .reject(let reason):
                guidance = Self.guidance(for: reason)
            }
        } catch {
            guidance = "当前帧未保存：\(error.localizedDescription)"
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
            guidance = "ScanBundle 已生成，可通过隔空投送或文件导出"
            phase = .readyToShare
        } catch {
            phase = .failed("导出失败：\(error.localizedDescription)")
        }
    }

    private static func guidance(for rejection: KeyframeRejection) -> String {
        switch rejection {
        case .trackingNotNormal:
            return "跟踪不稳定：放慢并回看刚才扫过的区域"
        case .tooSoon:
            return "继续缓慢移动，保持画面稳定"
        case .blurry:
            return "画面偏模糊：请放慢"
        case .exposureOutOfRange:
            return "画面过暗或过亮：换一个观察角度"
        case .insufficientMotion:
            return "向左、向右或前后移动一小步"
        }
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
            guidance = "跟踪不稳定：放慢并回看刚才扫过的区域"
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
