import SceneKit
import SwiftUI
import UIKit

@MainActor
final class ModelPreviewViewModel: ObservableObject {
    @Published private(set) var scene: SCNScene?
    @Published private(set) var manifest: ModelPreviewManifest?
    @Published private(set) var errorMessage: String?
    @Published private(set) var isLoading = false

    func load(backendURLString: String) async {
        guard !isLoading, scene == nil else { return }
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            let trimmed = backendURLString.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let baseURL = URL(string: trimmed), baseURL.host != nil else {
                throw ModelPreviewError.invalidBackendURL
            }
            let client = ModelPreviewClient(baseURL: baseURL)
            let (manifest, data) = try await client.downloadPreview()
            let buffers = try await Task.detached(priority: .userInitiated) {
                try PLYMeshParser.parse(data)
            }.value
            self.manifest = manifest
            scene = Self.makeScene(
                from: buffers,
                floorTriangleCount: manifest.floorMeshTriangles,
                captureCenter: manifest.captureCenterMeters,
                initialForwardXZ: manifest.initialForwardXZ
            )
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private static func makeScene(
        from buffers: PLYMeshBuffers,
        floorTriangleCount: Int?,
        captureCenter: [Float]?,
        initialForwardXZ: [Float]?
    ) -> SCNScene {
        let scene = SCNScene()
        let modelNode = SCNNode(
            geometry: buffers.makeGeometry(floorTriangleCount: floorTriangleCount)
        )
        scene.rootNode.addChildNode(modelNode)

        let extent = buffers.maximum - buffers.minimum
        let radius = max(0.25, simd_length(extent) / 2)
        let fallbackCenter = (buffers.minimum + buffers.maximum) / 2
        let capturePosition: SIMD3<Float>
        if let captureCenter,
           captureCenter.count == 3,
           captureCenter.allSatisfy({ $0.isFinite }) {
            capturePosition = SIMD3(captureCenter[0], captureCenter[1], captureCenter[2])
        } else {
            capturePosition = fallbackCenter
        }
        let forwardXZ: SIMD2<Float>
        if let initialForwardXZ,
           initialForwardXZ.count == 2,
           initialForwardXZ.allSatisfy({ $0.isFinite }),
           simd_length(SIMD2(initialForwardXZ[0], initialForwardXZ[1])) > 0.1 {
            forwardXZ = simd_normalize(SIMD2(initialForwardXZ[0], initialForwardXZ[1]))
        } else {
            forwardXZ = SIMD2(0, -1)
        }

        let cameraNode = SCNNode()
        cameraNode.name = "fosmo-capture-camera"
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.fieldOfView = 64
        cameraNode.camera?.zNear = 0.025
        cameraNode.camera?.zFar = Double(max(20, radius * 8))
        cameraNode.camera?.wantsHDR = false
        cameraNode.camera?.wantsExposureAdaptation = false
        cameraNode.position = SCNVector3(
            capturePosition.x,
            capturePosition.y,
            capturePosition.z
        )
        cameraNode.eulerAngles = SCNVector3(0, atan2(-forwardXZ.x, -forwardXZ.y), 0)
        scene.rootNode.addChildNode(cameraNode)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 170
        ambient.light?.color = UIColor(red: 0.72, green: 0.80, blue: 0.90, alpha: 1)
        scene.rootNode.addChildNode(ambient)

        let directional = SCNNode()
        directional.light = SCNLight()
        directional.light?.type = .directional
        directional.light?.intensity = 1150
        directional.light?.temperature = 5200
        directional.light?.castsShadow = true
        directional.light?.shadowMode = .deferred
        directional.light?.shadowRadius = 5
        directional.light?.shadowSampleCount = 16
        directional.light?.shadowColor = UIColor.black.withAlphaComponent(0.55)
        directional.eulerAngles = SCNVector3(-0.85, 0.65, 0.18)
        scene.rootNode.addChildNode(directional)

        let roomLight = SCNNode()
        roomLight.light = SCNLight()
        roomLight.light?.type = .omni
        roomLight.light?.intensity = 520
        roomLight.light?.temperature = 3900
        roomLight.light?.attenuationStartDistance = 0.5
        roomLight.light?.attenuationEndDistance = CGFloat(max(4, radius * 2.5))
        roomLight.position = SCNVector3(
            capturePosition.x,
            min(buffers.maximum.y - 0.15, capturePosition.y + 0.9),
            capturePosition.z
        )
        scene.rootNode.addChildNode(roomLight)
        return scene
    }
}

struct ModelPreviewView: View {
    let backendURLString: String
    @Environment(\.dismiss) private var dismiss
    @StateObject private var model = ModelPreviewViewModel()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let scene = model.scene {
                SceneKitPreview(scene: scene)
                    .ignoresSafeArea()
            } else if model.isLoading {
                VStack(spacing: 14) {
                    ProgressView()
                        .controlSize(.large)
                    Text("正在从 Mac 下载并解析模型")
                    Text("首次载入约 12 MB")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else if let errorMessage = model.errorMessage {
                ContentUnavailableView(
                    "无法载入模型",
                    systemImage: "cube.transparent",
                    description: Text(errorMessage)
                )
            }

            VStack {
                HStack {
                    Button("关闭") { dismiss() }
                        .buttonStyle(.borderedProminent)
                        .tint(.black.opacity(0.65))
                    Spacer()
                    if let manifest = model.manifest {
                        Text(manifestLabel(manifest))
                            .font(.caption.monospacedDigit().weight(.semibold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                }
                Spacer()
                if model.scene != nil {
                    Text("单指 360° 环视 · 双指缩放 · 双击复位")
                        .font(.caption.weight(.medium))
                        .padding(.horizontal, 14)
                        .padding(.vertical, 9)
                        .background(.ultraThinMaterial, in: Capsule())
                }
            }
            .padding()
        }
        .preferredColorScheme(.dark)
        .task { await model.load(backendURLString: backendURLString) }
    }

    private func manifestLabel(_ manifest: ModelPreviewManifest) -> String {
        if manifest.modelKind == "world_top", let segmentCount = manifest.wallSegmentCount {
            if manifest.materialKind == "fused_rgb_vertex_colors" {
                return "World Top · 原始材质 · \(segmentCount) 段墙"
            }
            return "World Top · \(segmentCount) 段墙"
        }
        return "\(manifest.triangleCount.formatted()) 面"
    }
}

private struct SceneKitPreview: UIViewRepresentable {
    let scene: SCNScene

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = scene
        view.pointOfView = scene.rootNode.childNode(
            withName: "fosmo-capture-camera",
            recursively: true
        )
        view.backgroundColor = UIColor(red: 0.025, green: 0.03, blue: 0.04, alpha: 1)
        view.allowsCameraControl = false
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = false
        if let camera = view.pointOfView {
            context.coordinator.attach(to: view, camera: camera)
        }
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        view.scene = scene
    }

    @MainActor
    final class Coordinator: NSObject {
        private weak var view: SCNView?
        private weak var camera: SCNNode?
        private var yaw: Float = 0
        private var pitch: Float = 0
        private var initialPosition = SCNVector3Zero
        private var initialEulerAngles = SCNVector3Zero
        private var initialFieldOfView: CGFloat = 64
        private var pinchStartFieldOfView: CGFloat = 64

        func attach(to view: SCNView, camera: SCNNode) {
            self.view = view
            self.camera = camera
            yaw = camera.eulerAngles.y
            pitch = camera.eulerAngles.x
            initialPosition = camera.position
            initialEulerAngles = camera.eulerAngles
            initialFieldOfView = camera.camera?.fieldOfView ?? 64

            let pan = UIPanGestureRecognizer(target: self, action: #selector(handlePan(_:)))
            pan.maximumNumberOfTouches = 1
            view.addGestureRecognizer(pan)
            view.addGestureRecognizer(
                UIPinchGestureRecognizer(target: self, action: #selector(handlePinch(_:)))
            )
            let reset = UITapGestureRecognizer(target: self, action: #selector(resetView))
            reset.numberOfTapsRequired = 2
            view.addGestureRecognizer(reset)
        }

        @objc private func handlePan(_ gesture: UIPanGestureRecognizer) {
            guard let view, let camera else { return }
            let delta = gesture.translation(in: view)
            gesture.setTranslation(.zero, in: view)
            yaw -= Float(delta.x) * 0.006
            pitch = min(1.25, max(-1.25, pitch - Float(delta.y) * 0.0045))
            camera.eulerAngles = SCNVector3(pitch, yaw, 0)
            view.setNeedsDisplay()
        }

        @objc private func handlePinch(_ gesture: UIPinchGestureRecognizer) {
            guard let view, let camera = camera?.camera else { return }
            if gesture.state == .began {
                pinchStartFieldOfView = camera.fieldOfView
            }
            camera.fieldOfView = min(85, max(38, pinchStartFieldOfView / gesture.scale))
            view.setNeedsDisplay()
        }

        @objc private func resetView() {
            guard let view, let camera else { return }
            camera.position = initialPosition
            camera.eulerAngles = initialEulerAngles
            camera.camera?.fieldOfView = initialFieldOfView
            yaw = initialEulerAngles.y
            pitch = initialEulerAngles.x
            view.setNeedsDisplay()
        }
    }
}
