import SceneKit
import SwiftUI

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
            scene = Self.makeScene(from: buffers)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private static func makeScene(from buffers: PLYMeshBuffers) -> SCNScene {
        let scene = SCNScene()
        let modelNode = SCNNode(geometry: buffers.makeGeometry())
        scene.rootNode.addChildNode(modelNode)

        let centerVector = (buffers.minimum + buffers.maximum) / 2
        let extent = buffers.maximum - buffers.minimum
        let radius = max(0.25, simd_length(extent) / 2)
        let targetNode = SCNNode()
        targetNode.position = SCNVector3(centerVector.x, centerVector.y, centerVector.z)
        scene.rootNode.addChildNode(targetNode)

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.zNear = Double(max(0.01, radius / 100))
        cameraNode.camera?.zFar = Double(radius * 20)
        cameraNode.position = SCNVector3(
            centerVector.x + radius * 0.25,
            centerVector.y + radius * 0.35,
            centerVector.z + radius * 1.7
        )
        let lookAt = SCNLookAtConstraint(target: targetNode)
        lookAt.isGimbalLockEnabled = true
        cameraNode.constraints = [lookAt]
        scene.rootNode.addChildNode(cameraNode)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 500
        ambient.light?.color = UIColor(white: 0.75, alpha: 1)
        scene.rootNode.addChildNode(ambient)

        let directional = SCNNode()
        directional.light = SCNLight()
        directional.light?.type = .directional
        directional.light?.intensity = 900
        directional.eulerAngles = SCNVector3(-0.8, 0.6, 0)
        scene.rootNode.addChildNode(directional)
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
                        Text("\(manifest.triangleCount.formatted()) 面")
                            .font(.caption.monospacedDigit().weight(.semibold))
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                }
                Spacer()
                if model.scene != nil {
                    Text("单指旋转 · 双指缩放 · 双指平移")
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
}

private struct SceneKitPreview: UIViewRepresentable {
    let scene: SCNScene

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.scene = scene
        view.pointOfView = scene.rootNode.childNodes.first { $0.camera != nil }
        view.backgroundColor = UIColor(red: 0.025, green: 0.03, blue: 0.04, alpha: 1)
        view.allowsCameraControl = true
        view.antialiasingMode = .multisampling4X
        view.rendersContinuously = false
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        view.scene = scene
    }
}

