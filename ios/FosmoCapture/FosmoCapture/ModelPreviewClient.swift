import Foundation

struct ModelPreviewManifest: Decodable, Sendable {
    let scanID: String
    let frameCount: Int
    let vertexCount: Int
    let triangleCount: Int
    let modelBytes: Int
    let modelURL: String
    let modelKind: String?
    let dimensionsMeters: [Double]?
    let heightMeters: Double?

    enum CodingKeys: String, CodingKey {
        case scanID = "scan_id"
        case frameCount = "frame_count"
        case vertexCount = "vertex_count"
        case triangleCount = "triangle_count"
        case modelBytes = "model_bytes"
        case modelURL = "model_url"
        case modelKind = "model_kind"
        case dimensionsMeters = "dimensions_meters"
        case heightMeters = "height_meters"
    }
}

enum ModelPreviewError: LocalizedError {
    case invalidBackendURL
    case invalidResponse
    case server(String)

    var errorDescription: String? {
        switch self {
        case .invalidBackendURL: "后端地址无效"
        case .invalidResponse: "后端返回了无法识别的模型数据"
        case .server(let message): message
        }
    }
}

actor ModelPreviewClient {
    private let baseURL: URL
    private let session: URLSession

    init(baseURL: URL) {
        self.baseURL = baseURL
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 120
        session = URLSession(configuration: configuration)
    }

    func downloadPreview() async throws -> (ModelPreviewManifest, Data) {
        let manifestURL = baseURL.appendingPathComponent("preview/manifest")
        let (manifestData, manifestResponse) = try await session.data(from: manifestURL)
        try Self.validate(manifestResponse, data: manifestData)
        let manifest = try JSONDecoder().decode(ModelPreviewManifest.self, from: manifestData)

        guard let modelURL = URL(string: manifest.modelURL, relativeTo: baseURL)?.absoluteURL else {
            throw ModelPreviewError.invalidResponse
        }
        let (modelData, modelResponse) = try await session.data(from: modelURL)
        try Self.validate(modelResponse, data: modelData)
        guard modelData.count == manifest.modelBytes else {
            throw ModelPreviewError.invalidResponse
        }
        return (manifest, modelData)
    }

    private static func validate(_ response: URLResponse, data: Data) throws {
        guard let response = response as? HTTPURLResponse else {
            throw ModelPreviewError.invalidResponse
        }
        guard (200..<300).contains(response.statusCode) else {
            let message = (try? JSONSerialization.jsonObject(with: data) as? [String: String])?["error"]
            throw ModelPreviewError.server(message ?? "模型服务返回 HTTP \(response.statusCode)")
        }
    }
}
