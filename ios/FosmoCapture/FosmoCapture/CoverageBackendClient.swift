import Foundation
import ScanBundleKit

struct CoverageStatusResponse: Decodable, Sendable {
    let sessionID: String?
    let complete: Bool
    let acceptedFrameCount: Int
    let rejectedFrameCount: Int
    let sectorsCovered: Int
    let sectorsRequired: Int
    let elapsedSeconds: Double
    let missingSectorCentersDegrees: [Double]
    let message: String
    let error: String?

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case complete
        case acceptedFrameCount = "accepted_frame_count"
        case rejectedFrameCount = "rejected_frame_count"
        case sectorsCovered = "sectors_covered"
        case sectorsRequired = "sectors_required"
        case elapsedSeconds = "elapsed_seconds"
        case missingSectorCentersDegrees = "missing_sector_centers_degrees"
        case message
        case error
    }
}

struct CoverageFramePayload: Encodable, Sendable {
    let frameID: Int
    let timestamp: Double
    let worldFromCamera: [Double]
    let sharpness: Double
    let meanLuminance: Double
    let thumbnailBase64: String

    enum CodingKeys: String, CodingKey {
        case frameID = "frame_id"
        case timestamp
        case worldFromCamera = "world_from_camera"
        case sharpness
        case meanLuminance = "mean_luminance"
        case thumbnailBase64 = "thumbnail_base64"
    }
}

enum CoverageBackendError: LocalizedError {
    case invalidURL
    case invalidResponse
    case rejected(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL: "后端地址无效"
        case .invalidResponse: "后端返回了无法识别的响应"
        case .rejected(let message): "后端拒绝了当前图像：\(message)"
        }
    }
}

actor CoverageBackendClient {
    private let baseURL: URL
    private let session: URLSession
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(baseURL: URL) {
        self.baseURL = baseURL
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 5
        configuration.timeoutIntervalForResource = 10
        self.session = URLSession(configuration: configuration)
    }

    func createSession() async throws -> CoverageStatusResponse {
        var request = URLRequest(url: baseURL.appendingPathComponent("coverage/sessions"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("{}".utf8)
        return try await send(request, acceptedStatusCodes: [201])
    }

    func submit(
        sessionID: String,
        frame: ScanFrame,
        quality: FrameQuality,
        thumbnailJPEG: Data
    ) async throws -> CoverageStatusResponse {
        let payload = CoverageFramePayload(
            frameID: frame.id,
            timestamp: frame.timestamp,
            worldFromCamera: frame.worldFromCamera.values,
            sharpness: quality.sharpness,
            meanLuminance: quality.meanLuminance,
            thumbnailBase64: thumbnailJPEG.base64EncodedString()
        )
        let url = baseURL
            .appendingPathComponent("coverage/sessions")
            .appendingPathComponent(sessionID)
            .appendingPathComponent("frames")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try encoder.encode(payload)
        return try await send(request, acceptedStatusCodes: [200])
    }

    private func send(
        _ request: URLRequest,
        acceptedStatusCodes: Set<Int>
    ) async throws -> CoverageStatusResponse {
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw CoverageBackendError.invalidResponse
        }
        let decoded = try decoder.decode(CoverageStatusResponse.self, from: data)
        guard acceptedStatusCodes.contains(httpResponse.statusCode) else {
            throw CoverageBackendError.rejected(decoded.error ?? decoded.message)
        }
        return decoded
    }
}
