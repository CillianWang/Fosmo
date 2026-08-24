import Foundation

public enum ScanBundleContractError: Error, Equatable {
    case invalidMatrixSize(expected: Int, actual: Int)
    case inconsistentRowSize(expected: Int, actual: Int)
    case nonFiniteMatrixValue
    case invalidFrameID
    case invalidImageFilename
}

public struct RowMajorMatrix3x3: Codable, Equatable, Sendable {
    public let values: [Double]

    public init(values: [Double]) throws {
        guard values.count == 9 else {
            throw ScanBundleContractError.invalidMatrixSize(expected: 9, actual: values.count)
        }
        guard values.allSatisfy(\.isFinite) else {
            throw ScanBundleContractError.nonFiniteMatrixValue
        }
        self.values = values
    }

    public init(rows: [[Double]]) throws {
        guard rows.count == 3 else {
            throw ScanBundleContractError.invalidMatrixSize(expected: 3, actual: rows.count)
        }
        guard let invalidRow = rows.first(where: { $0.count != 3 }) else {
            try self.init(values: rows.flatMap { $0 })
            return
        }
        throw ScanBundleContractError.inconsistentRowSize(expected: 3, actual: invalidRow.count)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(values: container.decode([Double].self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }
}

public struct RowMajorMatrix4x4: Codable, Equatable, Sendable {
    public let values: [Double]

    public init(values: [Double]) throws {
        guard values.count == 16 else {
            throw ScanBundleContractError.invalidMatrixSize(expected: 16, actual: values.count)
        }
        guard values.allSatisfy(\.isFinite) else {
            throw ScanBundleContractError.nonFiniteMatrixValue
        }
        self.values = values
    }

    public init(rows: [[Double]]) throws {
        guard rows.count == 4 else {
            throw ScanBundleContractError.invalidMatrixSize(expected: 4, actual: rows.count)
        }
        guard let invalidRow = rows.first(where: { $0.count != 4 }) else {
            try self.init(values: rows.flatMap { $0 })
            return
        }
        throw ScanBundleContractError.inconsistentRowSize(expected: 4, actual: invalidRow.count)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        try self.init(values: container.decode([Double].self))
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(values)
    }
}

public struct ScanDevice: Codable, Equatable, Sendable {
    public let model: String
    public let osVersion: String

    public init(model: String, osVersion: String) {
        self.model = model
        self.osVersion = osVersion
    }

    enum CodingKeys: String, CodingKey {
        case model
        case osVersion = "os_version"
    }
}

public struct CoordinateSystem: Codable, Equatable, Sendable {
    public let pose: String
    public let matrixLayout: String
    public let units: String

    public init(
        pose: String = "ARKit worldFromCamera",
        matrixLayout: String = "row-major",
        units: String = "meters"
    ) {
        self.pose = pose
        self.matrixLayout = matrixLayout
        self.units = units
    }

    enum CodingKeys: String, CodingKey {
        case pose
        case matrixLayout = "matrix_layout"
        case units
    }
}

public struct ScanFrame: Codable, Equatable, Sendable {
    public let id: Int
    public let image: String
    public let timestamp: Double
    public let imageWidth: Int
    public let imageHeight: Int
    public let calibrationWidth: Int
    public let calibrationHeight: Int
    public let intrinsics: RowMajorMatrix3x3
    public let worldFromCamera: RowMajorMatrix4x4
    public let trackingState: String
    public let trackingReason: String?
    public let imageOrientation: String
    public let imageFormatVersion: String

    public init(
        id: Int,
        timestamp: Double,
        imageWidth: Int,
        imageHeight: Int,
        calibrationWidth: Int,
        calibrationHeight: Int,
        intrinsics: RowMajorMatrix3x3,
        worldFromCamera: RowMajorMatrix4x4
    ) throws {
        guard id > 0 else { throw ScanBundleContractError.invalidFrameID }
        let image = String(format: "frames/%06d.jpg", id)
        guard image.utf8.count == 17 else { throw ScanBundleContractError.invalidImageFilename }
        self.id = id
        self.image = image
        self.timestamp = timestamp
        self.imageWidth = imageWidth
        self.imageHeight = imageHeight
        self.calibrationWidth = calibrationWidth
        self.calibrationHeight = calibrationHeight
        self.intrinsics = intrinsics
        self.worldFromCamera = worldFromCamera
        self.trackingState = "normal"
        self.trackingReason = nil
        self.imageOrientation = "sensor-native"
        self.imageFormatVersion = "jpeg-v1"
    }

    enum CodingKeys: String, CodingKey {
        case id
        case image
        case timestamp
        case imageWidth = "image_width"
        case imageHeight = "image_height"
        case calibrationWidth = "calibration_width"
        case calibrationHeight = "calibration_height"
        case intrinsics
        case worldFromCamera = "world_from_camera"
        case trackingState = "tracking_state"
        case trackingReason = "tracking_reason"
        case imageOrientation = "image_orientation"
        case imageFormatVersion = "image_format_version"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(image, forKey: .image)
        try container.encode(timestamp, forKey: .timestamp)
        try container.encode(imageWidth, forKey: .imageWidth)
        try container.encode(imageHeight, forKey: .imageHeight)
        try container.encode(calibrationWidth, forKey: .calibrationWidth)
        try container.encode(calibrationHeight, forKey: .calibrationHeight)
        try container.encode(intrinsics, forKey: .intrinsics)
        try container.encode(worldFromCamera, forKey: .worldFromCamera)
        try container.encode(trackingState, forKey: .trackingState)
        if let trackingReason {
            try container.encode(trackingReason, forKey: .trackingReason)
        } else {
            try container.encodeNil(forKey: .trackingReason)
        }
        try container.encode(imageOrientation, forKey: .imageOrientation)
        try container.encode(imageFormatVersion, forKey: .imageFormatVersion)
    }
}

public struct ScanManifest: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let scanID: UUID
    public let createdAt: String
    public let device: ScanDevice
    public let coordinateSystem: CoordinateSystem
    public let frames: [ScanFrame]

    public init(scanID: UUID, createdAt: String, device: ScanDevice, frames: [ScanFrame]) {
        self.schemaVersion = "1.0"
        self.scanID = scanID
        self.createdAt = createdAt
        self.device = device
        self.coordinateSystem = CoordinateSystem()
        self.frames = frames
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case scanID = "scan_id"
        case createdAt = "created_at"
        case device
        case coordinateSystem = "coordinate_system"
        case frames
    }
}
