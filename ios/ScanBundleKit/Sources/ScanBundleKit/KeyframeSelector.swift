import Foundation

public struct KeyframePolicy: Equatable, Sendable {
    public var minimumInterval: TimeInterval
    public var minimumTranslationMeters: Double
    public var minimumRotationRadians: Double
    public var minimumSharpness: Double
    public var acceptableLuminance: ClosedRange<Double>

    public init(
        minimumInterval: TimeInterval = 0.45,
        minimumTranslationMeters: Double = 0.08,
        minimumRotationRadians: Double = 8 * .pi / 180,
        minimumSharpness: Double = 0.12,
        acceptableLuminance: ClosedRange<Double> = 0.08...0.92
    ) {
        self.minimumInterval = minimumInterval
        self.minimumTranslationMeters = minimumTranslationMeters
        self.minimumRotationRadians = minimumRotationRadians
        self.minimumSharpness = minimumSharpness
        self.acceptableLuminance = acceptableLuminance
    }
}

public struct KeyframeCandidate: Equatable, Sendable {
    public let timestamp: TimeInterval
    public let worldFromCamera: RowMajorMatrix4x4
    public let trackingIsNormal: Bool
    public let sharpness: Double
    public let meanLuminance: Double

    public init(
        timestamp: TimeInterval,
        worldFromCamera: RowMajorMatrix4x4,
        trackingIsNormal: Bool,
        sharpness: Double,
        meanLuminance: Double
    ) {
        self.timestamp = timestamp
        self.worldFromCamera = worldFromCamera
        self.trackingIsNormal = trackingIsNormal
        self.sharpness = sharpness
        self.meanLuminance = meanLuminance
    }
}

public enum KeyframeRejection: Equatable, Sendable {
    case trackingNotNormal
    case tooSoon
    case blurry
    case exposureOutOfRange
    case insufficientMotion
}

public enum KeyframeDecision: Equatable, Sendable {
    case accept
    case reject(KeyframeRejection)
}

/// Deterministic, explainable keyframe selection for ScanBundle 1.0.
public struct KeyframeSelector: Sendable {
    public var policy: KeyframePolicy
    private var lastAccepted: KeyframeCandidate?

    public init(policy: KeyframePolicy = KeyframePolicy()) {
        self.policy = policy
    }

    public mutating func evaluate(_ candidate: KeyframeCandidate) -> KeyframeDecision {
        guard candidate.trackingIsNormal else { return .reject(.trackingNotNormal) }
        guard candidate.sharpness.isFinite, candidate.sharpness >= policy.minimumSharpness else {
            return .reject(.blurry)
        }
        guard candidate.meanLuminance.isFinite, policy.acceptableLuminance.contains(candidate.meanLuminance) else {
            return .reject(.exposureOutOfRange)
        }

        if let lastAccepted {
            guard candidate.timestamp - lastAccepted.timestamp >= policy.minimumInterval else {
                return .reject(.tooSoon)
            }
            let translation = Self.translationDistance(
                from: lastAccepted.worldFromCamera,
                to: candidate.worldFromCamera
            )
            let rotation = Self.rotationAngle(
                from: lastAccepted.worldFromCamera,
                to: candidate.worldFromCamera
            )
            guard translation >= policy.minimumTranslationMeters || rotation >= policy.minimumRotationRadians else {
                return .reject(.insufficientMotion)
            }
        }

        lastAccepted = candidate
        return .accept
    }

    static func translationDistance(from left: RowMajorMatrix4x4, to right: RowMajorMatrix4x4) -> Double {
        let dx = right.values[3] - left.values[3]
        let dy = right.values[7] - left.values[7]
        let dz = right.values[11] - left.values[11]
        return (dx * dx + dy * dy + dz * dz).squareRoot()
    }

    static func rotationAngle(from left: RowMajorMatrix4x4, to right: RowMajorMatrix4x4) -> Double {
        let rotationIndices = [0, 1, 2, 4, 5, 6, 8, 9, 10]
        let traceOfRelative = rotationIndices.reduce(0.0) { result, index in
            result + left.values[index] * right.values[index]
        }
        let cosine = max(-1.0, min(1.0, (traceOfRelative - 1.0) / 2.0))
        return acos(cosine)
    }
}
