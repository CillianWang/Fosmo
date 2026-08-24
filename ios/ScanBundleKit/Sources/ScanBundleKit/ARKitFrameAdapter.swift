#if os(iOS) && canImport(ARKit)
import ARKit
import CoreImage
import Foundation

public enum ARKitFrameAdapterError: Error, Equatable {
    case trackingIsNotNormal
    case jpegEncodingFailed
}

public struct CapturedScanFrame: Sendable {
    public let jpegData: Data
    public let frame: ScanFrame

    public init(jpegData: Data, frame: ScanFrame) {
        self.jpegData = jpegData
        self.frame = frame
    }
}

/// Converts an ARKit frame without rotating, cropping, mirroring, or resizing it.
public enum ARKitFrameAdapter {
    public static func candidate(
        from arFrame: ARFrame,
        sharpness: Double,
        meanLuminance: Double
    ) throws -> KeyframeCandidate {
        let transform = arFrame.camera.transform
        return KeyframeCandidate(
            timestamp: arFrame.timestamp,
            worldFromCamera: try RowMajorMatrix4x4(rows: (0..<4).map { row in
                (0..<4).map { column in Double(transform[column][row]) }
            }),
            trackingIsNormal: {
                if case .normal = arFrame.camera.trackingState { return true }
                return false
            }(),
            sharpness: sharpness,
            meanLuminance: meanLuminance
        )
    }

    public static func capture(
        _ arFrame: ARFrame,
        id: Int,
        context: CIContext = CIContext(options: [.cacheIntermediates: false]),
        compressionQuality: Double = 0.92
    ) throws -> CapturedScanFrame {
        guard case .normal = arFrame.camera.trackingState else {
            throw ARKitFrameAdapterError.trackingIsNotNormal
        }

        let pixelBuffer = arFrame.capturedImage
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let colorSpace = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
        guard let jpegData = context.jpegRepresentation(
            of: image,
            colorSpace: colorSpace,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: compressionQuality]
        ) else {
            throw ARKitFrameAdapterError.jpegEncodingFailed
        }

        let intrinsics = arFrame.camera.intrinsics
        let transform = arFrame.camera.transform
        let frame = try ScanFrame(
            id: id,
            timestamp: arFrame.timestamp,
            imageWidth: CVPixelBufferGetWidth(pixelBuffer),
            imageHeight: CVPixelBufferGetHeight(pixelBuffer),
            calibrationWidth: Int(arFrame.camera.imageResolution.width),
            calibrationHeight: Int(arFrame.camera.imageResolution.height),
            intrinsics: RowMajorMatrix3x3(rows: (0..<3).map { row in
                (0..<3).map { column in Double(intrinsics[column][row]) }
            }),
            worldFromCamera: RowMajorMatrix4x4(rows: (0..<4).map { row in
                (0..<4).map { column in Double(transform[column][row]) }
            })
        )
        return CapturedScanFrame(jpegData: jpegData, frame: frame)
    }
}
#endif
