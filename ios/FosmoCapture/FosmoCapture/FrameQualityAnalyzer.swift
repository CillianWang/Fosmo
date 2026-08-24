import CoreVideo
import Foundation

struct FrameQuality: Sendable {
    let sharpness: Double
    let meanLuminance: Double
}

enum FrameQualityAnalyzer {
    static func analyze(_ pixelBuffer: CVPixelBuffer, sampleStride: Int = 8) -> FrameQuality {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let plane = CVPixelBufferGetPlaneCount(pixelBuffer) > 0 ? 0 : -1
        let width = plane >= 0 ? CVPixelBufferGetWidthOfPlane(pixelBuffer, plane) : CVPixelBufferGetWidth(pixelBuffer)
        let height = plane >= 0 ? CVPixelBufferGetHeightOfPlane(pixelBuffer, plane) : CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = plane >= 0
            ? CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, plane)
            : CVPixelBufferGetBytesPerRow(pixelBuffer)
        let rawBase = plane >= 0
            ? CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, plane)
            : CVPixelBufferGetBaseAddress(pixelBuffer)

        guard let rawBase, width > sampleStride, height > sampleStride else {
            return FrameQuality(sharpness: 0, meanLuminance: 0.5)
        }

        let base = rawBase.assumingMemoryBound(to: UInt8.self)
        var luminanceTotal = 0.0
        var gradientTotal = 0.0
        var sampleCount = 0

        for y in stride(from: sampleStride, to: height, by: sampleStride) {
            let row = base.advanced(by: y * bytesPerRow)
            let previousRow = base.advanced(by: (y - sampleStride) * bytesPerRow)
            for x in stride(from: sampleStride, to: width, by: sampleStride) {
                let value = Double(row[x])
                luminanceTotal += value
                gradientTotal += abs(value - Double(row[x - sampleStride]))
                gradientTotal += abs(value - Double(previousRow[x]))
                sampleCount += 1
            }
        }

        guard sampleCount > 0 else {
            return FrameQuality(sharpness: 0, meanLuminance: 0.5)
        }
        return FrameQuality(
            sharpness: gradientTotal / Double(sampleCount * 2) / 255.0,
            meanLuminance: luminanceTotal / Double(sampleCount) / 255.0
        )
    }
}
