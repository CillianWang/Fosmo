import Foundation
import Testing
@testable import ScanBundleKit

private func pose(x: Double = 0, yawRadians: Double = 0) throws -> RowMajorMatrix4x4 {
    let cosine = cos(yawRadians)
    let sine = sin(yawRadians)
    return try RowMajorMatrix4x4(values: [
        cosine, 0, sine, x,
        0, 1, 0, 0,
        -sine, 0, cosine, 0,
        0, 0, 0, 1,
    ])
}

private func candidate(
    timestamp: Double,
    x: Double = 0,
    yawRadians: Double = 0,
    tracking: Bool = true,
    sharpness: Double = 0.5,
    luminance: Double = 0.5
) throws -> KeyframeCandidate {
    KeyframeCandidate(
        timestamp: timestamp,
        worldFromCamera: try pose(x: x, yawRadians: yawRadians),
        trackingIsNormal: tracking,
        sharpness: sharpness,
        meanLuminance: luminance
    )
}

@Test func firstGoodFrameIsAccepted() throws {
    var selector = KeyframeSelector()
    #expect(selector.evaluate(try candidate(timestamp: 0)) == .accept)
}

@Test func trackingAndImageQualityRejectBeforeMotion() throws {
    var selector = KeyframeSelector()
    #expect(selector.evaluate(try candidate(timestamp: 0, tracking: false)) == .reject(.trackingNotNormal))
    #expect(selector.evaluate(try candidate(timestamp: 0, sharpness: 0.01)) == .reject(.blurry))
    #expect(selector.evaluate(try candidate(timestamp: 0, luminance: 0.99)) == .reject(.exposureOutOfRange))
}

@Test func acceptedFramesNeedTimeAndTranslation() throws {
    var selector = KeyframeSelector()
    #expect(selector.evaluate(try candidate(timestamp: 0)) == .accept)
    #expect(selector.evaluate(try candidate(timestamp: 0.2, x: 0.2)) == .reject(.tooSoon))
    #expect(selector.evaluate(try candidate(timestamp: 0.6, x: 0.02)) == .reject(.insufficientMotion))
    #expect(selector.evaluate(try candidate(timestamp: 0.7, x: 0.09)) == .accept)
}

@Test func rotationCanAcceptAStationaryFrame() throws {
    var selector = KeyframeSelector()
    #expect(selector.evaluate(try candidate(timestamp: 0)) == .accept)
    #expect(selector.evaluate(try candidate(timestamp: 0.6, yawRadians: 10 * .pi / 180)) == .accept)
}
