import Foundation
import Testing
@testable import ScanBundleKit

@Test func matricesFlattenRowsExplicitly() throws {
    let intrinsics = try RowMajorMatrix3x3(rows: [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])
    let pose = try RowMajorMatrix4x4(rows: [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ])

    #expect(intrinsics.values == [1, 2, 3, 4, 5, 6, 7, 8, 9])
    #expect(pose.values == Array(1...16).map(Double.init))
}

@Test func manifestUsesFrozenSnakeCaseContract() throws {
    let intrinsics = try RowMajorMatrix3x3(values: [3, 0, 2, 0, 3, 1.5, 0, 0, 1])
    let pose = try RowMajorMatrix4x4(values: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    let frame = try ScanFrame(
        id: 1,
        timestamp: 0,
        imageWidth: 4,
        imageHeight: 3,
        calibrationWidth: 4,
        calibrationHeight: 3,
        intrinsics: intrinsics,
        worldFromCamera: pose
    )
    let scanID = try #require(UUID(uuidString: "d302af50-f183-4bdd-a537-878d24163e3f"))
    let manifest = ScanManifest(
        scanID: scanID,
        createdAt: "2026-08-24T10:00:00+08:00",
        device: ScanDevice(model: "iPhone", osVersion: "iOS 26.0"),
        frames: [frame]
    )

    let data = try JSONEncoder().encode(manifest)
    let json = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let encodedFrame = try #require((json["frames"] as? [[String: Any]])?.first)

    #expect(json["schema_version"] as? String == "1.0")
    #expect((json["coordinate_system"] as? [String: Any])?["matrix_layout"] as? String == "row-major")
    #expect(encodedFrame["image"] as? String == "frames/000001.jpg")
    #expect(encodedFrame["world_from_camera"] as? [Double] == pose.values)
    #expect(encodedFrame.keys.contains("tracking_reason"))
    #expect(encodedFrame["tracking_reason"] is NSNull)
}

@Test func writerCreatesVersionedBundleLayout() async throws {
    let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: output) }
    let scanID = UUID()
    let writer = try ScanBundleWriter(outputDirectory: output, scanID: scanID)
    let intrinsics = try RowMajorMatrix3x3(values: [3, 0, 2, 0, 3, 1.5, 0, 0, 1])
    let pose = try RowMajorMatrix4x4(values: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    let frame = try ScanFrame(
        id: 1,
        timestamp: 0,
        imageWidth: 4,
        imageHeight: 3,
        calibrationWidth: 4,
        calibrationHeight: 3,
        intrinsics: intrinsics,
        worldFromCamera: pose
    )

    try await writer.append(jpegData: Data([0xFF, 0xD8, 0xFF, 0xD9]), frame: frame)
    let bundle = try await writer.finalize(
        createdAt: "2026-08-24T10:00:00+08:00",
        device: ScanDevice(model: "iPhone", osVersion: "iOS 26.0")
    )

    #expect(FileManager.default.fileExists(atPath: bundle.appendingPathComponent("manifest.json").path))
    #expect(FileManager.default.fileExists(atPath: bundle.appendingPathComponent("frames/000001.jpg").path))
}
