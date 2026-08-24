import Foundation

public actor ScanBundleWriter {
    public let bundleURL: URL
    private let framesURL: URL
    private var frameRecords: [ScanFrame] = []

    public init(outputDirectory: URL, scanID: UUID, fileManager: FileManager = .default) throws {
        bundleURL = outputDirectory.appendingPathComponent("\(scanID.uuidString).scanbundle", isDirectory: true)
        framesURL = bundleURL.appendingPathComponent("frames", isDirectory: true)
        try fileManager.createDirectory(at: framesURL, withIntermediateDirectories: true)
    }

    public func append(jpegData: Data, frame: ScanFrame) throws {
        guard frame.id == frameRecords.count + 1 else {
            throw ScanBundleContractError.invalidFrameID
        }
        let filename = String(format: "%06d.jpg", frame.id)
        try jpegData.write(to: framesURL.appendingPathComponent(filename), options: .atomic)
        frameRecords.append(frame)
    }

    @discardableResult
    public func finalize(scanID: UUID, createdAt: String, device: ScanDevice) throws -> URL {
        let manifest = ScanManifest(scanID: scanID, createdAt: createdAt, device: device, frames: frameRecords)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let data = try encoder.encode(manifest)
        try data.write(to: bundleURL.appendingPathComponent("manifest.json"), options: .atomic)
        return bundleURL
    }
}
