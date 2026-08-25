import Foundation
import SceneKit
import UIKit

struct PLYMeshBuffers: Sendable {
    let positions: [Float]
    let normals: [Float]
    let colors: [UInt8]
    let indices: [UInt32]
    let vertexCount: Int
    let triangleCount: Int
    let minimum: SIMD3<Float>
    let maximum: SIMD3<Float>

    @MainActor
    func makeGeometry(floorTriangleCount: Int? = nil) -> SCNGeometry {
        let positionData = positions.withUnsafeBytes { Data($0) }
        let normalData = normals.withUnsafeBytes { Data($0) }
        let colorData = colors.withUnsafeBytes { Data($0) }
        let indexData = indices.withUnsafeBytes { Data($0) }
        let usesFloorAndWallMaterials = {
            guard let floorTriangleCount else { return false }
            return floorTriangleCount > 0 && floorTriangleCount < triangleCount
        }()
        var sources = [
            SCNGeometrySource(
                data: positionData,
                semantic: .vertex,
                vectorCount: vertexCount,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: 4,
                dataOffset: 0,
                dataStride: 12
            ),
            SCNGeometrySource(
                data: normalData,
                semantic: .normal,
                vectorCount: vertexCount,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: 4,
                dataOffset: 0,
                dataStride: 12
            ),
        ]
        if !usesFloorAndWallMaterials {
            sources.append(SCNGeometrySource(
                data: colorData,
                semantic: .color,
                vectorCount: vertexCount,
                usesFloatComponents: false,
                componentsPerVector: 4,
                bytesPerComponent: 1,
                dataOffset: 0,
                dataStride: 4
            ))
        }
        let elements: [SCNGeometryElement]
        if usesFloorAndWallMaterials, let floorTriangleCount {
            let floorIndexCount = floorTriangleCount * 3
            let floorData = indices.prefix(floorIndexCount).withUnsafeBytes { Data($0) }
            let wallData = indices.dropFirst(floorIndexCount).withUnsafeBytes { Data($0) }
            elements = [
                SCNGeometryElement(
                    data: floorData,
                    primitiveType: .triangles,
                    primitiveCount: floorTriangleCount,
                    bytesPerIndex: 4
                ),
                SCNGeometryElement(
                    data: wallData,
                    primitiveType: .triangles,
                    primitiveCount: triangleCount - floorTriangleCount,
                    bytesPerIndex: 4
                ),
            ]
        } else {
            elements = [SCNGeometryElement(
                data: indexData,
                primitiveType: .triangles,
                primitiveCount: triangleCount,
                bytesPerIndex: 4
            )]
        }
        let geometry = SCNGeometry(sources: sources, elements: elements)
        if elements.count == 2 {
            let floorMaterial = SCNMaterial()
            floorMaterial.name = "warm-floor"
            floorMaterial.diffuse.contents = UIColor(red: 0.34, green: 0.22, blue: 0.13, alpha: 1)
            floorMaterial.lightingModel = .physicallyBased
            floorMaterial.roughness.contents = 0.72
            floorMaterial.metalness.contents = 0.0
            floorMaterial.isDoubleSided = true
            let wallMaterial = SCNMaterial()
            wallMaterial.name = "cool-wall"
            wallMaterial.diffuse.contents = UIColor(red: 0.56, green: 0.72, blue: 0.84, alpha: 1)
            wallMaterial.lightingModel = .physicallyBased
            wallMaterial.roughness.contents = 0.58
            wallMaterial.metalness.contents = 0.0
            wallMaterial.isDoubleSided = true
            geometry.materials = [floorMaterial, wallMaterial]
        } else {
            let material = SCNMaterial()
            material.diffuse.contents = UIColor.white
            material.lightingModel = .physicallyBased
            material.isDoubleSided = true
            material.roughness.contents = 0.9
            geometry.materials = [material]
        }
        return geometry
    }
}

enum PLYMeshError: LocalizedError {
    case unsupported(String)
    case truncated

    var errorDescription: String? {
        switch self {
        case .unsupported(let detail): "暂不支持这个 PLY：\(detail)"
        case .truncated: "PLY 模型数据不完整"
        }
    }
}

enum PLYMeshParser {
    private static let headerTerminator = Data("end_header\n".utf8)

    static func parse(_ data: Data) throws -> PLYMeshBuffers {
        guard let terminatorRange = data.range(of: headerTerminator),
              let header = String(data: data[..<terminatorRange.upperBound], encoding: .ascii) else {
            throw PLYMeshError.unsupported("缺少有效文件头")
        }
        guard header.contains("format binary_little_endian 1.0"),
              header.contains("property double x\nproperty double y\nproperty double z"),
              header.contains("property double nx\nproperty double ny\nproperty double nz"),
              header.contains("property uchar red\nproperty uchar green\nproperty uchar blue"),
              header.contains("property list uchar uint vertex_indices") else {
            throw PLYMeshError.unsupported("顶点布局与 Fosmo 导出格式不一致")
        }
        guard let vertexCount = elementCount(named: "vertex", in: header),
              let faceCount = elementCount(named: "face", in: header),
              vertexCount > 0,
              faceCount > 0 else {
            throw PLYMeshError.unsupported("缺少顶点或三角面")
        }

        let vertexStride = 51
        let faceStride = 13
        let payloadOffset = terminatorRange.upperBound
        let expectedBytes = payloadOffset + vertexCount * vertexStride + faceCount * faceStride
        guard data.count >= expectedBytes else { throw PLYMeshError.truncated }

        var positions = [Float]()
        var normals = [Float]()
        var colors = [UInt8]()
        var indices = [UInt32]()
        positions.reserveCapacity(vertexCount * 3)
        normals.reserveCapacity(vertexCount * 3)
        colors.reserveCapacity(vertexCount * 4)
        indices.reserveCapacity(faceCount * 3)
        var minimum = SIMD3<Float>(repeating: .greatestFiniteMagnitude)
        var maximum = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        try data.withUnsafeBytes { bytes in
            var offset = payloadOffset
            for _ in 0..<vertexCount {
                let x = Float(readDouble(bytes, at: offset))
                let y = Float(readDouble(bytes, at: offset + 8))
                let z = Float(readDouble(bytes, at: offset + 16))
                positions.append(contentsOf: [x, y, z])
                normals.append(contentsOf: [
                    Float(readDouble(bytes, at: offset + 24)),
                    Float(readDouble(bytes, at: offset + 32)),
                    Float(readDouble(bytes, at: offset + 40)),
                ])
                colors.append(contentsOf: [bytes[offset + 48], bytes[offset + 49], bytes[offset + 50], 255])
                let point = SIMD3<Float>(x, y, z)
                minimum = simd_min(minimum, point)
                maximum = simd_max(maximum, point)
                offset += vertexStride
            }
            for _ in 0..<faceCount {
                guard bytes[offset] == 3 else {
                    throw PLYMeshError.unsupported("模型包含非三角面")
                }
                indices.append(readUInt32(bytes, at: offset + 1))
                indices.append(readUInt32(bytes, at: offset + 5))
                indices.append(readUInt32(bytes, at: offset + 9))
                offset += faceStride
            }
        }
        guard indices.allSatisfy({ $0 < UInt32(vertexCount) }) else {
            throw PLYMeshError.unsupported("三角面索引越界")
        }
        return PLYMeshBuffers(
            positions: positions,
            normals: normals,
            colors: colors,
            indices: indices,
            vertexCount: vertexCount,
            triangleCount: faceCount,
            minimum: minimum,
            maximum: maximum
        )
    }

    private static func elementCount(named name: String, in header: String) -> Int? {
        header.split(separator: "\n").first { $0.hasPrefix("element \(name) ") }
            .flatMap { Int($0.split(separator: " ").last ?? "") }
    }

    private static func readUInt32(_ bytes: UnsafeRawBufferPointer, at offset: Int) -> UInt32 {
        UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }

    private static func readDouble(_ bytes: UnsafeRawBufferPointer, at offset: Int) -> Double {
        let bits = UInt64(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt64.self))
        return Double(bitPattern: bits)
    }
}
