// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "ScanBundleKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "ScanBundleKit", targets: ["ScanBundleKit"]),
    ],
    targets: [
        .target(name: "ScanBundleKit"),
        .testTarget(name: "ScanBundleKitTests", dependencies: ["ScanBundleKit"]),
    ]
)
