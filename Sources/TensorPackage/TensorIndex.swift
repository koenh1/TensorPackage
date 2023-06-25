//
//  File.swift
//  
//
//  Created by Koen Hendrikx on 25/06/2023.
//

import Foundation

public typealias ScalarIndex = UInt32

public protocol TensorIteratorProtocol: IteratorProtocol where Element == ScalarIndex {
    associatedtype IndexType: TensorIndexProtocol
    var index: IndexType { get }
    init(size: IndexType, stride: IndexType)
    init(size: [ScalarIndex], stride: [ScalarIndex])

}
public protocol TensorIndexProtocol: Comparable, Hashable, Sequence where Element == ScalarIndex {
    static var scalarCount: Int { get }
    static var zero: Self { get }
    static var one: Self { get }
    init(_ values: [ScalarIndex])
    subscript(index: Int) -> ScalarIndex { get set }
    func inner(_ a: Self) -> ScalarIndex
    var product: ScalarIndex { get }
    var indices: Range<Int> { get }
    mutating func increment(size: Self) -> Bool
    mutating func swapAt(_ i: Int, _ j: Int)
}

public typealias TensorIndex = Comparable&Sequence&TensorIndexProtocol
extension SIMD2: Comparable where Scalar == ScalarIndex {}
extension SIMD2: Sequence where Scalar == ScalarIndex {}

extension SIMD2: TensorIndexProtocol where Scalar==ScalarIndex {}
extension SIMD3: Comparable where Scalar == ScalarIndex {}
extension SIMD3: Sequence where Scalar == ScalarIndex {}
extension SIMD3: TensorIndexProtocol where Scalar==ScalarIndex {}
extension SIMD4: Comparable where Scalar == ScalarIndex {}
extension SIMD4: Sequence where Scalar == ScalarIndex {}
extension SIMD4: TensorIndexProtocol where Scalar==ScalarIndex {}
extension SIMD8: Comparable where Scalar == ScalarIndex {}
extension SIMD8: Sequence where Scalar == ScalarIndex {}
extension SIMD8: TensorIndexProtocol where Scalar==ScalarIndex {}
public struct SIMDIterator<I: TensorIndexProtocol>: TensorIteratorProtocol {
    public typealias IndexType = I
    public var index: I = .zero
    let size: I
    let stride: I
    var last: Bool = false
    public init(size: I, stride: I) {
        self.size = size
        self.stride = stride
    }
    public init(size: [ScalarIndex], stride: [ScalarIndex]) {
        var s = size
        var t = stride
        if s.count < IndexType.scalarCount || t.count < IndexType.scalarCount {
            s.append(contentsOf: repeatElement(1, count: IndexType.scalarCount-s.count))
            t.append(contentsOf: repeatElement(0, count: IndexType.scalarCount-t.count))
        }
        self.init(size: IndexType(s), stride: IndexType(t))
    }
    mutating public func next(index j: Int) -> Bool {
        if last {
            return false
        }
        for i in Swift.stride(from: 0, to: j, by: 1) {
            index[i] = 0
        }
        for i in j..<IndexType.scalarCount {
            index[i] &+= 1
            if index[i] == size[i] {
                index[i] = 0
            } else {
                return true
            }
        }
        return false
    }
    mutating public func next() -> ScalarIndex? {
        if last {
            return nil
        }
        let t = index.inner(stride)
        if index.increment(size: size) {
            return t
        }
        last = true
        return t
    }
}

public struct SIMDPairIterator<S: TensorIndexProtocol>: IteratorProtocol {
    var index: S = .zero
    let size: S
    let stride1: S
    let stride2: S
    var last: Bool = false
    init(size: S, stride1: S, stride2: S) {
        self.size = size
        self.stride1 = stride1
        self.stride2 = stride2
    }
    mutating public func next() -> (ScalarIndex, ScalarIndex)? {
        if last {
            return nil
        }
        let t1 = index.inner(stride1)
        let t2 = index.inner(stride2)
        if !index.increment(size: size) {
            last = true
        }
        return (t1, t2)
    }
}

extension Shape: Sequence {
    public func makeIterator() -> SIMDIterator<IndexType> {
        .init(size: shape, stride: stride)
    }
    public func pairIterator(rhs: Self) -> SIMDPairIterator<IndexType> {
        if rhs.shape != shape {
            fatalError()
        }
        return .init(size: shape, stride1: stride, stride2: rhs.stride)
    }
}

public struct SIMDIndexingIterator<S: SIMD>: IteratorProtocol {
    var index: Int = 0
    let value: S
    mutating public func next() -> S.Scalar? {
        if index < S.scalarCount {
            defer {
                index += 1
            }
            return value[index]
        }
        return nil
    }
}
extension SIMD {
    public func makeIterator() -> SIMDIndexingIterator<Self> {
        .init(value: self)
    }
    mutating public func swapAt(_ i: Int, _ j: Int) {
        (self[i], self[j]) = (self[j], self[i])
    }
}
extension SIMD where Scalar: FixedWidthInteger {
    public var product: Scalar {
        var r: Scalar = 1
        self.indices.forEach {
            r *= self[$0]
        }
        return r
    }
    public func inner(_ s: Self) -> Scalar {
        (self &* s).wrappedSum()
    }
    public mutating func increment(size: Self) -> Bool {
        for i in indices {
            self[i] &+= 1
            if self[i] == size[i] {
                self[i] = 0
            } else {
                return true
            }
        }
        return false
    }
    public static func < (lhs: Self, rhs: Self) -> Bool {
        for i in lhs.indices.reversed() {
            if lhs[i] < rhs[i] {
                return true
            } else if lhs[i] > rhs[i] {
                return false
            }
        }
        return false
    }
}

public struct SIMD1: TensorIndexProtocol {
    public var indices: Range<Int> {
        0..<1
    }

    public mutating func swapAt(_ i: Int, _ j: Int) {
    }

    public var product: ScalarIndex {
        value
    }

    public init(_ values: [ScalarIndex]) {
        self.init(values.first!)
    }

    public func inner(_ a: SIMD1) -> ScalarIndex {
        value &* a.value
    }

    public mutating func increment(size: SIMD1) -> Bool {
        value &+= 1
        if value == size.value {
            value = 0
            return false
        }
        return true
    }

    public static var scalarCount: Int {
        1
    }
    public static var zero: SIMD1 {
        .init(.zero)
    }

    public static var one: SIMD1 {
        .init(1)
    }

    public typealias Element = ScalarIndex
    public init(_ value: ScalarIndex) {
        self.value = value
    }
    public init() {
        self.init(.zero)
    }

    public init(arrayLiteral elements: ScalarIndex...) {
        value = elements.first!
    }
    var value: ScalarIndex
    public subscript(index: Int) -> ScalarIndex {
        get {
            value
        }
        set {
            value = newValue
        }
    }
    public var scalarCount: Int {
        1
    }
    public func makeIterator() -> CollectionOfOne<ScalarIndex>.Iterator {
        CollectionOfOne(value).makeIterator()
    }
    public typealias MaskStorage = SIMD1
    public typealias ArrayLiteralElement = ScalarIndex
    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.value < rhs.value
    }
}
