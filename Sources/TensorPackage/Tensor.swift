//
//  Tensor.swift
//  Tensor
//
//  Created by Koen Hendrikx on 15/04/2023.
//

import Foundation
import RealModule
import RegexBuilder
// arange, concat, chunk, dsplit, cstack, dstack,
// masked_select, reshape, tile,

protocol ShapeProtocol {
    associatedtype IndexType: TensorIndex
    associatedtype Dimensions: TypeListProtocol
    init(shape: IndexType, stride: IndexType)
    init(shape: [ScalarIndex], stride: [ScalarIndex])
    var shape: IndexType { get }
    var stride: IndexType { get }
    static var order: Int { get }
    var count: Int { get }
    subscript(index: IndexType) -> ScalarIndex { get }
    subscript(index: ScalarIndex...) -> ScalarIndex { get }
}

public protocol TypeListProtocol {
    static func rcollect(_ a:inout [Any.Type])
    static var types: [Any.Type] { get }
    static var rtypes: [Any.Type] { get }
    static var count: Int { get }
    associatedtype Tail: TypeListProtocol
    associatedtype Head
}
public protocol NonEmptyList {

}
public enum NilTypeList: TypeListProtocol {
    public typealias Tail = Self
    public typealias Head = Never
    public static func rcollect(_ a: inout [Any.Type]) {
    }
    public static var types: [Any.Type] {
        []
    }
    public static var rtypes: [Any.Type] {
        []
    }
    public static var count: Int { 0 }
}

public struct TypeList<Head, Tail: TypeListProtocol>: TypeListProtocol, NonEmptyList {
    public static func rcollect(_ a:inout [Any.Type]) {
//        guard Head.self != Void.self else {
//            return
//        }
        Tail.rcollect(&a)
        a.append(Head.self)
    }
    public static var types: [Any.Type] {
        .init(rtypes.reversed())
    }
    public static var rtypes: [Any.Type] {
        var r: [Any.Type] = []
        rcollect(&r)
        return r
    }
    public static var count: Int {
//        guard Head.self != Void.self else {
//            return 0
//        }
        return 1+Tail.count
    }

}

func broadcastImpl<T: FixedWidthInteger>(shape1:inout [T], stride1:inout [T], shape2: [T], stride2: [T], t1: [Any.Type], t2: [Any.Type]) -> Bool {
    assert(shape2.count>=shape1.count)
    if t1.isEmpty {
        shape1 = shape2
        stride1 = .init(repeating: .zero, count: shape2.count)
        return true
    }
    typealias DIM = (T, T, Any.Type)
    let c = shape1.count
    let s1: [DIM] = .init(Zip3Sequence(a: shape1, b: stride1, c: t1))
    let s2: [DIM] = .init(Zip3Sequence(a: shape2, b: stride2, c: t2))
    let n = c+s2.count+2
    let source=n-2
    let sink=n-1
    var caps: [[UInt8]] = .init(repeating: .init(repeating: 0, count: n), count: n)
    for i in s1.indices {
        caps[source][i] = 1 // start -> A
        for j in s2.indices {
            if s1[i].2 == s2[j].2 && (s1[i].0 == s2[j].0 || s1[i].0 == 1 || s2[j].0 == 1) {
                caps[i][j+c] = 1 // A -> B
            }
        }
    }
    for i in s2.indices {
        caps[i+c][sink] = 1 // B -> end
    }
    let orig = caps
    _ = maxFlow(capacities: &caps, source: source, sink: sink)
    shape1 = []
    stride1 = []
    var matched = 0
    for j in s2.indices {
        let t = s2[j]
        if let i = s1.indices.first(where: {caps[$0][j+c] < orig[$0][j+c]}) {
            let s = s1[i]
            shape1.append(t.0)
            matched += 1
            stride1.append(s.0 == t.0 ? s.1 : 0)
        } else {
            shape1.append(t.0)
            stride1.append(0)
        }
    }

    return matched == c
}

public struct Shape<Dimensions: TypeListProtocol, IndexType: TensorIndex>: ShapeProtocol {
    let shape: IndexType
    let stride: IndexType
    init(shape: ScalarIndex...) {
        self.init(shape: [ScalarIndex](shape))
    }
    init() where Dimensions == NilTypeList, IndexType: FixedWidthInteger {
        self.init(shape: .init(IndexType()))
    }
    static func defaultStride(shape: [ScalarIndex]) -> [ScalarIndex] {
        var s: [ScalarIndex] = [1]
        for i in Swift.stride(from: 1, to: shape.count, by: 1) {
            s.append(s[i-1]*shape[i-1])
        }
        return s
    }
    static func isDefaultStride(shape: IndexType, stride: IndexType) -> Bool {
        if stride[0] != 1 {
            return false
        }
        for i in Swift.stride(from: 1, to: Dimensions.count, by: 1) {
            if stride[i] != stride[i-1]*shape[i-1] {
                return false
            }
        }
        return true
    }

    static func defaultStride(shape: IndexType) -> IndexType {
        var s: IndexType = .one
        for i in Swift.stride(from: 1, to: IndexType.scalarCount, by: 1) {
            s[i] = s[i-1]*shape[i-1]
        }
        return s
    }
    init(shape: IndexType) {
        self.init(shape: shape, stride: Self.defaultStride(shape: shape))
    }
    init(shape: ScalarIndex..., stride: ScalarIndex...) {
        self.init(shape: .init(shape), stride: .init(stride))
    }
    init(shape: [ScalarIndex], stride: [ScalarIndex]) {
        self.init(shape: IndexType(shape), stride: IndexType(stride))
    }
    init(shape: [ScalarIndex]) {
        if shape.count < IndexType.scalarCount {
            var shape = shape
            shape.append(contentsOf: [ScalarIndex](repeating: 1, count: IndexType.scalarCount-shape.count))
            self.init(shape: IndexType(shape))
        } else {
            self.init(shape: IndexType(shape))
        }
    }
    init(shape: IndexType, stride: IndexType) {
        self.shape = shape
        self.stride = stride
    }
    var count: Int {
        Int(shape.product)
    }
    static var types: [Any.Type] {
        Dimensions.types
    }
    static var order: Int {
        Dimensions.count
    }
    subscript(index: ScalarIndex...) -> ScalarIndex {
        self[.init(index)]
    }
    subscript(index: IndexType) -> ScalarIndex {
        index.inner(stride)
    }

    /// cast the shape, ignoring the types, optionally adding size=1 dimensions at the front, dropping size=1 dimensions from the back
    /// - Returns: S, throws a fatalError if not enough size=1 dimensions can be dropped
    func cast<S: ShapeProtocol>() -> S {
        if Self.self == S.self {
            return self as! S
        }
        var s: [ScalarIndex] = .init(shape)
        var t: [ScalarIndex] = .init(stride)
        if s.count != S.order {
            while s.count > S.order {
                if let ix = s.lastIndex(of: 1) {
                    s.remove(at: ix)
                    t.remove(at: ix)
                } else {
                    fatalError("cannot cast shape \(Self.self) to \(S.self)")
                }
            }
            while s.count < S.order {
                s.insert(1, at: 0)
                t.insert(0, at: 0)
            }
        }
        return .init(shape: .init(s), stride: .init(t))
    }
    /// Perform a broadcasting conversion, matching types, and optionally transposing dimensions
    /// - Parameter other: the shape the result should conform to
    /// - Returns: nil if the shape cannot be comverted according to broadcasting rules, otherwise a shape matching the type and dimensions of other
    func broadcast<S: ShapeProtocol>(like other: S) -> S? {
        if S.self == Self.self, let s = self as? S, s.shape == other.shape, s.stride == other.stride {
            return s
        }
        var s1 = shape.map {ScalarIndex($0)}
        var t1 = stride.map {ScalarIndex($0)}
        let s2: [ScalarIndex] = .init(other.shape)
        let t2: [ScalarIndex] = .init(other.stride)
        if broadcastImpl(shape1: &s1, stride1: &t1, shape2: s2, stride2: t2, t1: Dimensions.rtypes, t2: S.Dimensions.rtypes) {
            return S(shape: .init(s1), stride: .init(t1))
        } else {
            return nil
        }
    }
    /// transpose dimensions to match the expected result type
    /// - Returns: A shape of type S with sizes and strides of self transposed to match the types of S, throws a fatalError if the dimensions don't match
    func transpose<S: ShapeProtocol>() -> S? where S.IndexType == IndexType {
        guard Dimensions.count == S.Dimensions.count else { fatalError() }
        var t1 = Dimensions.rtypes
        let t2 = S.Dimensions.rtypes
        var ix: IndexType = shape
        var sx: IndexType = stride
        if Self.self == S.self {
            for i in t1.indices {
                for j in Swift.stride(from: i-1, through: 0, by: -1) {
                    if t1[i] == t1[j] {
                        t1.swapAt(i, j)
                        ix.swapAt(i, j)
                        sx.swapAt(i, j)
                        return .init(shape: ix, stride: sx)
                    }
                }
            }
            return self as? S
        }
    outer:
        for i in t1.indices {
            if t1[i] != t2[i] {
                for j in Swift.stride(from: i+1, to: t1.count, by: 1) {
                    if t1[j] == t2[i] {
                        t1.swapAt(i, j)
                        ix.swapAt(i, j)
                        sx.swapAt(i, j)
                        continue outer
                    }
                }
                return nil
            }
        }
        return .init(shape: ix, stride: sx)
    }
    @available(macOS 13.0, *)
    func flatten<S: ShapeProtocol>(size:inout IndexType, stride:inout IndexType) -> S? {
        var t1: [String] = Dimensions.rtypes.map {"\($0)"}
        let regex = Regex {"("
            Capture { OneOrMore { CharacterClass(.anyOf("()")).inverted } }
            ")" }
        let t2: [[String]] = S.Dimensions.rtypes.map { t in
            if let match = "\(t)".firstMatch(of: regex) {
                var r: [String] = match.output.1.split(separator: ", ").map {String($0)}
                r.reverse()
                return r
            }
            return ["\(t)"]
        }
        let t2f: [String] = t2.flatMap({$0})
        if t2f.count == t1.count {
            var newShape: [ScalarIndex] = []
            var j = 0
            for tt in t2 {
                var s: ScalarIndex = 1
                var d: ScalarIndex = stride[j]
                for t in tt {
                    if t != t1[j], let x = Swift.stride(from: j+1, to: t1.count, by: 1).first(where: {t1[$0]==t}) {
                        size.swapAt(j, x)
                        stride.swapAt(j, x)
                        t1.swapAt(j, x)
                    }
                    d = Swift.min(d, stride[j])
                    s *= size[j]
                    j += 1
                }
                newShape.append(s)
            }
            return S(shape: newShape, stride: Shape.defaultStride(shape: newShape))
        } else {
            return nil
        }
    }
}

extension Shape: Hashable {
    public static func == (lhs: Shape<Dimensions, IndexType>, rhs: Shape<Dimensions, IndexType>) -> Bool {
        lhs.shape == rhs.shape
    }
    public func hash(into hasher: inout Hasher) {
        hasher.combine(shape)
    }
}
extension Shape: CustomStringConvertible {
    public var description: String {
        "<\(Self.types.map {"\($0)".split(separator: ".").last!}.joined(separator: ","))>(\(shape.reversed().map {$0.description}.joined(separator: ",")))"
    }
}
extension Shape: CustomDebugStringConvertible {
    public var debugDescription: String {
        return "<\(Self.types.map {"\($0)".split(separator: ".").last!}.joined(separator: ","))>(\(shape.indices.reversed().map {"\(shape[$0]):\(stride[$0])"}.joined(separator: ",")))"
    }
}

public protocol TensorProtocol {
    associatedtype T
    associatedtype Types: TypeListProtocol
    associatedtype IndexType: TensorIndex
    associatedtype Tail: TensorProtocol where Tail.IndexType == IndexType, Tail.T == T
    init(buffer: Buffer<T>, shape: Shape<Types, IndexType>)
    var buffer: Buffer<T> { get set }
    var shape: Shape<Types, IndexType> { get }
    var isScalar: Bool { get }
    static var order: Int { get }
    var elements: TensorElementCollection<Types, IndexType, T> { get set }
    var elementCount: Int { get }
    mutating func apply(_ op:(inout T) -> Void)
    func mapTensor<R: TensorProtocol>(_ unaryOperator: (T)->R.T) -> R where R.IndexType == IndexType, R.Types == Types
    func multiply<S: TensorProtocol, R: TensorProtocol>(rhs: S, zero: T, multiplyOperator: (T, T) -> T, sumOperator: (T, T) -> T) -> R? where S.T == T, R.T == T
    func combine<S: TensorProtocol>(_ rhs: S, _ op: (T, S.T) -> T) -> Self?
    subscript(index: IndexType) -> T { get set }
    func cast<R: TensorProtocol>() -> R where R.T == T
    func applyUnary<R>(_ op: (T) -> R) -> Tensor<Types, IndexType, R>
    func applyBinaryWithBroadcast(rhs: Self, operator op: (T, T) -> T) -> Self
    func transpose<R: TensorProtocol>() -> R where R.IndexType == IndexType, R.T == T
    mutating func assignCombine<S: TensorProtocol>(_ rhs: S, _ op:(inout T, S.T) -> Void) -> Bool
    func reduceTensor<R: TensorProtocol>(zero: R.T, op: (R.T, T)->R.T) -> R?
    func compareAll<S: TensorProtocol>(_ rhs: S, _ op: (T, S.T) -> Bool) -> Bool?
}
protocol DefaultValue {
    static var defaultValue: Self? { get }
}
extension DefaultValue {
    public static var defaultValue: Self? { nil }
}
extension Double: DefaultValue {
    public static var defaultValue: Self? { .zero }
}
extension Int: DefaultValue {
    public static var defaultValue: Self? { .zero }
}
extension UInt: DefaultValue {
    public static var defaultValue: Self? { .zero }
}
extension String: DefaultValue {
    public static var defaultValue: Self? { "" }
}
extension Array: DefaultValue {
    public static var defaultValue: Self? { [] }
}
public struct Tensor<Types: TypeListProtocol, IndexType: TensorIndex, T>: TensorProtocol {
    public typealias ShapeType = Shape<Types, IndexType>
    public typealias Tail = Tensor<Types.Tail, IndexType, T>

    public var buffer: Buffer<T>
    public let shape: Shape<Types, IndexType>
    public init(buffer: Buffer<T>, shape: Shape<Types, IndexType>) {
//        assert(Types.self == NilTypeList.self || Types.Head.self != Void.self)
        self.buffer = buffer
        self.shape = shape
    }
    public init(shape: [ScalarIndex], function: (IndexType) -> T) {
        var shape = shape
        shape.reverse()
        while shape.count < Self.order {
            shape.append(1)
        }
        let size: Shape<Types, IndexType> = .init(shape: shape)
        self.init(buffer: .init(capacity: size.count), shape: size)
        buffer.apply { ptr in
            var index: IndexType = .zero
            for i in 0..<size.count {
                ptr.advanced(by: i).initialize(to: function(index))
                _ = index.increment(size: size.shape)
            }
        }
    }
    public init() {
        self.init(buffer: .init(capacity: 0), shape: .init(shape: .zero))
    }

    public init(value: T) where Types == NilTypeList, T: Numeric {
        self.init(buffer: .init(capacity: 1, initialValue: value), shape: .init(shape: 1))
    }
    public init<S: TensorProtocol>(_ values: S...) where S.T == T, S.Types == Types.Tail {
        self.init(elements: values)
    }
    public init<S: TensorProtocol>(elements values: [S]) where S.T == T, S.Types == Types.Tail {
        var size: IndexType = .one
        size[Self.order-1] = ScalarIndex(values.count)
        for value in values {
            for j in 0..<S.order {
                size[j] = Swift.max(size[j], value.shape.shape[j])
            }
        }
        let shape: Shape<Types, IndexType> = .init(shape: size)
        var buffer: Buffer<T> = .init(capacity: shape.count)
        buffer.apply { optr in
            for value in values.enumerated() {
                value.element.buffer.apply { iptr in
                    var it = value.element.shape.makeIterator()
                    var it1: IndexType = .zero
                    while let i = it.next() {
                        for j in 0..<S.order {
                            it1[j] = it.index[j]
                        }
                        it1[Self.order-1] = ScalarIndex(value.offset)
                        optr.advanced(by: Int(it1.inner(shape.stride))).initialize(to: iptr[Int(i)])
                    }
                }
            }
        }
        self.init(buffer: buffer, shape: shape)
    }
    public init(concat values: Self...) {
        var size: IndexType = .zero
        for value in values {
            for j in 0..<Self.order-1 {
                size[j] = Swift.max(size[j], value.shape.shape[j])
            }
            size[Self.order-1] += value.shape.shape[Self.order-1]
        }
        let shape: Shape<Types, IndexType> = .init(shape: size)
        var buffer: Buffer<T> = .init(capacity: shape.count)
        var row: ScalarIndex = 0
        buffer.apply { optr in
            for value in values {
                value.buffer.apply { iptr in
                    var it = value.shape.makeIterator()
                    var it1 = it.index
                    it1[Self.order-1] = row
                    while let i = it.next() {
                        optr.advanced(by: Int(it1.inner(shape.stride))).initialize(to: iptr[Int(i)])
                        it1 = it.index
                        it1[Self.order-1] = row + it.index[Self.order-1]
                    }
                }
                row += value.shape.shape[Self.order-1]
            }
        }
        self.init(buffer: buffer, shape: shape)
    }
    public init(shape: [ScalarIndex], initialValue: T) {
        assert(shape.count == Types.count, "expecting \(Types.count) sizes")
        var s: IndexType = .one
        for i in shape.indices {
            s[shape.count-i-1] = shape[i]
        }
        self.init(buffer: .init(capacity: Int(s.product), initialValue: initialValue), shape: .init(shape: s))
    }
    public init(shape: ShapeType, initialValue: T) {
        self.init(buffer: .init(capacity: Int(shape.count), initialValue: initialValue), shape: shape)
    }
    public init(shape: ScalarIndex..., initialValue: T) {
        self.init(shape: shape, initialValue: initialValue)
    }
    public init(shape: ScalarIndex...) where T: Numeric {
        self.init(shape: shape, initialValue: .zero)
    }
    public var size: [Int] {
        Swift.stride(from: Self.order-1, through: 0, by: -1).map {Int(shape.shape[$0])}
    }
    public static var order: Int {
        Types.count
    }
    public var elementCount: Int {
        Int(shape.shape.product)
    }
    public var count: Int {
        if Self.order == 0 {
            return 1
        }
        return Int(shape.shape[Self.order-1])
    }
    public var isScalar: Bool {
        Types.self == NilTypeList.self
    }
    public func broadcast<TP: TypeListProtocol, I: TensorIndex>(to: Shape<TP, I>) -> Tensor<TP, I, T>? {
        if let newShape = shape.broadcast(like: to) {
            return .init(buffer: buffer, shape: newShape)
        } else {
            return nil
        }
    }
    public func transpose<R: TensorProtocol>() -> R where R.IndexType == IndexType, R.T == T {
        if let newShape: Shape<R.Types, IndexType> = shape.transpose() {
            return .init(buffer: buffer, shape: newShape)
        } else {
            fatalError("incompatible types")
        }
    }
    public func cast<R: TensorProtocol>() -> R where R.T == T {
        let newShape: Shape<R.Types, R.IndexType> = shape.cast()
        return .init(buffer: buffer, shape: newShape)
    }
    public func mapTensor<R: TensorProtocol>(_ unaryOperator: (T)->R.T) -> R where R.IndexType == IndexType, R.Types == Types {
        let newShape: Shape<R.Types, IndexType> = .init(shape: shape.shape)
        var result: R = .init(buffer: .init(capacity: newShape.count), shape: newShape)
        var it1 = shape.makeIterator()
        var it2 = newShape.makeIterator()
        result.buffer.apply { optr in
            buffer.apply { iptr in
                while let i = it1.next(), let j = it2.next() {
                    optr.advanced(by: Int(j)).initialize(to: unaryOperator(iptr[Int(i)]))
                }
            }
        }
        return result
    }
    public func reduceTensor<R: TensorProtocol>(zero: R.T, op: (R.T, T)->R.T) -> R? {
        let t1 = Types.rtypes
        let t2 = R.Types.rtypes
        let d = Self.order - R.order
        guard d >= 0 else { return nil }
        var indices:[(source: Int, target: Int)] = []
        var resultsize: [ScalarIndex] = []
        var resultstride: [ScalarIndex] = []
        var rcount = 1
        for target in t2.indices {
            if let source = t1.indices.first(where: {s in t2[target] == t1[s] && !indices.contains(where: {$0.source==s})}) {
                let s = shape.shape[source]
                resultsize.append(s)
                resultstride.append(shape.stride[source])
                rcount *= Int(s)
                indices.append((source:source, target:target))
            }
        }
        var sourcesize: [ScalarIndex] = t1.indices.filter {source in !indices.contains(where: {$0.source==source})}.map {shape.shape[$0]}
        let sumcount = sourcesize.reduce(1, *)
        sourcesize.append(contentsOf: resultsize)
        var sourcestride: [ScalarIndex] = t1.indices.filter {source in !indices.contains(where: {$0.source==source})}.map {shape.stride[$0]}
        sourcestride.append(contentsOf: resultstride)
        guard indices.count == t2.count else { return nil }
        let resultshape: Shape<R.Types, R.IndexType> = .init(shape: resultsize)
        var rbuffer: Buffer<R.T> = .init(capacity: rcount)
        rbuffer.apply { optr in
            buffer.apply { iptr in
                var innerit = SIMDIterator<IndexType>(size: sourcesize, stride: sourcestride)
                if sumcount > 1024 && R.T.self == T.self {
                    let bits = (ScalarIndex.bitWidth - sumcount.leadingZeroBitCount)/2
                    for j in 0..<rcount {
                        var sums: [R.T] = .init(repeating: zero, count: 1+Int(sumcount)>>bits)
                        var k = 0
                        while k < sumcount {
                            let i = innerit.next()!
                            sums[k>>bits] = op(sums[k>>bits], iptr[Int(i)])
                            k += 1
                        }
                        var step = 1
                        while step < sums.count {
                            for i in stride(from: 0, to: sums.count-step, by: step) {
                                sums[i] = op(sums[i], sums[i+step] as! T)
                            }
                            step <<= 1
                        }
                        optr.advanced(by: Int(j)).initialize(to: sums[0])
                    }
                } else {
                    for j in 0..<rcount {
                        var sum = zero
                        for _ in 0..<sumcount {
                            let i = innerit.next()!
                            sum = op(sum, iptr[Int(i)])
                        }
                        optr.advanced(by: Int(j)).initialize(to: sum)
                    }
                }
            }
        }
        return .init(buffer: rbuffer, shape: resultshape)
    }
    public func outer<S: TensorProtocol, R: TensorProtocol>(rhs: S, _ op: (T, T) -> T) -> R where S.T == T, R.T == T {
        let t1 = Types.types
        let t2 = S.Types.types
        let t3 = R.Types.types
        guard zip((t1+t2), t3).allSatisfy({$0.0 == $0.1}) else { fatalError("incompatible types") }
        var rbuffer: Buffer<T> = .init(capacity: Int(shape.count*rhs.shape.count))
        let shape: Shape<R.Types, R.IndexType> = .init(shape: (rhs.shape.shape.map {ScalarIndex($0)} + self.shape.shape.map {ScalarIndex($0)}))
        var c = 0
        var itl = self.shape.makeIterator()
        rbuffer.apply { optr in
            buffer.apply { lptr in
                rhs.buffer.apply { rptr in
                    while let i = itl.next() {
                        let v = lptr[Int(i)]
                        var it2 = rhs.shape.makeIterator()
                        while let j = it2.next() {
                            optr.advanced(by: c).initialize(to: op(v, rptr[Int(j)]))
                            c += 1
                        }
                    }
                }
            }
        }
        return .init(buffer: rbuffer, shape: shape)
    }
    public func multiply<S: TensorProtocol, R: TensorProtocol>(rhs: S, zero: T, multiplyOperator: (T, T) -> T, sumOperator: (T, T) -> T) -> R? where S.T == T, R.T == T {
        let dx = Self.order+S.order-R.order
        let d = dx/2
        guard dx >= 0 && dx&1 == 0 else {
            if R.order == 0 {
                if Self.order == 0 {
                    let c = buffer.readable.pointee
                    let r = rhs.elements.reduce(zero) {sumOperator($0, multiplyOperator($1, c))}
                    return .init(buffer: .init(capacity: 1, initialValue: r), shape: .init(shape: []))
                } else if S.order == 0 {
                    let c = rhs.buffer.readable.pointee
                    let r = elements.reduce(zero) {sumOperator($0, multiplyOperator($1, c))}
                    return .init(buffer: .init(capacity: 1, initialValue: r), shape: .init(shape: []))
                }
            }
            print(rhs)
            print(self)
            print(Self.order)
            print(S.order)
            print(R.order)
            return nil
        }
        typealias DIM = (ScalarIndex, ScalarIndex, Any.Type)
        let s1: [DIM] = .init(Zip3Sequence(a: shape.shape, b: shape.stride, c: Types.types.reversed()))
        let s2: [DIM] = .init(Zip3Sequence(a: rhs.shape.shape, b: rhs.shape.stride, c: S.Types.types.reversed()))
        let s3: [Any.Type] = .init(R.Types.types.reversed())
        func match(abmap:[(key: Int, value: Int)]) -> (acmap:[(key: Int, value: Int)], bcmap:[(key: Int, value: Int)])? {
            var acmap: [(key: Int, value: Int)] = []
            var bcmap: [(key: Int, value: Int)] = []
            for i in s1.indices {
                if !abmap.contains(where: {$0.key==i}) {
                    for j in s3.indices {
                        if !acmap.contains(where: {$0.value==j}) {
                            if acmap.count < s1.count-d && s1[i].2 == s3[j] {
                                acmap.append((i, j))
                                break
                            }
                        }
                    }
                }
            }
            for i in s2.indices.reversed() {
                if !abmap.contains(where: {$0.value==i}) {
                    for j in s3.indices.reversed() {
                        if (!bcmap.contains(where: {$0.key==j})) {
                            if bcmap.count < s3.count-s1.count+d && s2[i].2 == s3[j] {
                                bcmap.append((i, j))
                                break
                            }
                        }
                    }
                }
            }
            return acmap.count == s1.count-d && bcmap.count == s3.count-s1.count+d ? (acmap:acmap, bcmap:bcmap) : nil
        }
        let abpairs:[(key: Int, value: Int)] = s1.indices.compactMap {i in
            s2.indices.reversed().first(where: {j in s1[i].2 == s2[j].2 && (s1[i].0 == s2[j].0 || s1[i].0 == 1 || s2[j].0 == 1)}).map({j in (key:i, value:j)})
        }
        func search(abmap:inout [(key: Int, value: Int)], index: Int) -> (acmap:[(key: Int, value: Int)], bcmap:[(key: Int, value: Int)])? {
            if abmap.count == d {
                return match(abmap: abmap)
            }
            if index >= abpairs.count {
                return nil
            }
            for i in index..<abpairs.count {
                abmap.append((abpairs[i]))
                if let r = search(abmap: &abmap, index: i+1) {
                    return r
                }
                abmap.removeLast()
            }
            return nil
        }
        var abmap: [(key: Int, value: Int)] = []
        if let r = search(abmap: &abmap, index: 0) {
            let acmap:[(key: Int, value: Int)] = r.acmap
            let bcmap:[(key: Int, value: Int)] = r.bcmap
            var r_size: [ScalarIndex] = .init(repeating: 0, count: R.order)
            var l_stride: [ScalarIndex] = .init(repeating: 0, count: Self.order+S.order-d)
            var r_stride: [ScalarIndex] = .init(repeating: 0, count: Self.order+S.order-d)
            var i_size: [ScalarIndex] = .init(repeating: 0, count: Self.order+S.order-d)
            bcmap.forEach {
                r_size[$0.value] = s2[$0.key].0
                r_stride[$0.value+d] = s2[$0.key].1
                i_size[$0.value+d] = s2[$0.key].0
            }
            acmap.forEach { ac in
                assert(!abmap.contains(where: {ab in ab.key==ac.key}))
                r_size[ac.value] = s1[ac.key].0
                l_stride[ac.value+d] = s1[ac.key].1
                i_size[ac.value+d] = s1[ac.key].0
            }
            abmap.enumerated().forEach {
                i_size[$0.offset] = Swift.max(s1[$0.element.key].0, s2[$0.element.value].0)
                if true || R.order == 0 {
                    l_stride[$0.offset] = s1[$0.element.key].1
                    r_stride[$0.offset] = s2[$0.element.value].1
                }
            }
            let sumcount = Int(abmap.reduce(1, {$0 * Swift.max(s1[$1.key].0, s2[$1.value].0)}))
            let resultshape: Shape<R.Types, R.IndexType> = .init(shape: r_size)
            switch S.order+Self.order - d {
            case 1...2:
                return multiplyImpl(indexType: SIMD2<ScalarIndex>.self, rhs: rhs, innershape: i_size, linnerstride: l_stride, rinnerstride: r_stride, resultshape: resultshape, zero: zero, sumcount: sumcount, multiplyOperator: multiplyOperator, sumOperator: sumOperator)
            case 3:
                return multiplyImpl(indexType: SIMD3<ScalarIndex>.self, rhs: rhs, innershape: i_size, linnerstride: l_stride, rinnerstride: r_stride, resultshape: resultshape, zero: zero, sumcount: sumcount, multiplyOperator: multiplyOperator, sumOperator: sumOperator)
            case 4:
                return multiplyImpl(indexType: SIMD4<ScalarIndex>.self, rhs: rhs, innershape: i_size, linnerstride: l_stride, rinnerstride: r_stride, resultshape: resultshape, zero: zero, sumcount: sumcount, multiplyOperator: multiplyOperator, sumOperator: sumOperator)
            case 5...8:
                return multiplyImpl(indexType: SIMD8<ScalarIndex>.self, rhs: rhs, innershape: i_size, linnerstride: l_stride, rinnerstride: r_stride, resultshape: resultshape, zero: zero, sumcount: sumcount, multiplyOperator: multiplyOperator, sumOperator: sumOperator)
            default:
                fatalError()
            }
        } else {
            fatalError()
        }
    }
    func multiplyImpl<R: TensorProtocol, S: TensorProtocol, IteratorType: TensorIndex>(indexType: IteratorType.Type, rhs: S, innershape: [ScalarIndex], linnerstride: [ScalarIndex], rinnerstride: [ScalarIndex], resultshape: Shape<R.Types, R.IndexType>, zero: T, sumcount: Int, multiplyOperator: (T, T) -> T, sumOperator: (T, T) -> T) -> R where S.T == T, R.T == T {
        var it1 = SIMDIterator<IteratorType>(size: innershape, stride: linnerstride)
        var it2 = SIMDIterator<IteratorType>(size: innershape, stride: rinnerstride)
        var rbuffer: Buffer<T> = .init(capacity: Int(resultshape.count))
        rbuffer.apply { optr in
            buffer.apply { lptr in
                rhs.buffer.apply { rptr in
                    for i in 0..<resultshape.count {
                        var sum = zero
                        for _ in 0..<sumcount {
                            let l: Int = Int(it1.next()!)
                            let r: Int = Int(it2.next()!)
//                            print("\(l) \(r) +\(lptr[l])*\(rptr[r])")
                            sum = sumOperator(sum, multiplyOperator(lptr[l], rptr[r]))
                        }
//                        print("r[\(i)]=\(sum)")
                        optr.advanced(by: i).initialize(to: sum)
                    }
                }
            }
        }
        return .init(buffer: rbuffer, shape: resultshape)
    }
    public func applyUnary<R>(_ op: (T) -> R) -> Tensor<Types, IndexType, R> {
        var it = shape.makeIterator()
        var c = 0
        var rbuffer: Buffer<R> = .init(capacity: shape.count)
        rbuffer.apply { optr in
            buffer.apply { lptr in
                while let i = it.next() {
                    optr.advanced(by: c).initialize(to: op(lptr[Int(i)]))
                    c += 1
                }
            }
        }
        return .init(buffer: rbuffer, shape: .init(shape: shape.shape))
    }
    public mutating func apply(_ op:(inout T) -> Void) {
        var it = shape.makeIterator()
        buffer.apply { optr in
            while let i = it.next() {
                op(&optr[Int(i)])
            }
        }
    }
    public mutating func assignCombine<S: TensorProtocol>(_ rhs: S, _ op:(inout T, S.T) -> Void) -> Bool {
        var it1 = shape.makeIterator()
        if let s2 = rhs.shape.broadcast(like: shape) {
            buffer.apply { lptr in
                rhs.buffer.apply { rptr in
                    var it2 = s2.makeIterator()
                    while let i = it1.next(), let j = it2.next() {
                        op(&lptr[Int(i)], rptr[Int(j)])
                    }
                }
            }
            return true
        }
        return false
    }
    public func combine<S: TensorProtocol>(_ rhs: S, _ op: (T, S.T) -> T) -> Self? {
        var it1 = shape.makeIterator()
        if let s2 = rhs.shape.broadcast(like: shape) {
            var rbuffer: Buffer<T> = .init(capacity: shape.count)
            var c = 0
            rbuffer.apply { optr in
                buffer.apply { lptr in
                    rhs.buffer.apply { rptr in
                        var it2 = s2.makeIterator()
                        while let i = it1.next(), let j = it2.next() {
                            let v = op(lptr[Int(i)], rptr[Int(j)])
                            optr.advanced(by: c).initialize(to: v)
                            c += 1
                        }
                    }
                }
            }
            return .init(buffer: rbuffer, shape: shape)
        }
        return nil
    }

    func combine2<S: TensorProtocol, R: TensorProtocol>(_ rhs: S, _ op: (T, S.T) -> R.T) -> R? where IndexType == S.IndexType, IndexType == R.IndexType {
        var it1 = shape.makeIterator()
        if let s2 = rhs.shape.broadcast(like: shape) {
            var rbuffer: Buffer<R.T> = .init(capacity: shape.count)
            var c = 0
            rbuffer.apply { optr in
                buffer.apply { lptr in
                    rhs.buffer.apply { rptr in
                        var it2 = s2.makeIterator()
                        while let i = it1.next(), let j = it2.next() {
                            let v = op(lptr[Int(i)], rptr[Int(j)])
                            optr.advanced(by: c).initialize(to: v)
                            c += 1
                        }
                    }
                }
            }
            return .init(buffer: rbuffer, shape: shape.cast())
        }
        return nil
    }

    public func compareAll<S: TensorProtocol>(_ rhs: S, _ op: (T, S.T) -> Bool) -> Bool? {
        var it1 = shape.makeIterator()
        if let s2 = rhs.shape.broadcast(like: shape) {
            return buffer.apply { lptr in
                rhs.buffer.apply { rptr in
                    var it2 = s2.makeIterator()
                    while let i = it1.next(), let j = it2.next() {
                        let v = op(lptr[Int(i)], rptr[Int(j)])
                        if !v {
                            return false
                        }
                    }
                    return true
                }
            }
        }
        return nil
    }
    public func applyBinary(rhs: Self, _ op: (T, T) -> T) -> Self {
        var it = shape.pairIterator(rhs: rhs.shape)
        var c = 0
        var rbuffer: Buffer<T> = .init(capacity: shape.count)
        rbuffer.apply { optr in
            buffer.apply { lptr in
                rhs.buffer.apply { rptr in
                    while let i = it.next() {
                        optr.advanced(by: c).initialize(to: op(lptr[Int(i.0)], rptr[Int(i.1)]))
                        c += 1
                    }
                }
            }
        }
        return .init(buffer: rbuffer, shape: .init(shape: shape.shape))
    }
    public mutating func applyBinary(rhs: Self, _ op: (inout T, T) -> Void) {
        var it = shape.pairIterator(rhs: rhs.shape)
        var c = 0
        buffer.apply { lptr in
            rhs.buffer.apply { rptr in
                while let i = it.next() {
                    op(&lptr[Int(i.0)], rptr[Int(i.1)])
                    c += 1
                }
            }
        }
    }
    public func applyBinaryWithBroadcast(rhs: Self, operator op: (T, T) -> T) -> Self {
        if count < rhs.count {
            if let s = broadcast(to: rhs.shape) {
                return s.applyBinary(rhs: rhs, op)
            } else {
                fatalError()
            }
        }
        if let s = rhs.broadcast(to: shape) {
            return applyBinary(rhs: s, op)
        } else {
            fatalError()
        }
    }

    public var elements: TensorElementCollection<Types, IndexType, T> {
        get {
            .init(shape: shape, buffer: buffer)
        }
        set {
            if newValue.shape.shape == shape.shape {
                if newValue.shape.stride == shape.stride {
                    buffer.transfer(range: 0..<shape.count, from: newValue.buffer)
                } else {
                    var it = newValue.shape.makeIterator()
                    var it1 = shape.makeIterator()
                    buffer.apply { optr in
                        newValue.buffer.apply { iptr in
                            if let i=it.next(), let j=it1.next() {
                                optr[Int(j)] = iptr[Int(i)]
                            }
                        }
                    }
                }
            } else {
                fatalError()
            }
        }
        _modify {
            var r = TensorElementCollection<Types, IndexType, T>(shape: shape, buffer: buffer)
            yield &r
            buffer.transfer(range: 0..<shape.count, from: r.buffer)
        }
    }
    public subscript(position: IndexType) -> T {
        get {
            let index = position.inner(shape.stride)
            //            print("getting \(index) -> \(buffer.readable[Int(index)])")
            return buffer.readable[Int(index)]
        }
        _modify {
            let index = position.inner(shape.stride)
            //            print("set \(index) -> \(newValue)")
            yield &buffer.writable[Int(index)]
        }
        set {
            let index = position.inner(shape.stride)
            buffer.writable.advanced(by: Int(index)).pointee = newValue
        }
    }
}

public struct TensorElementCollection<Types: TypeListProtocol, IndexType: TensorIndex, T>: MutableCollection {
    var shape: Shape<Types, IndexType>
    var buffer: Buffer<T>
    public typealias Element = T
    public typealias Iterator = BufferIndexingIterator<Shape<Types, IndexType>.Iterator, T>

    public func makeIterator() -> BufferIndexingIterator<Shape<Types, IndexType>.Iterator, T> {
        .init(it: shape.makeIterator(), mem: buffer.readable)
    }
    public typealias Index = IndexType

    public var count: Int {
        Int(shape.count)
    }

    public subscript(position: IndexType) -> T {
        get {
            let index = position.inner(shape.stride)
            return buffer.readable[Int(index)]
        }
        _modify {
            let index = position.inner(shape.stride)
            yield &buffer.writable[Int(index)]
        }
        set {
            let index = position.inner(shape.stride)
            buffer.writable.advanced(by: Int(index)).pointee = newValue
        }
    }

    public var startIndex: Index {
        .zero
    }

    public var endIndex: Index {
        var s: Index = .zero
        s[Types.count-1] = shape.shape[Types.count-1]
        return s
    }

    public func index(after i: Index) -> Index {
        var r = i
        formIndex(after: &r)
        return r
    }
    public func formIndex(after r: inout Index) {
        for i in 0..<Types.count {
            r[i] &+= 1
            if r[i] == shape.shape[i] && i+1 != Types.count {
                r[i] = 0
            } else {
                break
            }
        }
    }
    func mapToString(toString: (T) -> String) -> String {

        if shape.count == 1 {
            return toString(self[self.startIndex])
        }
        var r = ""
        var i = startIndex
        var sep = ""
        var l = 0
        while i < endIndex {
            r += sep
            sep = ", "
            for j in 0..<Types.count {
                if i[j] == .zero {
                    r += "["
                    l += 1
                } else {
                    break
                }
            }
            r += toString(self[i])
            self.formIndex(after: &i)
            for j in 0..<Types.count {
                if i[j] == .zero {
                    r += "]"
                    l -= 1
                } else {
                    break
                }
            }
        }
        for _ in stride(from: 0, to: l, by: 1) {
            r += "]"
        }
        return r
    }
}

extension Tensor: CustomStringConvertible where T: CustomStringConvertible {
    public var description: String {
        elements.mapToString(toString: \.description)
    }
}
extension Tensor: CustomDebugStringConvertible where T: CustomDebugStringConvertible {
    public var debugDescription: String {
        elements.mapToString(toString: \.debugDescription)
    }
}
protocol ArrayValueProtocol {
    var array: [Any] { get }
}
protocol ScalarValueProtocol {
    associatedtype T
    var value: T { get set }
}
extension Tensor where Types == NilTypeList, T: Numeric {
    public init() {
        self.init(buffer: .init(capacity: 1), shape: .init(shape: 1))
    }
}
@available(macOS 13.0, *)
extension Tensor: ArrayValueProtocol where Types: NonEmptyList {
    public var array: [Any] {
        var result: [Any] = []
        for i in 0..<Int(shape.shape[Self.order-1]) {
            let t: Tail = self[i]
            if let v = t as? any ArrayValueProtocol {
                result.append(v.array)
            } else if let v = t as? any ScalarValueProtocol {
                result.append(v.value)
            }
        }
        return result
    }
    public func flatten<S: TensorProtocol>() -> S? where S.T == T {
        var size = shape.shape
        var stride = shape.stride
        if let newShape: Shape<S.Types, S.IndexType> = shape.flatten(size: &size, stride: &stride) {
            if size == shape.shape && stride == shape.stride && Shape<Types, IndexType>.isDefaultStride(shape: size, stride: stride) {
                return .init(buffer: buffer, shape: newShape)
            } else {
                var rbuffer: Buffer<T> = .init(capacity: shape.count)
                rbuffer.apply { optr in
                    buffer.apply { iptr in
                        var it1 = SIMDIterator(size: size, stride: stride)
                        var it2 = newShape.makeIterator()
                        while let i = it1.next(), let j = it2.next() {
                            optr.advanced(by: Int(j)).initialize(to: iptr[Int(i)])
                        }
                    }
                }
                return .init(buffer: rbuffer, shape: newShape)
            }
        } else {
            return nil
        }
    }
}
extension Tensor: ExpressibleByArrayLiteral where Types: NonEmptyList {
    public typealias ArrayLiteralElement = Tail
    fileprivate static func computeDimension<S>(a: S, dim:inout [ScalarIndex], stack:inout [ScalarIndex]) {
        if let _ = a as? T {

        } else if let t = a as? Tail {
            if t.isScalar {
                return
            }
            for i in t.size {
                stack.append(ScalarIndex(i))
            }
            for i in 0..<Swift.min(stack.count, dim.count) {
                dim[i] = Swift.max(dim[i], stack[i])
            }
            if dim.count < stack.count {
                dim.append(contentsOf: stack[dim.count...])
            }
            for _ in t.size {
                stack.removeLast()
            }
        } else
        if let s:(any Sequence) = a as? (any Sequence) {
            var count = 0
            if let c:(any Collection) = s as? (any Collection) {
                count = c.count
            } else {
                for _ in s {
                    count += 1
                }
            }
            stack.append(ScalarIndex(count))
            for i in s {
                computeDimension(a: i, dim: &dim, stack: &stack)
            }
            for i in 0..<Swift.min(stack.count, dim.count) {
                dim[i] = Swift.max(dim[i], stack[i])
            }
            if dim.count < stack.count {
                dim.append(contentsOf: stack[dim.count...])
            }
            stack.removeLast()
        }
    }
    fileprivate static func computeDimension<S>(_ a: S) -> [ScalarIndex] {
        var r: [ScalarIndex] = []
        var stack: [ScalarIndex] = []
        computeDimension(a: a, dim: &r, stack: &stack)
        return r
    }
    fileprivate static func assign<S>(a: S, index:inout IndexType, shape: IndexType, stride: IndexType, level: Int, ptr: UnsafeMutablePointer<T>) {
        if let v: T = a as? T {
            let i = index.inner(stride)
            ptr[Int(i)] = v
        } else if let v = a as? Tail {
            let i = index.inner(stride)
            ptr.advanced(by: Int(i)).update(from: v.buffer.readable, count: v.shape.count)
        } else
        if let s:(any Sequence) = a as? (any Sequence) {
            for i in s {
                assign(a: i, index: &index, shape: shape, stride: stride, level: level-1, ptr: ptr)
            }
        } else {
            fatalError()
        }
        if level < Self.order {
            index[level] += 1
            for i in 0..<level {
                index[i] = 0
            }
        }
    }
    fileprivate static func initialize<S>(a: S, index:inout IndexType, shape: IndexType, stride: IndexType, level: Int, ptr: UnsafeMutablePointer<T>) -> Int {
        var result = 0
        if let v: T = a as? T {
            let i = index.inner(stride)
            ptr.advanced(by: Int(i)).initialize(to: v)
            result += 1
        } else if let v = a as? Tail {
            let i = index.inner(stride)
            ptr.advanced(by: Int(i)).initialize(from: v.buffer.readable, count: v.shape.count)
            result += v.shape.count
        } else
        if let s:(any Sequence) = a as? (any Sequence) {
            for i in s {
                result+=initialize(a: i, index: &index, shape: shape, stride: stride, level: level-1, ptr: ptr)
            }
        } else {
            fatalError()
        }
        if level < Self.order {
            index[level] += 1
            for i in 0..<level {
                index[i] = 0
            }
        }
        return result
    }
    public init(elements: [ArrayLiteralElement]) {
        let dim = Self.computeDimension(elements)
        var index: IndexType = .zero
        var sdim = [ScalarIndex](dim.reversed())
        while sdim.count < Self.order {
            sdim.append(1)
        }
        let s: Shape<Types, IndexType> = .init(shape: sdim)
        var buf: Buffer<T>
        if let d: T = (T.self as? DefaultValue.Type)?.defaultValue as! T? {
            buf = .init(capacity: s.count, initialValue: d)
            buf.apply { (ptr: UnsafeMutablePointer<T>) in
                Self.assign(a: elements, index: &index, shape: s.shape, stride: s.stride, level: dim.count, ptr: ptr)
            }
        } else {
            buf = .init(capacity: s.count)
            if s.count != buf.apply({ (ptr: UnsafeMutablePointer<T>) in
                Self.initialize(a: elements, index: &index, shape: s.shape, stride: s.stride, level: dim.count, ptr: ptr)
            }) {
                fatalError("some elements were not initialized")
            }
        }
        self.init(buffer: buf, shape: s)
    }
    public init<S: Sequence, X>(fromArray array: S) where S.Element == X {
        let dim = Self.computeDimension(array)
        var index: IndexType = .zero
        var sdim = [ScalarIndex](dim.reversed())
        while sdim.count < Self.order {
            sdim.append(1)
        }
        let s: Shape<Types, IndexType> = .init(shape: sdim)
        var buf: Buffer<T>
        if let d: T = (T.self as? DefaultValue.Type)?.defaultValue as! T? {
            buf = .init(capacity: s.count, initialValue: d)
            buf.apply { (ptr: UnsafeMutablePointer<T>) in
                Self.assign(a: array, index: &index, shape: s.shape, stride: s.stride, level: dim.count, ptr: ptr)
            }
        } else {
            buf = .init(capacity: s.count)
            if s.count != buf.apply({ (ptr: UnsafeMutablePointer<T>) in
                Self.initialize(a: array, index: &index, shape: s.shape, stride: s.stride, level: dim.count, ptr: ptr)
            }) {
                fatalError("some elements were not initialized")
            }
        }
        self.init(buffer: buf, shape: s)
    }

    public init(arrayLiteral elements: ArrayLiteralElement...) {
        self.init(elements: elements)
    }
}

extension Tensor: Collection {
    public typealias SubSequence = Self
    public func index(after i: Int) -> Int {
        i+1
    }
    public func index(before i: Int) -> Int {
        i-1
    }
    public func formIndex(before i: inout Int) {
        i -= 1
    }
    public func distance(from start: Int, to end: Int) -> Int {
        end-start
    }
    public func formIndex(after i: inout Int) {
        i += 1
    }
    public func index(_ i: Int, offsetBy distance: Int) -> Int {
        i+distance
    }
    public func index(_ i: Int, offsetBy distance: Int, limitedBy limit: Int) -> Int? {
        let r = i+distance
        if distance >= 0 {
            return r < limit || limit < i ? r : nil
        } else {
            return r > limit || limit > i ? r : nil
        }
    }
    public var startIndex: Int {
        0
    }
    public var endIndex: Int {
        count
    }

    public subscript<R: RangeExpression>(bounds: R) -> Self where R.Bound == Int {
        get {
            return self[bounds.relative(to: 0..<count)]
        }
        set {
            self[bounds.relative(to: 0..<count)]=newValue
        }
        _modify {
            yield &self[bounds.relative(to: 0..<count)]
        }
    }

    public subscript(bounds: Range<Int>) -> Self {
        get {
            var size = shape.shape
            let index = bounds.startIndex * Int(shape.stride[Self.order-1])
            size[Self.order-1] = ScalarIndex(bounds.count)
            return .init(buffer: .init(owner: buffer, offset: index, count: Int(size.product), shared: false), shape: .init(shape: size, stride: shape.stride))
        }
        _modify {
            var size = shape.shape
            let index = bounds.startIndex * Int(shape.stride[Self.order-1])
            size[Self.order-1] = ScalarIndex(bounds.count)
            var result: Self = .init(buffer: .init(owner: buffer, offset: index, count: Int(size.product), shared: true), shape: .init(shape: size, stride: shape.stride))
//            print(result)
            yield &result
//            print(result)
        }
        set {
            var size = shape.shape
            let index = bounds.startIndex * Int(shape.stride[Self.order-1])
            size[Self.order-1] = ScalarIndex(bounds.count)
            let shape = Shape<Types, IndexType>(shape: size, stride: shape.stride)
            if shape != newValue.shape {
                fatalError()
            }
            var it1 = shape.makeIterator()
            var it2 = newValue.shape.makeIterator()
            buffer.apply { optr in
                newValue.buffer.apply { iptr in
                    while let i = it1.next(), let j = it2.next() {
                        optr[Int(i)+index] = iptr[Int(j)]
                    }
                }
            }
        }
    }

    public subscript(index: Int) -> Tail {
        get {
            var s: IndexType = shape.shape
            var t: IndexType = shape.stride
            let ix = Swift.max(0, Types.count-1)
            if Types.count > 0 && index >= s[ix] {
                fatalError()
            }
            s[ix] = 1
            t[ix] = 0
            let size = Int(s.product)
            let start = index*size
            return .init(buffer: buffer[start..<start+size, false], shape: .init(shape: s, stride: t))
        }
        _modify {
            var s: IndexType = shape.shape
            var t: IndexType = shape.stride
            let ix = Swift.max(0, Types.count-1)
            if Types.count > 0 && index >= s[ix] {
                fatalError()
            }
            s[ix] = 1
            t[ix] = 0
            let size = Int(s.product)
            let start = index*size
            var result: Tail = .init(buffer: buffer[start..<start+size, false], shape: .init(shape: s, stride: t))
//            print(result)
            yield &result
            if result.shape.shape != s {
                fatalError("cannot change the size of a subtensor")
            }
            buffer.transfer(range: start..<start+size, from: result.buffer)
//            print(result)
        }
        set {
            var s: IndexType = shape.shape
            let ix = Swift.max(0, Types.count-1)
            if Types.count > 0 && index >= s[ix] {
                fatalError("index \(index) out of bounds (>=\(s[Types.count-1]))")
            }
            s[ix] = 1
            let size = Int(s.product)
            let rsize = Int(newValue.shape.count)
            if size != rsize {
                fatalError("size mismatch when assigning a subtensor (expected \(size) but got \(rsize))")
            }
            let start = index*size
            if newValue.shape.stride == shape.stride && Shape<Types, IndexType>.isDefaultStride(shape: newValue.shape.shape, stride: newValue.shape.stride) {
                buffer.transfer(range: start..<start+size, from: newValue.buffer)
            } else {
                var it = newValue.shape.makeIterator()
                var it2 = SIMDIterator(size: s, stride: shape.stride)
                buffer.apply { optr in
                    newValue.buffer.apply { iptr in
                        while let i = it.next(), let j = it2.next() {
                            optr[start+Int(j)] = iptr[Int(i)]
                        }
                    }
                }
            }
        }
    }
}
extension Tensor: BidirectionalCollection {}

extension Tensor: MutableCollection {}
extension Tensor: RandomAccessCollection {}
extension Tensor: RangeReplaceableCollection {
    public func replaceSubrange<C>(_ subrange: Range<Int>, with newElements: C) where C: Collection, Tensor<Types.Tail, IndexType, T> == C.Element {
        fatalError()
    }
}

extension Tensor where Types == NilTypeList {
    public init(_ value: T) {
        self.init(shape: [], initialValue: value)
    }
}
extension Tensor: ExpressibleByFloatLiteral where T: ExpressibleByFloatLiteral {
    public init(floatLiteral value: T.FloatLiteralType) {
        self.init(buffer: .init(capacity: 1, initialValue: .init(floatLiteral: value)), shape: .init(shape: 1))
    }
}
extension Tensor: ExpressibleByUnicodeScalarLiteral where T: ExpressibleByUnicodeScalarLiteral {
    public init(unicodeScalarLiteral value: T.UnicodeScalarLiteralType) {
        self.init(buffer: .init(capacity: 1, initialValue: .init(unicodeScalarLiteral: value)), shape: .init(shape: 1))
    }
}

extension Tensor: ExpressibleByExtendedGraphemeClusterLiteral where T: ExpressibleByExtendedGraphemeClusterLiteral {
    public init(extendedGraphemeClusterLiteral value: T.ExtendedGraphemeClusterLiteralType) {
        self.init(buffer: .init(capacity: 1, initialValue: .init(extendedGraphemeClusterLiteral: value)), shape: .init(shape: 1))
    }
}

extension Tensor: ExpressibleByStringLiteral where T: ExpressibleByStringLiteral {
    public init(stringLiteral value: T.StringLiteralType) {
        self.init(buffer: .init(capacity: 1, initialValue: .init(stringLiteral: value)), shape: .init(shape: 1))
    }
}
extension Tensor: ExpressibleByBooleanLiteral where T: ExpressibleByBooleanLiteral {
    public init(booleanLiteral value: T.BooleanLiteralType) {
        self.init(buffer: .init(capacity: 1, initialValue: .init(booleanLiteral: value)), shape: .init(shape: 1))
    }
}
extension Tensor: ExpressibleByIntegerLiteral where T: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: T.IntegerLiteralType) {
        self.init(buffer: .init(capacity: 1, initialValue: .init(integerLiteral: value)), shape: .init(shape: 1))
    }
}
extension Tensor: ExpressibleByNilLiteral where T: ExpressibleByNilLiteral {
    public init(nilLiteral: ()) {
        self.init(buffer: .init(capacity: 1, initialValue: .init(nilLiteral: ())), shape: .init(shape: 1))
    }
}

extension Tensor: Equatable where T: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        if lhs.shape != rhs.shape {
            return false
        }
        var it = lhs.shape.pairIterator(rhs: rhs.shape)
        while let i = it.next() {
            if lhs.buffer.readable[Int(i.0)] != rhs.buffer.readable[Int(i.1)] {
                return false
            }
        }
        return true
    }
}

extension Tensor: Hashable where T: Hashable {
    public func hash(into hasher: inout Hasher) {
        for i in elements {
            hasher.combine(i)
        }
    }
}
extension TensorProtocol where T: AdditiveArithmetic {
    public func sum<S: TensorProtocol>() -> S where S.T == T {
        reduceTensor(zero: T.zero, op: +)!
    }
    public var elementSum: T {
        elements.reduce(.zero, +)
    }
}
extension Tensor: AdditiveArithmetic where T: AdditiveArithmetic {
    public static func zeros(like: Self) -> Self {
        .init(buffer: .init(capacity: like.elementCount, initialValue: .zero), shape: like.shape)
    }
    public static func zeros(shape sizes: ScalarIndex...) -> Self {
        let shape: Shape<Types, IndexType> = .init(shape: sizes.reversed())
        return .init(buffer: .init(capacity: shape.count, initialValue: .zero), shape: shape)
    }

    public static func - (lhs: Self, rhs: Self) -> Self {
        if let s = rhs.broadcast(to: lhs.shape) {
            return lhs.applyBinary(rhs: s, -)
        } else {
            fatalError()
        }
    }

    public static func + (lhs: Self, rhs: Self) -> Self {
        if let s = rhs.broadcast(to: lhs.shape) {
            return lhs.applyBinary(rhs: s, +)
        } else {
            fatalError()
        }
    }
    public static func -= (lhs: inout Self, rhs: Self) {
        if let s = rhs.broadcast(to: lhs.shape) {
            lhs.applyBinary(rhs: s, -=)
        } else {
            fatalError()
        }
    }
    public static func += (lhs: inout Self, rhs: Self) {
        if let s = rhs.broadcast(to: lhs.shape) {
            lhs.applyBinary(rhs: s, +=)
        } else {
            fatalError()
        }
    }

    public static func +=<S: TensorProtocol> (lhs: inout Self, rhs: S) where T == S.T {
        if !lhs.assignCombine(rhs, +=) {
            fatalError()
        }
    }
    public static func -=<S: TensorProtocol> (lhs: inout Self, rhs: S) where T == S.T {
        if !lhs.assignCombine(rhs, -=) {
            fatalError()
        }
    }

    public static func +<S: TensorProtocol> (lhs: Self, rhs: S) -> Self where T == S.T {
        if let r = lhs.combine(rhs, +) {
            return r
        } else {
            fatalError()
        }
    }
    public static func -<S: TensorProtocol> (lhs: Self, rhs: S) -> Self where T == S.T {
        if let r = lhs.combine(rhs, -) {
            return r
        } else {
            fatalError()
        }
    }

    public static var zero: Self {
        .init(buffer: .init(capacity: 1, initialValue: .zero), shape: .init(shape: .one, stride: .zero))
    }
}

extension Tensor: Comparable where T: Comparable {
    public func compareLexicographically(rhs: Self) -> Int {
        if let r = rhs.broadcast(to: shape) {
            var it1 = shape.makeIterator()
            var it2 = r.shape.makeIterator()
            return buffer.apply { lptr in
                rhs.buffer.apply { rptr in
                    while let i = it1.next(), let j = it2.next() {
                        if lptr[Int(i)] != rptr[Int(j)] {
                            return lptr[Int(i)] < rptr[Int(j)] ? -1 : 1
                        }
                    }
                    return 0
                }
            }
        } else {
            fatalError()
        }
    }
    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.compareLexicographically(rhs: rhs) < 0
    }
    public static func > (lhs: Self, rhs: Self) -> Bool {
        lhs.compareLexicographically(rhs: rhs) > 0
    }
    public static func <= (lhs: Self, rhs: Self) -> Bool {
        lhs.compareLexicographically(rhs: rhs) <= 0
    }
    public static func >= (lhs: Self, rhs: Self) -> Bool {
        lhs.compareLexicographically(rhs: rhs) >= 0
    }
}

infix operator ⨂ : MultiplicationPrecedence

extension TensorProtocol where T: Numeric {
    public var normSquared: Scalar<T> {
        .init(elements.reduce(T.zero, {$0+$1*$1}))
    }
    public var maxMagnitude: T.Magnitude {
        elements.reduce(T.Magnitude.zero, {Swift.max($0, $1.magnitude)})
    }
    public var sumMagnitude: T.Magnitude {
        elements.reduce(T.Magnitude.zero, {$0+$1.magnitude})
    }

    public var magnitude: Tensor<Types, IndexType, T.Magnitude> {
        applyUnary(\.magnitude)
    }
}

extension Tensor: Numeric where T: Numeric {
    public typealias Magnitude = Tensor<Types, IndexType, T.Magnitude>

    public init?<TT>(exactly source: TT) where TT: BinaryInteger {
        if let s = T(exactly: source) {
            self.init(buffer: .init(capacity: 1, initialValue: s), shape: .init(shape: .one, stride: .zero))
        } else {
            return nil
        }
    }

    public static func * (lhs: Self, rhs: Self) -> Self {
        lhs.applyBinaryWithBroadcast(rhs: rhs, operator: *)
    }
    public static func *= (lhs: inout Self, rhs: Self) {
        if let s = rhs.broadcast(to: lhs.shape) {
            lhs.applyBinary(rhs: s, *=)
        } else {
            fatalError()
        }
    }
    public static func * (lhs: Self, rhs: T) -> Self {
        lhs.applyUnary({$0*rhs})
    }
    public static func *<S: TensorProtocol, R: TensorProtocol> (lhs: Self, rhs: S) -> R where T == S.T, T == R.T {
        lhs ⨂ rhs
    }
    public static func ⨂<S: TensorProtocol, R: TensorProtocol> (lhs: Self, rhs: S) -> R where T == S.T, T == R.T {
        if let r: R = lhs.multiply(rhs: rhs, zero: .zero, multiplyOperator: *, sumOperator: +) {
            return r
        } else {
            fatalError()
        }
    }
}

extension Tensor: SignedNumeric where T: SignedNumeric {
    public mutating func negate() {
        apply {$0 = -$0}
    }
    public var negative: Self {
        applyUnary(-)
    }
}

extension Tensor: Strideable where T: Strideable {
   public typealias Stride = Tensor<Types, IndexType, T.Stride>
   public func distance(to other: Self) -> Stride {
       if count < other.count {
           if let r: Stride = other.combine2(self, { $1.distance(to: $0)}) {
               return r
           } else {
               fatalError()
           }
       }
       if let r: Stride = combine2(other, { $0.distance(to: $1)}) {
           return r
       } else {
           fatalError()
       }
   }

   public func advanced(by n: Stride) -> Self {
       if let r: Self = combine(n, {$0.advanced(by: $1)}) {
           return r
       } else {
           fatalError()
       }
   }
}

extension Tensor: ScalarValueProtocol where Types == NilTypeList {
    public var value: T {
        get {
            buffer.readable.pointee
        }
        _modify {
            yield &buffer.writable.pointee
        }
    }
}

extension Tensor: BinaryInteger where T: BinaryInteger {

    public static func <<= <RHS>(lhs: inout Self, rhs: RHS) where RHS: BinaryInteger {
        lhs.apply {$0 <<= rhs}
    }

    public static func >>= <RHS>(lhs: inout Self, rhs: RHS) where RHS: BinaryInteger {
        lhs.apply {$0 >>= rhs}
    }

    public static prefix func ~ (x: Self) -> Self {
        x.applyUnary(~)
    }

    public static func /= (lhs: inout Self, rhs: Self) {
        lhs.applyBinary(rhs: rhs, /=)
    }

    public init<TT>(_ source: TT) where TT: BinaryInteger {
        self.init(buffer: .init(capacity: 1, initialValue: T(source)), shape: .init(shape: 1))
    }
    public init<TT>(clamping source: TT) where TT: BinaryInteger {
        self.init(buffer: .init(capacity: 1, initialValue: T(clamping: source)), shape: .init(shape: 1))
    }

    public init<TT>(_ source: TT) where TT: BinaryFloatingPoint {
        self.init(buffer: .init(capacity: 1, initialValue: T(source)), shape: .init(shape: 1))
    }
    public init<TT>(truncatingIfNeeded source: TT) where TT: BinaryInteger {
        self.init(buffer: .init(capacity: 1, initialValue: T(truncatingIfNeeded: source)), shape: .init(shape: 1))
    }
    public init<TT>(_truncatingBits source: TT) where TT: BinaryInteger {
        self.init(buffer: .init(capacity: 1, initialValue: T(clamping: source)), shape: .init(shape: 1))
    }
    public init?<TT>(exactly source: TT) where TT: BinaryFloatingPoint {
        if let value: T = T(exactly: source) {
            self.init(buffer: .init(capacity: 1, initialValue: value), shape: .init(shape: 1))
        } else {
            return nil
        }
    }

    public var words: [UInt] {
        elements.flatMap(\.words)
    }

    public static var isSigned: Bool {
        T.isSigned
    }
    public var bitWidth: Int {
        elements.reduce(0, {$0+$1.bitWidth})
    }

    public var trailingZeroBitCount: Int {
        elements.reduce(-1, {$0 == -1 ? $1.trailingZeroBitCount : Swift.min($0, $1.trailingZeroBitCount)})
    }

    public static func / (lhs: Self, rhs: Self) -> Self {
        if let r = lhs.combine(rhs, /) {
            return r
        } else {
            fatalError()
        }
    }

    public static func % (lhs: Self, rhs: Self) -> Self {
        if let r = lhs.combine(rhs, %) {
            return r
        } else {
            fatalError()
        }
    }

    public static func %= (lhs: inout Self, rhs: Self) {
        if !lhs.assignCombine(rhs, %=) {
            fatalError()
        }
    }

    public static func &= (lhs: inout Self, rhs: Self) {
        if !lhs.assignCombine(rhs, &=) {
            fatalError()
        }
    }

    public static func |= (lhs: inout Self, rhs: Self) {
        if !lhs.assignCombine(rhs, |=) {
            fatalError()
        }
    }

    public static func ^= (lhs: inout Self, rhs: Self) {
        if !lhs.assignCombine(rhs, ^=) {
            fatalError()
        }
    }
}
extension Tensor: SignedInteger where T: SignedInteger {

}
extension Tensor: UnsignedInteger where T: UnsignedInteger {

}

extension Tensor: LosslessStringConvertible where T: LosslessStringConvertible {
    public init?(_ description: String) {
        if Types.self == NilTypeList.self {
            if let r = T(description) {
                self.init(buffer: .init(capacity: 1, initialValue: r), shape: .init(shape: 1))
                return
            } else {
                return nil
            }
        }
        var level = 0
        var list: [String] = []
        var current: String.Index = description.startIndex
        for i in description.indices {
            switch description[i] {
            case "[":level += 1
                if level == 1 {
                    current = description.index(after: i)
                }
                break
            case "]":level -= 1
                if level == 0 {
                    list.append(String(description[current..<i]))
                    current = description.index(after: i)
                }
                break
            case ",":
                if level == 1 {
                    list.append(String(description[current..<i]))
                    current = description.index(after: i)
                }
                break
            default:
                break
            }
        }
        let elements: [Tail] = list.compactMap {.init($0)}
        if elements.count == list.count {
            self.init(elements: elements)
        } else {
            return nil
        }
    }

}

extension Tensor: FixedWidthInteger where T: FixedWidthInteger {

    public init<TT>(_truncatingBits source: TT) where TT: FixedWidthInteger {
        self.init(buffer: .init(capacity: 1, initialValue: T(clamping: source)), shape: .init(shape: 1))
    }

    public static var min: Self {
        .init(buffer: .init(capacity: 1, initialValue: .min), shape: .init(shape: 1))
    }

    public static var max: Self {
        .init(buffer: .init(capacity: 1, initialValue: .max), shape: .init(shape: 1))
    }

    public static var bitWidth: Int {
        T.bitWidth
    }

    public func addingReportingOverflow(_ rhs: Self) -> (partialValue: Self, overflow: Bool) {
        var result = self
        var overflow = false
        if !result.assignCombine(rhs, {
            let v = $0.addingReportingOverflow($1)
            $0 = v.partialValue
            if v.overflow {
                overflow = true
            }
        }) {
            fatalError()
        }
        return (partialValue:result, overflow:overflow)
    }

    public func subtractingReportingOverflow(_ rhs: Self) -> (partialValue: Self, overflow: Bool) {
        var result = self
        var overflow = false
        if !result.assignCombine(rhs, {
            let v = $0.subtractingReportingOverflow($1)
            $0 = v.partialValue
            if v.overflow {
                overflow = true
            }
        }) {
            fatalError()
        }
        return (partialValue:result, overflow:overflow)
    }

    public func multipliedReportingOverflow(by rhs: Self) -> (partialValue: Self, overflow: Bool) {
        var result = self
        var overflow = false
        if !result.assignCombine(rhs, {
            let v = $0.multipliedReportingOverflow(by: $1)
            $0 = v.partialValue
            if v.overflow {
                overflow = true
            }
        }) {
            fatalError()
        }
        return (partialValue:result, overflow:overflow)
    }

    public func dividedReportingOverflow(by rhs: Self) -> (partialValue: Self, overflow: Bool) {
        var result = self
        var overflow = false
        if !result.assignCombine(rhs, {
            let v = $0.dividedReportingOverflow(by: $1)
            $0 = v.partialValue
            if v.overflow {
                overflow = true
            }
        }) {
            fatalError()
        }
        return (partialValue:result, overflow:overflow)
    }

    public func remainderReportingOverflow(dividingBy rhs: Self) -> (partialValue: Self, overflow: Bool) {
        var result = self
        var overflow = false
        if !result.assignCombine(rhs, {
            let v = $0.remainderReportingOverflow(dividingBy: $1)
            $0 = v.partialValue
            if v.overflow {
                overflow = true
            }
        }) {
            fatalError()
        }
        return (partialValue:result, overflow:overflow)
    }

    public func dividingFullWidth(_ dividend: (high: Self, low: Tensor<Types, IndexType, T.Magnitude>)) -> (quotient: Self, remainder: Self) {
        var quotient: Buffer<T> = .init(capacity: shape.count)
        var remainder: Buffer<T> = .init(capacity: shape.count)
        quotient.apply { qptr in
            remainder.apply { remptr in
                buffer.apply { rptr in
                    dividend.high.buffer.apply { hptr in
                        dividend.low.buffer.apply { lptr in
                            var it = shape.makeIterator()
                            var c = 0
                            var ith = dividend.high.shape.makeIterator()
                            var itl = dividend.low.shape.makeIterator()
                            while let i = it.next(), let j = ith.next(), let k = itl.next() {
                                let q = rptr[Int(i)].dividingFullWidth((high:hptr[Int(j)], low:lptr[Int(k)]))
                                remptr.advanced(by: c).initialize(to: q.remainder)
                                qptr.advanced(by: c).initialize(to: q.quotient)
                                c += 1
                            }
                        }
                    }
                }
            }
        }
        return (quotient:.init(buffer: quotient, shape: .init(shape: shape.shape)), remainder:.init(buffer: remainder, shape: .init(shape: shape.shape)))
    }

    public var nonzeroBitCount: Int {
        elements.reduce(0, {$0+$1.nonzeroBitCount})
    }

    public var leadingZeroBitCount: Int {
        elements.reduce(T.bitWidth, {Swift.min($0, $1.leadingZeroBitCount)})
    }

    public var byteSwapped: Self {
        applyUnary(\.byteSwapped)
    }

}
extension Tensor: FloatingPoint where T: FloatingPoint {
    public init<Source>(_ value: Source) where Source: BinaryInteger {
        self.init(buffer: .init(capacity: 1, initialValue: T(value)), shape: .init(shape: 1))
    }

    public init(_ value: Int) {
        self.init(buffer: .init(capacity: 1, initialValue: T(value)), shape: .init(shape: 1))
    }

    public init(signOf: Self, magnitudeOf: Self) {
        if let r = magnitudeOf.broadcast(to: signOf.shape) {
            var buffer: Buffer<T> = .init(capacity: signOf.shape.count)
            buffer.apply { rptr in
                signOf.buffer.apply { sptr in
                    magnitudeOf.buffer.apply { mptr in
                        var o = 0
                        var it1 = signOf.shape.makeIterator()
                        var it2 = r.shape.makeIterator()
                        while let i = it1.next(), let j = it2.next() {
                            rptr.advanced(by: o).initialize(to: T(signOf: sptr[Int(i)], magnitudeOf: mptr[Int(j)]))
                            o += 1
                        }
                    }
                }
            }
            self.init(buffer: buffer, shape: signOf.shape)
        } else {
            fatalError()
        }
    }

    public init(sign: FloatingPointSign, exponent: Tensor<Types, IndexType, T.Exponent>, significand: Self) {
        if let r = exponent.broadcast(to: significand.shape) {
            var buffer: Buffer<T> = .init(capacity: significand.shape.count)
            buffer.apply { rptr in
                exponent.buffer.apply { eptr in
                    significand.buffer.apply { sptr in
                        var o = 0
                        var it1 = significand.shape.makeIterator()
                        var it2 = r.shape.makeIterator()
                        while let i = it1.next(), let j = it2.next() {
                            rptr.advanced(by: o).initialize(to: T(sign: sign, exponent: eptr[Int(j)], significand: sptr[Int(i)]))
                            o += 1
                        }
                    }
                }
            }
            self.init(buffer: buffer, shape: significand.shape)
        } else {
            fatalError()
        }
    }
    public var ulp: Self {
        applyUnary(\.ulp)
    }
    public static var radix: Int {
        T.radix
    }

    public static var nan: Self {
        self.init(buffer: .init(capacity: 1, initialValue: .nan), shape: .init(shape: 1))
    }

    public static var signalingNaN: Self {
        self.init(buffer: .init(capacity: 1, initialValue: .signalingNaN), shape: .init(shape: 1))
    }

    public static var infinity: Self {
        self.init(buffer: .init(capacity: 1, initialValue: .infinity), shape: .init(shape: 1))
    }

    public static var greatestFiniteMagnitude: Self {
        self.init(buffer: .init(capacity: 1, initialValue: .greatestFiniteMagnitude), shape: .init(shape: 1))
    }

    public static var pi: Self {
        self.init(buffer: .init(capacity: 1, initialValue: .pi), shape: .init(shape: 1))
    }

    public static var leastNormalMagnitude: Self {
        self.init(buffer: .init(capacity: 1, initialValue: .leastNormalMagnitude), shape: .init(shape: 1))
    }

    public static var leastNonzeroMagnitude: Self {
        self.init(buffer: .init(capacity: 1, initialValue: .leastNonzeroMagnitude), shape: .init(shape: 1))
    }
    public var significand: Self {
        applyUnary(\.significand)
    }
    public mutating func addProduct(_ lhs: Self, _ rhs: Self) {
        if let l = lhs.broadcast(to: shape), let r = rhs.broadcast(to: shape) {
            var it = shape.makeIterator()
            var it1 = l.shape.makeIterator()
            var it2 = r.shape.makeIterator()
            buffer.apply { optr in
                lhs.buffer.apply { lptr in
                    rhs.buffer.apply { rptr in
                        while let i = it.next(), let j = it1.next(), let k = it2.next() {
                            optr[Int(i)].addProduct(lptr[Int(j)], rptr[Int(k)])
                        }
                    }
                }
            }
        } else {
            fatalError()
        }
    }
    public var nextUp: Self {
        applyUnary(\.nextUp)
    }

}

extension TensorProtocol where T: FloatingPoint {
    public mutating func round(_ rule: FloatingPointRoundingRule) {
        apply {$0.round(rule)}
    }

    public static func /= (lhs: inout Self, rhs: Self) {
        if !lhs.assignCombine(rhs, /=) {
            fatalError()
        }
    }

    public func avg<S: TensorProtocol>() -> S where S.T == T {
        var r: S = sum()
        let f = T(r.elementCount)/T(self.elementCount)
        r.apply {$0 *= f}
        return r
    }

    public func stddev<S: TensorProtocol>() -> S where Types: NonEmptyList, S.T == T {
        let r: Tensor<S.Types, S.IndexType, (T, T)> = reduceTensor(zero: (0, 0), op: {($0.0+$1, $0.1+$1*$1)})!
        let f = T(r.elementCount)/T(self.elementCount)
        if f == 1 {
            fatalError()
        }
        let f1 = f/(1 - f)
        return r.applyUnary {(($0.1-$0.0*$0.0*f)*f1).squareRoot()} as! S
    }

    public var exponent: Tensor<Types, IndexType, T.Exponent> {
        applyUnary(\.exponent)
    }

    public var sign: FloatingPointSign {
        let s: Set<FloatingPointSign> = .init(elements.map(\.sign))
        if s.count == 1 {
            return s.first!
        } else {
            fatalError()
        }
    }

    public static func / (lhs: Self, rhs: Self) -> Self {
        if let r = lhs.combine(rhs, /) {
            return r
        } else {
            fatalError()
        }
    }

    public mutating func formRemainder(dividingBy other: Self) {
        if !assignCombine(other, {$0.formRemainder(dividingBy: $1)}) {
            fatalError()
        }
    }

    public mutating func formTruncatingRemainder(dividingBy other: Self) {
        if !assignCombine(other, {$0.formTruncatingRemainder(dividingBy: $1)}) {
            fatalError()
        }
    }

    public mutating func formSquareRoot() {
        apply {$0.formSquareRoot()}
    }

    public func isEqual(to other: Self) -> Bool {
        if let r = compareAll(other, {$0.isEqual(to: $1)}) {
            return r
        } else {
            fatalError()
        }
    }

    public func isLess(than other: Self) -> Bool {
        if let r = compareAll(other, {$0.isLess(than: $1)}) {
            return r
        } else {
            fatalError()
        }
    }

    public func isLessThanOrEqualTo(_ other: Self) -> Bool {
        if let r = compareAll(other, {$0.isLessThanOrEqualTo($1)}) {
            return r
        } else {
            fatalError()
        }
    }

    public func isTotallyOrdered(belowOrEqualTo other: Self) -> Bool {
        if let r = compareAll(other, {$0.isTotallyOrdered(belowOrEqualTo: $1)}) {
            return r
        } else {
            fatalError()
        }
    }

    public var isNormal: Bool {
        for i in elements {
            if !i.isNormal {
                return false
            }
        }
        return true
    }

    public var isFinite: Bool {
        for i in elements {
            if !i.isFinite {
                return false
            }
        }
        return true
    }

    public var isZero: Bool {
        for i in elements {
            if !i.isZero {
                return false
            }
        }
        return true
    }

    public var isSubnormal: Bool {
        for i in elements {
            if i.isSubnormal {
                return true
            }
        }
        return false
    }

    public var isInfinite: Bool {
        for i in elements {
            if i.isInfinite {
                return true
            }
        }
        return false
    }

    public var isNaN: Bool {
        for i in elements {
            if i.isNaN {
                return true
            }
        }
        return false
    }

    public var isSignalingNaN: Bool {
        for i in elements {
            if i.isSignalingNaN {
                return true
            }
        }
        return false
    }

    public var isCanonical: Bool {
        for i in elements {
            if !i.isCanonical {
                return false
            }
        }
        return true
    }

    public var norm: Scalar<T> {
        .init(elements.reduce(T.zero, {$0.addingProduct($1, $1)}).squareRoot())
    }

    public typealias Exponent = Tensor<Types, IndexType, T.Exponent>

}

extension Tensor: BinaryFloatingPoint where T: BinaryFloatingPoint {
    public init(sign: FloatingPointSign, exponentBitPattern: Tensor<Types, IndexType, T.RawExponent>, significandBitPattern: Tensor<Types, IndexType, T.RawSignificand>) {
        if let r = exponentBitPattern.broadcast(to: significandBitPattern.shape) {
            var buffer: Buffer<T> = .init(capacity: significandBitPattern.shape.count)
            buffer.apply { rptr in
                exponentBitPattern.buffer.apply { eptr in
                    significandBitPattern.buffer.apply { sptr in
                        var o = 0
                        var it1 = significandBitPattern.shape.makeIterator()
                        var it2 = r.shape.makeIterator()
                        while let i = it1.next(), let j = it2.next() {
                            rptr.advanced(by: o).initialize(to: T(sign: sign, exponentBitPattern: eptr[Int(j)], significandBitPattern: sptr[Int(i)]))
                            o += 1
                        }
                    }
                }
            }
            self.init(buffer: buffer, shape: significandBitPattern.shape)
        } else {
            fatalError()
        }
    }
    public init(shape: [ScalarIndex], uniformValuesIn range: Range<T>) where T.RawSignificand: FixedWidthInteger {
        var g = SystemRandomNumberGenerator()
        self.init(shape: shape, uniformValuesIn: range, using: &g)
    }
    public init<R: RandomNumberGenerator>(shape: [ScalarIndex], uniformValuesIn range: Range<T>, using rng:inout R) where T.RawSignificand: FixedWidthInteger {
        self.init(shape: shape, function: {_ in T.random(in: range, using: &rng)})
    }

    public init<R: RandomNumberGenerator>(shape: [ScalarIndex], mean: T, stddev: T, using rng:inout R) where T.RawSignificand: FixedWidthInteger {
        var n: NormalDistribution<T, R> = .init(using: rng)
        self.init(shape: shape, function: {_ in
            (n.next()!*stddev)+mean
        })
        rng = n.rng
    }

    public static var exponentBitCount: Int {
        T.exponentBitCount
    }

    public static var significandBitCount: Int {
        T.significandBitCount
    }

    public var exponentBitPattern: Tensor<Types, IndexType, T.RawExponent> {
        applyUnary(\.exponentBitPattern)
    }

    public var significandBitPattern: Tensor<Types, IndexType, T.RawSignificand> {
        applyUnary(\.significandBitPattern)
    }

    public var binade: Self {
        applyUnary(\.binade)
    }

    public var significandWidth: Int {
        elements.reduce(0, {$0 + $1.significandWidth})
    }
    public typealias RawSignificand = Tensor<Types, IndexType, T.RawSignificand>
    public typealias RawExponent = Tensor<Types, IndexType, T.RawExponent>
}

extension Tensor: ElementaryFunctions where T: ElementaryFunctions {
    public static func exp(_ x: Self) -> Self {
        x.mapTensor(T.exp)
    }
    public static func expMinusOne(_ x: Self) -> Self {
        x.mapTensor(T.expMinusOne)
    }
    public static func cosh(_ x: Self) -> Self {
        x.mapTensor(T.cosh)
    }
    public static func sinh(_ x: Self) -> Self {
        x.mapTensor(T.sinh)
    }
    public static func tanh(_ x: Self) -> Self {
        x.mapTensor(T.tanh)
    }
    public static func cos(_ x: Self) -> Self {
        x.mapTensor(T.cos)
    }
    public static func sin(_ x: Self) -> Self {
        x.mapTensor(T.sin)
    }
    public static func tan(_ x: Self) -> Self {
        x.mapTensor(T.tan)
    }
    public static func log(_ x: Self) -> Self {
        x.mapTensor {T.log($0)}
    }
    public static func log(onePlus x: Self) -> Self {
        x.mapTensor(T.log(onePlus: ))
    }
    public static func acosh(_ x: Self) -> Self {
        x.mapTensor(T.acosh)
    }
    public static func asinh(_ x: Self) -> Self {
        x.mapTensor(T.asinh)
    }
    public static func atanh(_ x: Self) -> Self {
        x.mapTensor(T.atanh)
    }
    public static func acos(_ x: Self) -> Self {
        x.mapTensor(T.acos)
    }
    public static func asin(_ x: Self) -> Self {
        x.mapTensor(T.asin)
    }
    public static func atan(_ x: Self) -> Self {
        x.mapTensor(T.atan)
    }
    public static func pow(_ x: Self, _ y: Self) -> Self {
        if let r = x.combine(y, T.pow) {
            return r
        } else {
            fatalError()
        }
    }
    public static func pow(_ x: Self, _ n: Int) -> Self {
        x.mapTensor({T.pow($0, n)})
    }
    public static func sqrt(_ x: Self) -> Self {
        x.mapTensor(T.sqrt)
    }
    public static func root(_ x: Self, _ n: Int) -> Self {
        x.mapTensor({T.root($0, n)})
    }
}
extension Tensor: RealFunctions where T: RealFunctions {
    public static func atan2(y: Self, x: Self) -> Self {
        if let r = x.combine(y, T.atan2(y:x:)) {
            return r
        } else {
            fatalError()
        }
    }
    public static func erf(_ x: Self) -> Self {
        x.mapTensor(T.erf)
    }
    public static func erfc(_ x: Self) -> Self {
        x.mapTensor(T.erfc)
    }
    public static func exp2(_ x: Self) -> Self {
        x.mapTensor(T.exp2)
    }
    public static func exp10(_ x: Self) -> Self {
        x.mapTensor(T.exp10)
    }
    public static func hypot(_ x: Self, _ y: Self) -> Self {
        if let r = x.combine(y, {T.hypot($0, $1)}) {
            return r
        } else {
            fatalError()
        }
    }
    public static func gamma(_ x: Self) -> Self {
        x.mapTensor(T.gamma)
    }
    public static func log2(_ x: Self) -> Self {
        x.mapTensor(T.log2)
    }
    public static func log10(_ x: Self) -> Self {
        x.mapTensor(T.log10)
    }
    public static func logGamma(_ x: Self) -> Self {
        x.mapTensor(T.logGamma)
    }
    public static func signGamma(_ x: Self) -> FloatingPointSign {
        let s: Set<FloatingPointSign> = x.elements.reduce(into: []) {
            $0.insert(T.signGamma($1))
            if $0.count > 1 {
                fatalError()
            }
        }
        return s.first!
    }
    public static func _mulAdd(_ a: Self, _ b: Self, _ c: Self) -> Self {
        var result = c
        if let sa = a.shape.broadcast(like: a.shape) {
            if let sb = b.shape.broadcast(like: a.shape) {
                var it = c.shape.makeIterator()
                var it1 = sa.shape.makeIterator()
                var it2 = sb.shape.makeIterator()
                result.buffer.apply { optr in
                    a.buffer.apply {lptr in
                        b.buffer.apply { rptr in
                            while let i = it.next(), let j = it1.next(), let k = it2.next() {
                                optr[Int(i)] = T._mulAdd(optr[Int(i)], lptr[Int(j)], rptr[Int(k)])
                            }
                        }
                    }
                }
                return result
            }
        }
        fatalError("shape mismatch \(a.size) \(b.size) \(c.size)")
    }
}
extension TensorProtocol where Types: NonEmptyList, Types.Tail: NonEmptyList {
    public typealias Transpose = Tensor<TypeList<Types.Tail.Head, TypeList<Types.Head, Types.Tail.Tail>>, IndexType, T>
    public var Transposed: Transpose {
        get {
            transpose()
        }
        _modify {
            var r: Transpose = transpose()
            yield &r
            if r.elementCount != elementCount {
                fatalError()
            }
            buffer.transfer(range: 0..<elementCount, from: r.buffer)
        }
    }
}

extension Tensor where Types: NonEmptyList, Types.Tail: NonEmptyList, Types.Tail.Tail==NilTypeList, T: Numeric {
    public init(diag: [T]) {
        self.init(shape: [ScalarIndex(diag.count), ScalarIndex(diag.count)], function: {
            $0[0] == $0[1] ? diag[Int($0[0])] : .zero
        })
    }
}

extension Tensor where T: Numeric, Types: NonEmptyList, Types.Tail: NonEmptyList {
    public static func eye(shape size: ScalarIndex...) -> Self {
        let shape: Shape<Types, IndexType> = .init(shape: size.reversed())
        var r: Self = .init(buffer: .init(capacity: shape.count), shape: shape)
        var ix0 = 0
        var ix1 = 1
        for i in 1..<Self.order {
            for j in 0..<i {
                if shape.shape[i] == shape.shape[j] && (shape.shape[ix0] != shape.shape[ix1] || shape.shape[i]*shape.shape[i] > shape.shape[ix0]*shape.shape[ix1]) {
                    ix0 = i
                    ix1 = j
                }
            }
        }
        var it = shape.makeIterator()
        r.buffer.apply { ptr in
            var val: T = 1
            while let n = it.next() {
                ptr[Int(n)] = val
                val = it.index[ix0] == it.index[ix1] ? 1 : 0
            }
        }
        return r
    }
}
extension Tensor: AlgebraicField where T: AlgebraicField {
    public static func /= (a: inout Self, b: Self) {
        a.applyBinary(rhs: b, /=)
    }
}

extension Sequence where Element: TensorProtocol {
    var flattened: TensorSequence<Self> {
        .init(data: self)
    }
}

extension MutableCollection where Element: TensorProtocol {
    mutating func update(from: TensorSequence<Self>) {
        self = from .data
    }
}

struct TensorSequence<A: Sequence>: Sequence where A.Element: TensorProtocol {
    var data: A
    struct LocalIterator<A: IteratorProtocol>: IteratorProtocol where A.Element: TensorProtocol {
        var sequences: A
        var it: TensorElementCollection<A.Element.Types, A.Element.IndexType, A.Element.T>.Iterator?
        init(sequences: A) {
            self.sequences = sequences
            self.it = self.sequences.next()?.elements.makeIterator()
        }
        mutating func next() -> A.Element.T? {
            while true {
                if let n = it?.next() {
                    return n
                }
                it = sequences.next()?.elements.makeIterator()
                if it == nil {
                    return nil
                }
            }
        }
    }
    func makeIterator() -> LocalIterator<A.Iterator> {
        .init(sequences: data.makeIterator())
    }
}

extension TensorSequence: Collection where A: Collection {
    var count: Int {
        data.reduce(0, {$0+$1.elementCount})
    }
    func index(after i: Index) -> Index {
        if i.b < data[i.a].elements.endIndex {
            let j = data[i.a].elements.index(after: i.b)
            if j < data[i.a].elements.endIndex {
                return .init(a: i.a, b: j)
            }
        }
        let k = data.index(after: i.a)
        if k == data.endIndex {
            return endIndex
        }
        return .init(a: k, b: data[k].elements.startIndex)
    }
    func index(_ i: Index, offsetBy distance: Int) -> Index {
        let x = data[i.a].elements.distance(from: i.b, to: data[i.a].elements.endIndex)
        if i.b < data[i.a].elements.endIndex && distance < x {
            let j = data[i.a].elements.index(i.b, offsetBy: distance)
            if j < data[i.a].elements.endIndex {
                return .init(a: i.a, b: j)
            }
        }
        var d = distance - x
        var a = data.index(after: i.a)
        if a >= data.endIndex {
            return endIndex
        }
        var b = data[a].elements.startIndex
        while a < data.endIndex && data[a].elementCount < d {
            d -= data[a].elementCount
            a = data.index(after: a)
        }
        if a >= data.endIndex {
            return endIndex
        }
        b = data[a].elements.index(data[a].elements.startIndex, offsetBy: d)
        if b >= data[a].elements.endIndex {
            return endIndex
        }
        return .init(a: a, b: b)
    }
    subscript(position: Index) -> A.Element.T {
        _read {
            yield data[position.a].elements[position.b]
        }
    }
    struct Index: Comparable {
        static func < (lhs: TensorSequence<A>.Index, rhs: TensorSequence<A>.Index) -> Bool {
            lhs.a < rhs.a || (lhs.a == rhs.a && lhs.b < rhs.b)
        }

        let a: A.Index
        let b: TensorElementCollection<A.Element.Types, A.Element.IndexType, A.Element.T>.Index

    }
    var startIndex: Index {
        .init(a: data.startIndex, b: data.first!.elements.startIndex)
    }
    var endIndex: Index {
        .init(a: data.endIndex, b: data.first!.elements.startIndex)
    }
}

extension TensorSequence: MutableCollection where A: MutableCollection {
    subscript(position: Index) -> A.Element.T {
        _read {
            yield data[position.a][position.b]
        }
        _modify {
            yield &data[position.a][position.b]
        }
        set {
            data[position.a][position.b]=newValue
        }
    }
}

extension TensorSequence: CustomStringConvertible where A.Element.T: CustomStringConvertible {
    var description: String {
        "[\(map(\.description).joined(separator: ", "))]"
    }
}

public typealias Wrapping<V, S: TensorProtocol> = Tensor<TypeList<V, S.Types>, S.IndexType, S.T>
public typealias Scalar<Element> = Tensor<NilTypeList, SIMD1, Element>
public typealias Vector<R, Element> = Tensor<TypeList<R, NilTypeList>, SIMD1, Element>
public typealias Matrix<R, C, Element> = Tensor<TypeList<R, TypeList<C, NilTypeList>>, SIMD2<ScalarIndex>, Element>
public typealias Tensor3<T, R, C, Element> = Tensor<TypeList<T, TypeList<R, TypeList<C, NilTypeList>>>, SIMD3<ScalarIndex>, Element>
public typealias Tensor4<V, T, R, C, Element> = Tensor<TypeList<V, TypeList<T, TypeList<R, TypeList<C, NilTypeList>>>>, SIMD4<ScalarIndex>, Element>
public typealias Tensor5<W, V, T, R, C, Element> = Wrapping<W, Tensor4<V, T, R, C, Element>>
public typealias Tensor6<U, W, V, T, R, C, Element> = Wrapping<U, Tensor5<W, V, T, R, C, Element>>
public typealias Tensor7<U, W, V, T, R, C, Element> = Wrapping<U, Tensor6<U, W, V, T, R, C, Element>>
public typealias Tensor8<Z, U, W, V, T, R, C, Element> = Wrapping<Z, Tensor7<U, W, V, T, R, C, Element>>
