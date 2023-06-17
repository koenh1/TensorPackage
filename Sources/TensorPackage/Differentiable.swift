// TODO: isConstant
// makeConstant

import Foundation
import RealModule
import RegexBuilder

typealias CallBack<T: DifferentiableValue> = (header: Header<T>, value: T)

public protocol DifferentiableProtocol {
    associatedtype ValueType: DifferentiableValue
    var grad: ValueType { get }
    var value: ValueType { get set }
    var generation: UInt32 { get }
    func clearGradient()
    var gradient: Gradients<ValueType> { get }
    var isConstant: Bool { get }
    func gradients<C: Collection>(for parameters: C) -> Gradients<ValueType>  where C.Element: DifferentiableProtocol, C.Element.ValueType == ValueType
//    mutating func updateValue(stepSize: ValueType)
}

var globalGeneration: UInt32 = 0

public struct Gradients<ValueType: DifferentiableValue>: CustomStringConvertible {
    var result: [ValueType]
    init<C>(of f: Differentiable<C.Element.ValueType>, for parameters: C) where C: Collection, C.Element: DifferentiableProtocol, C.Element.ValueType == ValueType {
        globalGeneration += 1
        
        var list: [any HeaderProtocol] = []
        func listSorted(_ header: any HeaderProtocol) {
            if header.generation != globalGeneration {
                if header.isConstant {
                    return
                }
                header.generation = globalGeneration
                header.applyLeft(visitor: listSorted)
                header.applyRight(visitor: listSorted)
                list.append(header)
                header.clearGradient()
            }
        }
        listSorted(f.header)
        for i in parameters {
            i.clearGradient()
        }
        f.header.unitGradient()
        for x in list.reversed() {
            x.updateGrad()
        }
        result = parameters.map {$0.grad}
    }
    public func normSquared() -> ValueType where ValueType: Numeric {
        result.reduce(0, {$0+$1*$1})
    }
    public func update<C: MutableCollection>(stepSize: ValueType, parameters:inout C) where C.Element == ValueType, ValueType: Numeric {
        if parameters.count != result.count {
            fatalError()
        }
        var j = 0
        parameters.indices.forEach { i in
            parameters[i] += stepSize * result[j]
            j += 1
        }
    }
    public func update<C: MutableCollection>(stepSize: ValueType, parameters:inout C) where C.Element: DifferentiableProtocol, C.Element.ValueType==ValueType, ValueType: Numeric {
        if parameters.count != result.count {
            fatalError()
        }
        var j = 0
        parameters.indices.forEach { i in
            parameters[i].value += stepSize * result[j]
            j += 1
        }
    }
    public subscript<X: DifferentiableProtocol>(e: X) -> X.ValueType {
        e.generation == globalGeneration ? e.grad : .zero
    }
    public var description: String {
        result.debugDescription
    }
}
//struct Graph<ValueType: DifferentiableValue>: CustomStringConvertible {
//    var nodes:[(id: ObjectIdentifier, gradient: ValueType)] = []
//    var links:[(parent: ObjectIdentifier, child: ObjectIdentifier)] = []
//    init(_ header: any HeaderProtocol) {
//        var list: [any HeaderProtocol] = []
//        globalGeneration += 1
//        func listSorted(_ header: any HeaderProtocol) {
//            if header.generation != globalGeneration {
//                header.clearGradient()
//                header.applyLeft(visitor: listSorted)
//                header.applyRight(visitor: listSorted)
//                list.append(header)
//            }
//        }
//        listSorted(header)
//        for i in list {
//            i.clearGradient()
//        }
//        header.unitGradient()
//        for x in list.reversed() {
//            x.updateGrad()
//        }
//        for node in list {
//            let id = ObjectIdentifier(node)
////            let grad = node.grad
////            let r:(id: ObjectIdentifier, gradient: Any) = (id:id,gradient:grad)
//            node.applyLeft(visitor: {h in links.append((parent:ObjectIdentifier(node), child:ObjectIdentifier(h)))})
//            node.applyRight(visitor: {h in links.append((parent:ObjectIdentifier(node), child:ObjectIdentifier(h)))})
////            nodes.append(r)
//        }
//    }
//    var description: String {
//        if #available(macOS 13.0, *) {
//            let regex = Regex  {
//                "ObjectIdentifier(0"
//                Capture {
//                    "x"
//                    OneOrMore {CharacterClass(.anyOf(")")).inverted}
//                }
//                ")"
//            }
//            func rep(_ id: ObjectIdentifier) -> String {
//                id.debugDescription.replacing(regex, with: \.1)
//            }
//
//            return "digraph G {\nrankdir=LR\n"
//            +
//            nodes.map {"\(rep($0.id)) [shape=record, label=\"{{\($0.gradient)}}\"];"}.joined(separator: "\n")
//            + "\n" +
//            links.map {"\(rep($0.parent))->\(rep($0.child));"}.joined(separator: "\n")
//            + "\n}"
//        } else {
//            return ""
//        }
//    }
//}
public protocol DifferentiableValue: AdditiveArithmetic {
    static var one: Self { get }
}
extension ExpressibleByIntegerLiteral {
    public static var one: Self {
        1
    }
}
extension Int: DifferentiableValue {}
extension Float: DifferentiableValue {}
extension Double: DifferentiableValue {}
protocol HeaderProtocol: AnyObject {
    associatedtype ValueType
    func applyLeft(visitor: (any HeaderProtocol) -> Void)
    func applyRight(visitor: (any HeaderProtocol) -> Void)
    func updateGrad()
    var generation: UInt32 { get set }
    var isConstant: Bool { get }
    func clearGradient()
    func unitGradient()
    var grad: ValueType { get set }
}
class AbstractHeader<ValueType>: HeaderProtocol where ValueType: DifferentiableValue {
    init(_ g: ValueType) {
        grad = g
    }
    convenience init() {
        self.init(.zero)
    }
    var grad: ValueType
    var generation: UInt32 = 0
    func updateGrad() {
    }
    func unitGradient() {
        grad = .one
    }
    func clearGradient() {
        generation = globalGeneration
        grad = .zero
    }
    var isConstant: Bool {
        Self.self == ConstantHeader<ValueType>.self
    }
    func applyLeft(visitor: (any HeaderProtocol) -> Void) {
    }
    func applyRight(visitor: (any HeaderProtocol) -> Void) {
    }
    func add(_ v: ValueType) where ValueType: DifferentiableValue {
        grad = v + grad
    }
}
class Header<ValueType: DifferentiableValue>: AbstractHeader<ValueType> {
}

private class ConstantHeader<ValueType: DifferentiableValue>: Header<ValueType> {}
private class CastHeader<ValueType: BinaryFloatingPoint&DifferentiableValue, InputType: BinaryFloatingPoint&DifferentiableValue>: Header<ValueType> {
    init(delegate: Header<InputType>) {
        self.delegate = delegate
        super.init(.zero)
    }
    var delegate: Header<InputType>
    override var generation: UInt32 {
        get {
            delegate.generation
        }
        set {
            delegate.generation = newValue
        }
    }
    override func applyLeft(visitor: (any HeaderProtocol) -> Void) {
        delegate.applyLeft(visitor: visitor)
    }
    override func clearGradient() {
        delegate.clearGradient()
    }
    override func unitGradient() {
        delegate.unitGradient()
    }
    override func updateGrad() {
        delegate.updateGrad()
        grad = ValueType(delegate.grad)
    }
    override var isConstant: Bool {
        delegate.isConstant
    }
}
private class CastIntHeader<ValueType: BinaryFloatingPoint&DifferentiableValue, InputType: BinaryInteger&DifferentiableValue>: Header<ValueType> {
    init(delegate: Header<InputType>) {
        self.delegate = delegate
        super.init(.zero)
    }
    var delegate: Header<InputType>
    override var generation: UInt32 {
        get {
            delegate.generation
        }
        set {
            delegate.generation = newValue
        }
    }
    override func applyLeft(visitor: (any HeaderProtocol) -> Void) {
        delegate.applyLeft(visitor: visitor)
    }
    override func clearGradient() {
        delegate.clearGradient()
    }
    override func unitGradient() {
        delegate.unitGradient()
    }
    override func updateGrad() {
        delegate.updateGrad()
        grad = ValueType(delegate.grad)
    }
    override var isConstant: Bool {
        delegate.isConstant
    }
}
private class Header1<ValueType: DifferentiableValue>: Header<ValueType> {
    let back: (ValueType, UnsafePointer<(header: Header<ValueType>, value: ValueType)>) -> Void
    var data:(header: Header<ValueType>, value: ValueType)
    init(_ p: (Header<ValueType>, ValueType), back:@escaping (ValueType, UnsafePointer<(header: Header<ValueType>, value: ValueType)>) -> Void) {
        self.data = p
        self.back = back
        super.init(.zero)
    }
    override func applyLeft(visitor: (any HeaderProtocol) -> Void) {
        visitor(data.header)
    }
    override func updateGrad() {
        back(grad, &data)
    }
}
private class Header2<ValueType: DifferentiableValue>: Header<ValueType> {
    let back: (ValueType, UnsafePointer<(header: Header<ValueType>, value: ValueType)>) -> Void
    var data1:(header: Header<ValueType>, value: ValueType)
    var data2:(header: Header<ValueType>, value: ValueType)
    init(_ p1: (Header<ValueType>, ValueType), _ p2: (Header<ValueType>, ValueType), back:@escaping (ValueType, UnsafePointer<(header: Header<ValueType>, value: ValueType)>) -> Void) {
        self.data1 = p1
        self.data2 = p2
        self.back = back
        super.init(.zero)
    }
    override func applyLeft(visitor: (any HeaderProtocol) -> Void) {
        visitor(data1.header)
    }
    override func applyRight(visitor: (any HeaderProtocol) -> Void) {
        visitor(data2.header)
    }
    override func updateGrad() {
        [data1, data2].withUnsafeBufferPointer {
            back(grad, $0.baseAddress!)
        }
    }
}
private class Header3<ValueType: DifferentiableValue>: Header<ValueType> {
    let back: (ValueType, UnsafePointer<(header: Header<ValueType>, value: ValueType)>) -> Void
    var data1:(header: Header<ValueType>, value: ValueType)
    var data2:(header: Header<ValueType>, value: ValueType)
    var data3:(header: Header<ValueType>, value: ValueType)
    init(_ p1: (Header<ValueType>, ValueType), _ p2: (Header<ValueType>, ValueType), _ p3: (Header<ValueType>, ValueType), back:@escaping (ValueType, UnsafePointer<(header: Header<ValueType>, value: ValueType)>) -> Void) {
        self.data1 = p1
        self.data2 = p2
        self.data3 = p3
        self.back = back
        super.init(.zero)
    }
    override func applyLeft(visitor: (any HeaderProtocol) -> Void) {
        visitor(data1.header)
        visitor(data2.header)
    }
    override func applyRight(visitor: (any HeaderProtocol) -> Void) {
        visitor(data3.header)
    }
    override func updateGrad() {
        [data1, data2, data3].withUnsafeBufferPointer {
            back(grad, $0.baseAddress!)
        }
    }
}

private func mul_<ValueType: SignedNumeric>(_ g: ValueType, _ pred: UnsafePointer<CallBack<ValueType>>) {
    pred[0].header.add(pred[1].value*g)
    pred[1].header.add(pred[0].value*g)
}

private func plus_<ValueType>(_ g: ValueType, _ pred: UnsafePointer<CallBack<ValueType>>) {
    pred[0].header.add(g)
    pred[1].header.add(g)
}
private func minus_<ValueType: SignedNumeric>(_ g: ValueType, _ pred: UnsafePointer<CallBack<ValueType>>) {
    pred[0].header.add(g)
    pred[1].header.add(-g)
}

public struct Differentiable<ValueType: DifferentiableValue>: DifferentiableProtocol {

    mutating func updateValue(stepSize: ValueType) where ValueType: Numeric {
        value += stepSize*grad
    }

    func unitGradient() {
        header.unitGradient()
    }
    public var value: ValueType
    public var grad: ValueType {
        header.grad
    }
    public var generation: UInt32 {
        header.generation
    }
    var callback: CallBack<ValueType> {
        (header:header as! Header<ValueType>, value:value)
    }
    typealias BackProp = (ValueType, UnsafePointer<CallBack<ValueType>>) -> Void
    let header: AbstractHeader<ValueType>
    init(value: ValueType, header: AbstractHeader<ValueType>) {
        self.value = value
        self.header = header
    }
    init(value: ValueType, predecessors: Self, back:@escaping BackProp) {
        if predecessors.isConstant {
            self.init(value: value, header: Header())
        } else {
            self.init(value: value, header: Header1(predecessors.callback, back: back))
        }
    }
    init(value: ValueType, predecessors: (Self, Self), back:@escaping BackProp) {
        self.init(value: value, header: Header2<ValueType>(predecessors.0.callback, predecessors.1.callback, back: back))
    }
    fileprivate init(value: ValueType, predecessors: (Self, Self, Self), back:@escaping BackProp) {
        self.init(value: value, header: Header3(predecessors.0.callback, predecessors.1.callback, predecessors.2.callback, back: back))
    }
    public init(value: ValueType) {
        self.value = value
        header = Header<ValueType>()
    }
    public init(constant value: ValueType) {
        self.value = value
        header = ConstantHeader()
    }
    public var asConstant: Self {
        .init(constant: value)
    }
    public var asParameter: Self {
        isConstant ? .init(value: value) : self
    }
    public var isConstant: Bool {
        header.isConstant
    }
    public var gradient: Gradients<ValueType> {
        .init(of: self, for: [Self]())
    }
    public func gradients<C: Collection>(for parameters: C) -> Gradients<ValueType>  where C.Element: DifferentiableProtocol, C.Element.ValueType == ValueType {
        .init(of: self, for: parameters)
    }
//    public var graph: String {
//        Graph<ValueType>(header).description
//    }
    public func clearGradient() {
        header.clearGradient()
    }
}

extension Differentiable: CustomStringConvertible where ValueType: CustomStringConvertible {
    public var description: String {
        value.description
    }
}
extension Differentiable: Equatable where ValueType: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.value == rhs.value
    }
}
extension Differentiable: Hashable where ValueType: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(value)
    }
}
extension Differentiable: Comparable where ValueType: Comparable {
    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.value<rhs.value
    }
}
extension Differentiable: ExpressibleByIntegerLiteral where ValueType: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: ValueType.IntegerLiteralType) {
        self.init(constant: ValueType(integerLiteral: value))
    }
}
extension Differentiable: ExpressibleByFloatLiteral where ValueType: ExpressibleByFloatLiteral {
    public init(floatLiteral value: ValueType.FloatLiteralType) {
        self.init(constant: ValueType(floatLiteral: value))
    }
}
extension Differentiable: AdditiveArithmetic where ValueType: AdditiveArithmetic&SignedNumeric {
    public static func - (lhs: Self, rhs: Self) -> Self {
        let v = lhs.value - rhs.value
        let c1 = lhs.isConstant
        let c2 = rhs.isConstant
        if c1||c2 {
            return c1&&c2 ? .init(constant: v) : .init(value: v, header: c1 ? rhs.header : lhs.header)
        }
        return .init(value: v, predecessors: (lhs, rhs), back: minus_(_:_:))
    }
    public static func + (lhs: Self, rhs: Self) -> Self {
        let v = lhs.value + rhs.value
        let c1 = lhs.isConstant
        let c2 = rhs.isConstant
        if c1||c2 {
            return c1&&c2 ? .init(constant: v) : .init(value: v, header: c1 ? rhs.header : lhs.header)
        }
        return .init(value: v, predecessors: (lhs, rhs), back: plus_(_:_:))
    }
    public static var zero: Self {
        .init(constant: .zero)
    }
}

extension Differentiable: BinaryInteger where ValueType: BinaryInteger, ValueType.Magnitude==ValueType, ValueType.Stride==ValueType {
    public typealias Words = ValueType.Words

    public init?<T>(exactly source: T) where T: BinaryFloatingPoint {
        if let v = ValueType(exactly: source) {
            self.init(value: v)
        } else {
            return nil
        }
    }
    public init<T>(_ source: T) where T: BinaryFloatingPoint {
        self.init(value: ValueType(source))
    }
    public static func <<= <RHS>(lhs: inout Self, rhs: RHS) where RHS: BinaryInteger {
        lhs = lhs << rhs
    }
    public static func << <RHS>(lhs: Self, rhs: RHS) -> Self where RHS: BinaryInteger {
        if lhs.value == .zero {
            return lhs
        }
        let v = lhs.value << rhs
        return .init(value: v, predecessors: lhs, back: { g, pred in
            pred[0].header.add((ValueType(1)<<rhs)*g)
        })
    }
    public static func >>= <RHS>(lhs: inout Self, rhs: RHS) where RHS: BinaryInteger {
        lhs = lhs >> rhs
    }
    public static func >> <RHS>(lhs: Self, rhs: RHS) -> Self where RHS: BinaryInteger {
        if rhs == .zero {
            return lhs
        }
        return .init(value: lhs.value >> rhs)
    }
    public static prefix func ~ (x: Self) -> Self {
        .init(value: ~x.value)
    }
    public static func /= (a: inout Self, b: Self) {
        a = a / b
    }
    public init<T>(clamping source: T) where T: BinaryInteger {
        self.init(value: ValueType(clamping: source))
    }
    public init<T>(_ source: T) where T: BinaryInteger {
        self.init(value: ValueType(source))
    }
    public init<T>(truncatingIfNeeded source: T) where T: BinaryInteger {
        self.init(value: ValueType(truncatingIfNeeded: source))
    }
    public static var isSigned: Bool {
        ValueType.isSigned
    }
    public var words: ValueType.Words {
        value.words
    }
    public var bitWidth: Int {
        value.bitWidth
    }
    public var trailingZeroBitCount: Int {
        value.trailingZeroBitCount
    }
    public static func / (a: Self, b: Self) -> Self {
        let q = a.value / b.value
        return .init(value: q, predecessors: (a, b), back: {g, pred in
            pred[0].header.add(g/pred[1].value)
            pred[1].header.add(-g*pred[0].value/(pred[1].value*pred[1].value))
        })
    }
    public static func % (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value % rhs.value)
    }
    public static func %= (lhs: inout Self, rhs: Self) {
        lhs = lhs % rhs
    }
    public static func &= (lhs: inout Self, rhs: Self) {
        lhs = lhs & rhs
    }
    public static func & (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value&rhs.value)
    }
    public static func |= (lhs: inout Self, rhs: Self) {
        lhs = lhs | rhs
    }
    public static func | (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value|rhs.value)
    }
    public static func ^= (lhs: inout Self, rhs: Self) {
        lhs = lhs ^ rhs
    }
    public static func ^ (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value^rhs.value)
    }
}

extension Differentiable: Numeric where ValueType: SignedNumeric&Comparable {
    public typealias Magnitude = Self

    public init?<T>(exactly source: T) where T: BinaryInteger {
        if let v = ValueType(exactly: source) {
            self.init(value: v)
        } else {
            return nil
        }
    }
    @_disfavoredOverload
    public var magnitude: Differentiable<ValueType> {
        .init(value: value < .zero ? -value : value, predecessors: self, back: { g, pred in
            pred[0].header.add(pred[0].value < .zero ? -g : g)
        })
    }
    public static func *= (lhs: inout Self, rhs: Self) {
         lhs = lhs * rhs
    }

    @_disfavoredOverload
    public static func * (lhs: Self, rhs: Self) -> Self {
        let v = lhs.value*rhs.value
        let c1 = lhs.isConstant
        let c2 = rhs.isConstant
        if c1||c2 {
            return c1&&c2 ? .init(constant: v) : .init(value: v, predecessors: c1 ? rhs : lhs, back: { g, pred in
                pred[0].header.add((c1 ? lhs.value : rhs.value)*g)
            })
        }
        return .init(value: v, predecessors: (lhs, rhs), back: mul_(_:_:))
    }
    private static func powImpl(lhs: ValueType, rhs: Int) -> ValueType {
        if rhs == 0 {
            return .zero
        }
        if rhs == 1 {
            return lhs
        }
        var v: ValueType = 1
        var vv: ValueType = lhs
        var p = Int(rhs)
        while p != 1 {
            if p&1 == 1 {
                v*=lhs
                p -= 1
            } else {
                vv*=vv
                p >>= 1
            }
        }
        return v*vv
    }
    static func pow<RHS: BinaryInteger>(_ lhs: Self, _ rhs: RHS) -> Self {
        return .init(value: powImpl(lhs: lhs.value, rhs: Int(rhs)), predecessors: lhs, back: { g, pred in
            pred[0].header.add(ValueType(exactly: rhs)!*powImpl(lhs: pred[0].value, rhs: Int(rhs)-1)*g)
        })
    }
}

extension Differentiable: SignedNumeric where ValueType: SignedNumeric&Comparable {
}

extension Differentiable: Strideable where ValueType: Strideable&SignedNumeric {
    public func distance(to other: Self) -> Differentiable<ValueType> {
        .init(value: other.value-value, predecessors: (self, other), back: { g, pred in
            pred[0].header.add(-g)
            pred[1].header.add(g)
        })
    }
    public func advanced(by n: Differentiable<ValueType>) -> Self {
        .init(value: value+n.value, predecessors: (self, n), back: { g, pred in
            pred[0].header.add(g)
            pred[1].header.add(g)
        })
    }
}

extension Differentiable: FloatingPoint where ValueType: FloatingPoint {
    public init<Source>(_ value: Source) where Source: BinaryInteger {
        self.init(value: ValueType(value))
    }
    public init(signOf: Self, magnitudeOf: Self) {
        let v = ValueType(signOf.value.sign == magnitudeOf.value.sign ? 1 : -1)
        self.init(value: .init(signOf: signOf.value, magnitudeOf: magnitudeOf.value),
                  predecessors: magnitudeOf, back: { g, pred in
            pred[0].header.add(v*g)
        })
    }
    public init(sign: FloatingPointSign, exponent: ValueType.Exponent, significand: Self) {
        let v = ValueType(sign: sign, exponent: exponent, significand: 1)
        self.init(value: .init(sign: sign, exponent: exponent, significand: significand.value), predecessors: significand, back: {g, pred in
            pred[0].header.add(v*g)
        })
    }
    mutating public func round(_ rule: FloatingPointRoundingRule) {
        value.round(rule)
    }
    public var exponent: ValueType.Exponent {
        value.exponent
    }
    public static var radix: Int {
        ValueType.radix
    }
    public static var nan: Self {
        .init(value: ValueType.nan)
    }
    public static var signalingNaN: Self {
        .init(value: ValueType.signalingNaN)
    }
    public static var infinity: Self {
        .init(value: ValueType.infinity)
    }
    public static var greatestFiniteMagnitude: Self {
        .init(value: ValueType.greatestFiniteMagnitude)
    }
    public static var pi: Self {
        .init(value: ValueType.pi)
    }
    public var ulp: Self {
        .init(value: value.ulp)
    }
    public static var leastNormalMagnitude: Self {
        .init(value: ValueType.leastNormalMagnitude)
    }
    public static var leastNonzeroMagnitude: Self {
        .init(value: ValueType.leastNonzeroMagnitude)
    }
    public var sign: FloatingPointSign {
        value.sign
    }
    public var significand: Self {
        .init(value: value.significand, predecessors: self, back: {p, pred in
            pred[0].header.add((value<0 ? -1:1)/ValueType(1<<value.exponent)*p)
        })
    }
    public static func / (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value/rhs.value, predecessors: (lhs, rhs), back: { g, pred in
            pred[0].header.add(g/pred[1].value)
            pred[1].header.add(-pred[0].value/(pred[1].value*pred[1].value)*g)
        })
    }
    public static func /= (lhs: inout Self, rhs: Self) {
        lhs = lhs / rhs
    }
    mutating public func formRemainder(dividingBy other: Self) {
        self = remainder(dividingBy: other)
    }
    public func remainder(dividingBy other: Self) -> Self {
        .init(value: value.remainder(dividingBy: other.value), predecessors: (self, other), back: { g, pred in
            pred[0].header.add(g)
            pred[1].header.add(-(pred[0].value/pred[1].value).rounded(.toNearestOrEven)*g)
        })
    }
    mutating public func formTruncatingRemainder(dividingBy other: Self) {
        self = truncatingRemainder(dividingBy: other)
    }
    public func truncatingRemainder(dividingBy other: Differentiable<ValueType>) -> Differentiable<ValueType> {
        .init(value: value.truncatingRemainder(dividingBy: other.value), predecessors: (self, other), back: { g, pred in
            pred[0].header.add(g)
            pred[1].header.add(-(pred[0].value/pred[1].value).rounded(.towardZero)*g)
        })
    }
    mutating public func formSquareRoot() {
        self = self.squareRoot()
    }
    public func squareRoot() -> Self {
        .init(value: value.squareRoot(), predecessors: self, back: { g, pred in
            pred[0].header.add(1/(2*pred[0].value.squareRoot())*g)
        })
    }
    mutating public func addProduct(_ lhs: Self, _ rhs: Self) {
        self = self.addingProduct(lhs, rhs)
    }
    public func addingProduct(_ lhs: Self, _ rhs: Self) -> Self {
        .init(value: value.addingProduct(lhs.value, rhs.value), predecessors: (self, lhs, rhs), back: { g, pred in
            pred[0].header.add(g)
            pred[1].header.add(pred[2].value*g)
            pred[2].header.add(pred[1].value*g)
        })
    }
    public var nextUp: Self {
        .init(value: value.nextUp, predecessors: self, back: { _, pred in
            pred[0].header.add(1)
        })
    }
    public var nextDown: Self {
        .init(value: value.nextDown, predecessors: self, back: { _, pred in
            pred[0].header.add(1)
        })
    }
    public func isEqual(to other: Self) -> Bool {
        value.isEqual(to: other.value)
    }
    public func isLess(than other: Self) -> Bool {
        value.isLess(than: other.value)
    }
    public func isLessThanOrEqualTo(_ other: Self) -> Bool {
        value.isLessThanOrEqualTo(other.value)
    }
    public func isTotallyOrdered(belowOrEqualTo other: Self) -> Bool {
        value.isTotallyOrdered(belowOrEqualTo: other.value)
    }
    public var isNormal: Bool {
        value.isNormal
    }
    public var isFinite: Bool {
        value.isFinite
    }
    public var isZero: Bool {
        value.isZero
    }
    public var isSubnormal: Bool {
        value.isSubnormal
    }
    public var isInfinite: Bool {
        value.isInfinite
    }
    public var isNaN: Bool {
        value.isNaN
    }
    public var isSignalingNaN: Bool {
        value.isSignalingNaN
    }
    public var isCanonical: Bool {
        value.isCanonical
    }
}
extension Differentiable: BinaryFloatingPoint where ValueType: BinaryFloatingPoint, ValueType.Stride==ValueType {
    public init(sign: FloatingPointSign, exponentBitPattern: ValueType.RawExponent, significandBitPattern: ValueType.RawSignificand) {
        self.init(constant: ValueType(sign: sign, exponentBitPattern: exponentBitPattern, significandBitPattern: significandBitPattern))
    }
    public init<Source>(_ value: Source) where Source: BinaryFloatingPoint {
        self.init(constant: ValueType(value))
    }
    public init<Source>(_ value: Source) where Source: BinaryInteger {
        self.init(constant: ValueType(value))
    }
    public init<Source>(value: Differentiable<Source>) where Source: BinaryFloatingPoint {
        self.init(value: ValueType(value.value), header: CastHeader<ValueType, Source>(delegate: value.header as! Header<Source>))
    }
//    @_disfavoredOverload
//    public init<Source>(value: Source) where Source: BinaryInteger {
//        self.init(value: ValueType(value))
//    }
//    public init<Source>(value: Source) where Source: BinaryFloatingPoint {
//        self.init(value: ValueType(value))
//    }
    @_disfavoredOverload
    public init<Source>(value: Differentiable<Source>) where Source: BinaryInteger {
        self.init(value: ValueType(value.value), header: CastIntHeader<ValueType, Source>(delegate: value.header as! Header<Source>))
    }
    public static var exponentBitCount: Int {
        ValueType.exponentBitCount
    }

    public static var significandBitCount: Int {
        ValueType.significandBitCount
    }

    public var exponentBitPattern: ValueType.RawExponent {
        value.exponentBitPattern
    }

    public var significandBitPattern: ValueType.RawSignificand {
        value.significandBitPattern
    }

    public var binade: Differentiable<ValueType> {
        .init(constant: value.binade)
    }

    public var significandWidth: Int {
        value.significandWidth
    }

    public typealias RawSignificand = ValueType.RawSignificand

    public typealias RawExponent = ValueType.RawExponent

}

extension Differentiable: ElementaryFunctions where ValueType: ElementaryFunctions&SignedNumeric&ExpressibleByFloatLiteral {
    public static func exp(_ x: Self) -> Self {
        let v = ValueType.exp(x.value)
        return .init(value: v, predecessors: x, back: { g, pred in
            pred[0].header.add(v*g)
        })
    }
    public static func expMinusOne(_ x: Self) -> Self {
        let v = ValueType.expMinusOne(x.value)
        return .init(value: v, predecessors: x, back: { g, pred in
            pred[0].header.add((v+1)*g)
        })
    }
    public static func cosh(_ x: Self) -> Self {
        .init(value: ValueType.cosh(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.sinh(pred[0].value)*g)
        })
    }
    public static func sinh(_ x: Self) -> Self {
        .init(value: ValueType.sinh(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.cosh(pred[0].value)*g)
        })
    }
    public static func tanh(_ x: Self) -> Self {
        .init(value: ValueType.tanh(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(ValueType.cosh(pred[0].value), -2)*g)
        })
    }
    public static func cos(_ x: Self) -> Self {
        .init(value: ValueType.cos(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(-ValueType.sin(pred[0].value)*g)
        })
    }
    public static func sin(_ x: Self) -> Self {
        .init(value: ValueType.sin(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.cos(pred[0].value)*g)
        })
    }
    public static func tan(_ x: Self) -> Self {
        .init(value: ValueType.tan(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(ValueType.cos(pred[0].value), -2)*g)
        })
    }
    public static func log(_ x: Self) -> Self {
        .init(value: ValueType.log(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(pred[0].value, -1)*g)
        })
    }
    public static func log(onePlus x: Self) -> Self {
        .init(value: ValueType.log(onePlus: x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(pred[0].value+1, -1)*g)
        })
    }
    public static func acosh(_ x: Self) -> Self {
        .init(value: ValueType.acosh(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow((pred[0].value-1) * (pred[0].value+1), -0.5)*g)
        })
    }
    public static func asinh(_ x: Self) -> Self {
        .init(value: ValueType.asinh(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(pred[0].value * pred[0].value+1, -0.5)*g)
        })
    }
    public static func atanh(_ x: Self) -> Self {
        .init(value: ValueType.atanh(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(1-pred[0].value * pred[0].value, -1)*g)
        })
    }
    public static func acos(_ x: Self) -> Self {
        .init(value: ValueType.acos(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(-ValueType.pow(1-pred[0].value * pred[0].value, -0.5)*g)
        })
    }
    public static func asin(_ x: Self) -> Self {
        .init(value: ValueType.asin(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(1-pred[0].value * pred[0].value, -0.5)*g)
        })
    }
    public static func atan(_ x: Self) -> Self {
        .init(value: ValueType.atan(x.value), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(1+pred[0].value * pred[0].value, -1)*g)
        })
    }
    public static func pow(_ x: Self, _ y: Self) -> Self {
        let vv = ValueType.pow(x.value, y.value)
        return .init(value: vv, predecessors: (x, y), back: { g, pred in
            pred[0].header.add(y.value*ValueType.pow(pred[0].value, pred[1].value-1)*g)
            pred[1].header.add(vv*ValueType.log(pred[0].value)*g)
        })
    }
    public static func pow(_ x: Self, _ n: Int) -> Self {
        .init(value: ValueType.pow(x.value, n), predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType(exactly: n)!*ValueType.pow(pred[0].value, n-1)*g)
        })
    }
    public static func sqrt(_ x: Self) -> Self {
        let v = ValueType.sqrt(x.value)
        return .init(value: v, predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.pow(v*2, -1)*g)
        })
    }
    public static func root(_ x: Self, _ n: Int) -> Self {
        let v = ValueType.root(x.value, n)
        return .init(value: v, predecessors: x, back: { g, pred in
            pred[0].header.add(ValueType.root(pred[0].value, n-1)*ValueType.pow(ValueType(exactly: n)!, -1)*g)
        })
    }
}

extension Differentiable: RealFunctions where ValueType: RealFunctions&FloatingPoint&ExpressibleByFloatLiteral {
    public static func atan2(y: Self, x: Self) -> Self {
        .init(value: ValueType.atan2(y: y.value, x: x.value), predecessors: (y, x), back: {g, pred in
            let d = pred[0].value*pred[0].value+pred[1].value*pred[1].value
            pred[1].header.add(-pred[0].value*ValueType.pow(d, -1)*g)
            pred[0].header.add(pred[1].value*ValueType.pow(d, -1)*g)
        })
    }

    public static func erf(_ x: Self) -> Self {
        .init(value: ValueType.erf(x.value), predecessors: x, back: {g, pred in
            pred[0].header.add(ValueType.exp(-pred[0].value*pred[0].value)*(2/ValueType.pi.squareRoot())*g)
        })
    }
    public static func erfc(_ x: Self) -> Self {
        .init(value: ValueType.erfc(x.value), predecessors: x, back: {g, pred in
            pred[0].header.add(-ValueType.exp(-pred[0].value*pred[0].value)*(2/ValueType.pi.squareRoot())*g)
        })
    }
    public static func exp2(_ x: Self) -> Self {
        let v = ValueType.exp2(x.value)
        return .init(value: v, predecessors: x, back: {g, pred in
            pred[0].header.add(v*ValueType.log(2)*g)
        })
    }
    public static func exp10(_ x: Self) -> Self {
        let v = ValueType.exp10(x.value)
        return .init(value: v, predecessors: x, back: {g, pred in
            pred[0].header.add(v*ValueType.log(10)*g)
        })
    }
    public static func hypot(_ x: Self, _ y: Self) -> Self {
        .init(value: ValueType.hypot(x.value, y.value), predecessors: (y, x), back: {g, pred in
            let d = ValueType.pow(pred[0].value*pred[0].value+pred[1].value*pred[1].value, -0.5)
            pred[0].header.add(pred[0].value*d*g)
            pred[1].header.add(pred[1].value*d*g)
        })
    }
    public static func gamma(_ x: Self) -> Self {
        let v = ValueType.exp10(x.value)
        return .init(value: v, predecessors: x, back: {_, _ in
            fatalError("undefined")
        })
    }
    public static func log2(_ x: Self) -> Self {
        .init(value: ValueType.log2(x.value), predecessors: x, back: {g, pred in
            pred[0].header.add(g/(pred[0].value*ValueType.log(2)))
        })
    }
    public static func log10(_ x: Self) -> Self {
        .init(value: ValueType.log10(x.value), predecessors: x, back: {g, pred in
            pred[0].header.add(g/(pred[0].value*ValueType.log(10)))
        })
    }
    public static func logGamma(_ x: Self) -> Self {
        .init(value: ValueType.logGamma(x.value), predecessors: x, back: {_, _ in
            fatalError("undefined")
        })
    }
    public static func signGamma(_ x: Self) -> FloatingPointSign {
        ValueType.signGamma(x.value)
    }
    public static func _mulAdd(_ a: Self, _ b: Self, _ c: Self) -> Differentiable<ValueType> {
        .init(value: ValueType._mulAdd(a.value, b.value, c.value), predecessors: (a, b, c), back: {g, pred in
            pred[0].header.add(pred[1].value*g)
            pred[1].header.add(pred[0].value*g)
            pred[2].header.add(g)
        })
    }
}

extension Differentiable /*: AlgebraicField*/ where ValueType: AlgebraicField&Strideable, ValueType.Magnitude==ValueType {
//    public static func /= (a: inout Self, b: Self) {
//        a = a / b
//    }
    public var reciprocal: Self? {
        if let r = value.reciprocal {
            return .init(value: r, predecessors: self, back: { g, pred in
                pred[0].header.add(-r*r*g)
            })
        } else {
            return nil
        }
    }
//    public static func / (a: Self, b: Self) -> Self {
//        let q = a.value / b.value
//        return .init(value: q, predecessors: (a, b), back: {g,pred in
//            let v = g.value
//            a._grad.update(v, 1/b.value)
//            b._grad.update(-v, a.value/(b.value*b.value))
//        })
//    }
}

// extension Differentiable: Real where ValueType: Real&ExpressibleByFloatLiteral, ValueType.Stride == ValueType {}

extension Differentiable: DifferentiableValue where ValueType: SignedNumeric {
    static public var one: Differentiable<ValueType> {
        .init(value: 1)
    }
}
