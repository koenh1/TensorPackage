//
//  Undifferentiable.swift
//  Tensor
//
//  Created by Koen Hendrikx on 24/05/2023.
//

import Foundation

public struct Undifferentiable<ValueType: DifferentiableValue>: DifferentiableValue {
    public static var one: Undifferentiable<ValueType> {
        .init(value: .one)
    }

    var value: ValueType
    init(value: ValueType) {
        self.value = value
    }
    var dummy1: Int64 = 0
}

extension Undifferentiable: CustomStringConvertible where ValueType: CustomStringConvertible {
    public var description: String {
        value.description
    }
}
extension Undifferentiable: Equatable where ValueType: Equatable {
    public static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.value == rhs.value
    }
}
extension Undifferentiable: Hashable where ValueType: Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(value)
    }
}
extension Undifferentiable: Comparable where ValueType: Comparable {
    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.value<rhs.value
    }
}
extension Undifferentiable: ExpressibleByIntegerLiteral where ValueType: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: ValueType.IntegerLiteralType) {
        self.init(value: ValueType(integerLiteral: value))
    }
}
extension Undifferentiable: ExpressibleByFloatLiteral where ValueType: ExpressibleByFloatLiteral {
    public init(floatLiteral value: ValueType.FloatLiteralType) {
        self.init(value: ValueType(floatLiteral: value))
    }
}
extension Undifferentiable: AdditiveArithmetic where ValueType: AdditiveArithmetic {
    public static func - (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value-rhs.value)
    }
    public static func + (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value+rhs.value)
    }
    public static var zero: Self {
        .init(value: .zero)
    }
}
extension Undifferentiable: BinaryInteger where ValueType: BinaryInteger, ValueType.Magnitude==ValueType, ValueType.Stride==ValueType {

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
        .init(value: lhs.value << rhs)
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
    public static func /= (lhs: inout Self, rhs: Self) {
        lhs = lhs / rhs
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
    public static func / (lhs: Self, rhs: Self) -> Self {
        let q = lhs.value / rhs.value
        return .init(value: q)
    }
    public static func % (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value % rhs.value)
    }
    public static func %= (lhs: inout Self, rhs: Self) {
        lhs = lhs % rhs
    }
    public static func * (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value*rhs.value)
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
extension Undifferentiable: Numeric where ValueType: SignedNumeric&Comparable, ValueType.Magnitude==ValueType {

    public init?<T>(exactly source: T) where T: BinaryInteger {
        if let v = ValueType(exactly: source) {
            self.init(value: v)
        } else {
            return nil
        }
    }
    public var magnitude: Undifferentiable<ValueType.Magnitude> {
        .init(value: value.magnitude)
    }
    public static func *= (lhs: inout Self, rhs: Self) {
         lhs = lhs + rhs
    }
    public static func * (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value*rhs.value)
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
        .init(value: powImpl(lhs: lhs.value, rhs: Int(rhs)))
    }
}

extension Undifferentiable: SignedNumeric where ValueType: SignedNumeric, ValueType.Magnitude==ValueType {
}

extension Undifferentiable: Strideable where ValueType: Strideable&SignedNumeric, ValueType.Magnitude==ValueType, ValueType.Stride==ValueType {
    public func distance(to other: Self) -> Undifferentiable<ValueType> {
        .init(value: value.distance(to: other.value))
    }
    public func advanced(by n: Undifferentiable<ValueType>) -> Self {
        .init(value: value.advanced(by: n.value))
    }
}

extension TensorProtocol where T: DifferentiableProtocol, T.ValueType: Numeric {
    public var view: Tensor<Types, IndexType, Undifferentiable<T.ValueType>> {
        .init(buffer: buffer.view(as: Undifferentiable.self), shape: shape)
    }
}

extension Undifferentiable: FloatingPoint where ValueType: FloatingPoint, ValueType.Stride==ValueType {
    public typealias Exponent = ValueType.Exponent

    public init<Source>(_ value: Source) where Source: BinaryInteger {
        self.init(value: ValueType(value))
    }
    public init(signOf: Self, magnitudeOf: Self) {
        self.init(value: .init(signOf: signOf.value, magnitudeOf: magnitudeOf.value))
    }
    public init(sign: FloatingPointSign, exponent: ValueType.Exponent, significand: Self) {
        self.init(value: .init(sign: sign, exponent: exponent, significand: significand.value))
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
        .init(value: value.significand)
    }
    public static func / (lhs: Self, rhs: Self) -> Self {
        .init(value: lhs.value/rhs.value)
    }
    public static func /= (lhs: inout Self, rhs: Self) {
        lhs = lhs / rhs
    }
    mutating public func formRemainder(dividingBy other: Self) {
        self = remainder(dividingBy: other)
    }
    public func remainder(dividingBy other: Self) -> Self {
        .init(value: value.remainder(dividingBy: other.value))
    }
    mutating public func formTruncatingRemainder(dividingBy other: Self) {
        self = truncatingRemainder(dividingBy: other)
    }
    public func truncatingRemainder(dividingBy other: Self) -> Self {
        .init(value: value.truncatingRemainder(dividingBy: other.value))
    }
    mutating public func formSquareRoot() {
        self = self.squareRoot()
    }
    public func squareRoot() -> Self {
        .init(value: value.squareRoot())
    }
    mutating public func addProduct(_ lhs: Self, _ rhs: Self) {
        self = self.addingProduct(lhs, rhs)
    }
    public func addingProduct(_ lhs: Self, _ rhs: Self) -> Self {
        .init(value: value.addingProduct(lhs.value, rhs.value))
    }
    public var nextUp: Self {
        .init(value: value.nextUp)
    }
    public var nextDown: Self {
        .init(value: value.nextDown)
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
