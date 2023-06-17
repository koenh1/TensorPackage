//
//  Tensor+Differentiable.swift
//  Tensor
//
//  Created by Koen Hendrikx on 24/05/2023.
//

import Foundation
import RealModule

extension Tensor: DifferentiableValue where T: AdditiveArithmetic&ExpressibleByIntegerLiteral {
    public static var one: Tensor<Types, IndexType, T> {
        .init(shape: [UInt32](repeatElement(1, count: Types.count)), initialValue: 1)
    }
}
public struct TensorGradients<Parameters, ValueType: DifferentiableValue>: CustomStringConvertible {
    public var description: String {
        "\(result)"
    }
    var result: [String: Any]
    var count: Int = 0
    static func visitParameters(parameters: Any, key: String, _ f: (String, any TensorProtocol) -> Void) {
        let m = Mirror(reflecting: parameters)
        if let mm = parameters as? any TensorProtocol {
            f("", mm)
        }
        for mc in m.children.enumerated() {
            let k = "\(key)\(mc.element.label ?? "_\(mc.offset)")"
            if let mm = mc.element.value as? any TensorProtocol {
                f(k, mm)
            } else {
                visitParameters(parameters: mc.element.value, key: "\(k).", f)
            }
        }
    }

    public init<X: DifferentiableProtocol>(of f: X, forTensors parameters: Parameters) where X.ValueType == ValueType {
        let header = (f as! Differentiable<ValueType>).header
        var count = 0
        func gather<S: TensorProtocol>(_ key: String, _ s: S) {
            for i in s.elements {
                if let v = i as? any DifferentiableProtocol {
                    if !v.isConstant {
                        v.clearGradient()
                        count += 1
                    }
                }
            }
        }
        Self.visitParameters(parameters: parameters, key: "") { n, s in
            gather(n, s)
        }
        self.count = count
        var list: [any HeaderProtocol] = []
        globalGeneration += 1
        func listSorted(_ header: any HeaderProtocol) {
            if header.generation != globalGeneration {
                if header.isConstant {
                    return
                }
                header.applyLeft(visitor: listSorted)
                header.applyRight(visitor: listSorted)
                list.append(header)
                header.clearGradient()
            }
        }
        listSorted(header)
        header.unitGradient()
        for x in list.reversed() {
            x.updateGrad()
        }
        var tensors: [String: Any] = [:]
        func gatherGrads<S: TensorProtocol>(_ key: String, _ s: S) {
            var isConstant = true
            let t: Tensor<S.Types, S.IndexType, ValueType> = s.mapTensor { i -> ValueType in
                if let v = i as? any DifferentiableProtocol {
                    if !v.isConstant {
                        isConstant = false
                    }
                    return v.grad as! ValueType
                }
                return .zero
            }
            if !isConstant {
                tensors[key] = t
            }
        }
        Self.visitParameters(parameters: parameters, key: "") {n, s in
            gatherGrads(n, s)
        }
        result = tensors
    }
    public init(forTensors parameters: Parameters) {
        var tensors: [String: Any] = [:]
        func gather<S: TensorProtocol>(_ key: String, _ s: S) {
            tensors[key]=Tensor<S.Types, S.IndexType, ValueType>(shape: s.shape.map {$0}, initialValue: .zero)
        }
        Self.visitParameters(parameters: parameters, key: "") { n, s in
            gather(n, s)
        }
        result = tensors
    }
    public subscript<R: TensorProtocol>(key: String) -> R? {
        result[key] as? R
    }

    public func update(stepSize: ValueType, parameters:inout Parameters) -> ValueType where ValueType: Numeric {
        update(parameters: &parameters, op: {(p:inout Differentiable<ValueType>, g: ValueType) in
            let t = stepSize*g
            p.value += t
            return t*g
        })
    }
    public func update<X>(stepSize: ValueType, views:inout X) -> ValueType where ValueType: Numeric {
        update(parameters: &views, op: {(p:inout Undifferentiable<ValueType>, g: ValueType) in
            let t = stepSize*g
            p.value += t
            return t*g
        })
    }
    public func update<X, Y>(parameters:inout Y, op:(inout X, ValueType) -> ValueType) -> ValueType {
        func gatherGrads<S: TensorProtocol>(result:inout ValueType, _ key: String, _ param: S) {
            if let t = self.result[key] as? Tensor<S.Types, S.IndexType, ValueType> {
                var param = param as! Tensor<S.Types, S.IndexType, X>
                var it=t.shape.makeIterator()
                var it2=param.shape.makeIterator()
                param.buffer.applyShared { sptr in
                    t.buffer.apply { tptr in
                        while let i=it.next(), let j=it2.next() {
                            result += op(&sptr[Int(j)], tptr[Int(i)])
                        }
                    }
                }
            }
        }
        var result: ValueType = .zero
        Self.visitParameters(parameters: parameters, key: "") {n, s in
            gatherGrads(result: &result, n, s)
        }
        return result
    }
    public mutating func updateGradients(factor: ValueType, factor2: ValueType, other: Self) where ValueType: FloatingPoint {
        func update<S: TensorProtocol>(result:inout S, other: Any) {
            var a = result as! Tensor<S.Types, S.IndexType, ValueType>
            let b = other as! Tensor<S.Types, S.IndexType, ValueType>
            if !a.assignCombine(b, {$0 = factor*$0 + factor2*$1}) {
                fatalError()
            }
            result = a as! S
        }
        for i in result {
            if let x = other.result[i.key] {
                if let v = x as? any TensorProtocol {
                    if var w = i.value as? any TensorProtocol {
                        update(result: &w, other: v)
                        result[i.key] = w
                    }
                }
            } else {
                fatalError()
            }
        }
    }
}
// struct GradientDescent<TensorType: TensorProtocol> where TensorType.T: Real&DifferentiableValue {
//    typealias DTensorType = Tensor<TensorType.Types, TensorType.IndexType, Differentiable<TensorType.T>>
//    var function: (TensorType) -> TensorType.T
//    var diffFunction: (DTensorType) -> Differentiable<TensorType.T>
//    init(function:@escaping (TensorType) -> TensorType.T, diffFunction:@escaping (DTensorType) -> Differentiable<TensorType.T>) {
//        self.function = function
//        self.diffFunction = diffFunction
//    }
//
//    func step(steps: Int, stepSize: TensorType.T, parameters:inout TensorType) -> TensorType.T {
//        var first: TensorType.T?
//        var last: TensorType.T?
//        for _ in 0..<steps {
//            let x = parameters.constant
//            let f = diffFunction(x)
//            let f1 = f.value
//            if first == nil {
//                first = f1
//            }
//            let g = f.gradients(for: x.elements)
//            print("gradient =\(g.normSquared())")
//            g.update(stepSize: stepSize, parameters: &parameters.elements)
//            let f2 = function(parameters)
//            if f2 > f1 {
//                g.update(stepSize: -stepSize, parameters: &parameters.elements)
//                break
//            }
//            last = f2
//            print("new value =\(f2)")
//        }
//        if let first = first, let last=last {
//            return first - last
//        } else {
//            return 0
//        }
//    }
// }

extension TensorProtocol where T: DifferentiableValue&SignedNumeric {
    public var differentiable: Tensor<Types, IndexType, Differentiable<T>> {
        mapTensor(Differentiable.init(value:))
    }
    public var constant: Tensor<Types, IndexType, Differentiable<T>> {
        mapTensor(Differentiable.init(constant:))
    }
}
extension Tensor where T: DifferentiableProtocol {
    public func gradients(_ x: T) -> Tensor<Types, IndexType, T.ValueType> {
        mapTensor {$0.gradient[x]}
    }
    public func gradients(of x: T) -> Tensor<Types, IndexType, T.ValueType> {
        gradients(of: x, op: {$0})
    }
    public func gradients(of x: T, multipliedBy factor: T.ValueType) -> Tensor<Types, IndexType, T.ValueType> where T.ValueType: FloatingPoint {
        gradients(of: x, op: {$0*factor})
    }
    public func gradients(of x: T, op: (T.ValueType)->T.ValueType) -> Tensor<Types, IndexType, T.ValueType> {
        let a = x.gradients(for: self.elements).result
        var buffer: Buffer<T.ValueType> = .init(capacity: a.count)
        buffer.apply { ptr in
            var it = shape.makeIterator()
            for i in a {
                if let j = it.next() {
                    ptr.advanced(by: Int(j)).initialize(to: op(i))
                }
            }
        }
        return .init(buffer: buffer, shape: shape)
    }
    public var values: Tensor<Types, IndexType, T.ValueType> {
        get {
            mapTensor(\.value)
        }
        set {
            if shape.shape != newValue.shape.shape {
                fatalError()
            }
            var it = shape.makeIterator()
            var it1 = newValue.shape.makeIterator()
            buffer.apply { optr in
                newValue.buffer.apply { iptr in
                    while let i = it.next(), let j = it1.next() {
                        optr[Int(i)].value = iptr[Int(j)]
                    }
                }
            }
        }
    }
}

infix operator ⨂ : MultiplicationPrecedence

private class Header2<LHS: TensorProtocol&DifferentiableValue, RHS: TensorProtocol&DifferentiableValue, RESULT: TensorProtocol&DifferentiableValue>: AbstractHeader<RESULT> where LHS.IndexType.Element==RHS.IndexType.Element, LHS.IndexType.Element==RESULT.IndexType.Element, LHS.T==RHS.T, RESULT.T==LHS.T, LHS.T: Numeric&DifferentiableValue, RESULT.IndexType == LHS.IndexType, LHS.IndexType==RHS.IndexType {
    var data1:(header: AbstractHeader<LHS>, value: LHS)
    var data2:(header: AbstractHeader<RHS>, value: RHS)
    init(_ p1: (AbstractHeader<LHS>, LHS), _ p2: (AbstractHeader<RHS>, RHS)) {
        self.data1 = p1
        self.data2 = p2
        super.init(.zero)
    }
    override func applyLeft(visitor: (any HeaderProtocol) -> Void) {
        visitor(data1.header)
    }
    override func applyRight(visitor: (any HeaderProtocol) -> Void) {
        visitor(data2.header)
    }
    override func updateGrad() {
        if let r: LHS = data2.value.multiply(rhs: grad, zero: .zero, multiplyOperator: *, sumOperator: +) {
            data1.header.grad = r+data1.header.grad
        }
        if let r2: RHS = data1.value.multiply(rhs: grad, zero: .zero, multiplyOperator: *, sumOperator: +) {
            data2.header.grad = r2+data2.header.grad
        }
    }
}

extension Differentiable where
    ValueType: TensorProtocol, ValueType.T: SignedNumeric&Comparable&DifferentiableValue {
    public static func ⨂<S: TensorProtocol&DifferentiableValue, R: TensorProtocol&DifferentiableValue> (lhs: Self, rhs: Differentiable<S>) -> Differentiable<R> where  ValueType.T == S.T, ValueType.IndexType.Element == R.IndexType.Element, ValueType.T == R.T, R.T: SignedNumeric&DifferentiableValue, ValueType.IndexType==R.IndexType, S.IndexType==ValueType.IndexType {
        if let r: R = lhs.value.multiply(rhs: rhs.value, zero: .zero, multiplyOperator: *, sumOperator: +) {
            return .init(value: r, header: Header2<ValueType, S, R>((lhs.header, lhs.value), (rhs.header, rhs.value)))
        } else {
            fatalError()
        }
    }
}

private class NormHeader<InputType: TensorProtocol&DifferentiableValue>: AbstractHeader<Scalar<InputType.T>> where InputType.T: FloatingPoint {
    var data:(header: AbstractHeader<InputType>, value: InputType)
    init(delegate: AbstractHeader<InputType>, value: InputType, norm: InputType.T) {
        data=(header:delegate, value:value)
        self.norm = norm
        super.init(.zero)
    }
    let norm: InputType.T
    override func applyLeft(visitor: (any HeaderProtocol) -> Void) {
        visitor(data.header)
    }
    override func updateGrad() {
        let r: InputType = data.value.mapTensor {grad.value*$0*norm}
        data.header.grad = r + data.header.grad
    }
}

private class AbsHeader<InputType: TensorProtocol&DifferentiableValue>: AbstractHeader<InputType> where InputType.T: FloatingPoint {
    var data:(header: AbstractHeader<InputType>, value: InputType)
    init(delegate: AbstractHeader<InputType>, value: InputType) {
        data=(header:delegate, value:value)
        super.init(.zero)
    }
    override func applyLeft(visitor: (any HeaderProtocol) -> Void) {
        visitor(data.header)
    }
    override func updateGrad() {
        let r: InputType = data.value.mapTensor {($0.sign == .plus ? 1 : -1)}
        let v = r.combine(grad, *)!
        data.header.grad = v + data.header.grad
    }
}
extension Differentiable where ValueType: TensorProtocol, ValueType.T: FloatingPoint {

    public var normSquared: Differentiable<Scalar<ValueType.T>> {
        let n = value.normSquared
        return .init(value: n, header: NormHeader(delegate: header, value: value, norm: 2))
    }
    public var norm: Differentiable<Scalar<ValueType.T>> {
        let n = value.norm
        return .init(value: n, header: NormHeader(delegate: header, value: value, norm: 1/n.value))
    }

    public var magnitude: Differentiable<ValueType> {
        let n: ValueType = value.magnitude as! ValueType
        return .init(value: n, header: AbsHeader(delegate: header, value: value))
    }

}
