//
//  NNTests.swift
//  TensorTests
//
//  Created by Koen Hendrikx on 21/05/2023.
//

import XCTest
import RealModule
import TensorPackage
enum DataDimension {}
enum LayerDimension {}

struct Layer<InputDimension, OutputDimension, T: BinaryFloatingPoint&DifferentiableValue&ElementaryFunctions> {
    var weight: Matrix<InputDimension, OutputDimension, T>
    var bias: Vector<OutputDimension, T>
    init<R: RandomNumberGenerator>(nin: UInt32, nout: UInt32, rng:inout R) where T.RawSignificand: FixedWidthInteger {
        weight = Matrix<InputDimension, OutputDimension, T>(shape: [nin, nout], uniformValuesIn: -1..<1, using: &rng)
        bias = Vector<OutputDimension, T>(shape: [nout], uniformValuesIn: -1..<1, using: &rng)
    }
    func call<D>(batch: Matrix<D, InputDimension, T>) -> Matrix<D, OutputDimension, T> {
        .tanh(weight*batch+bias)
    }
    func call(input: Vector<InputDimension, T>) -> Vector<OutputDimension, T> {
        .tanh(weight*input+bias)
    }
    func calld<D>(batch: Matrix<D, InputDimension, Differentiable<T>>) -> Matrix<D, OutputDimension, Differentiable<T>> {
        .tanh(weight.differentiable*batch+bias.differentiable)
    }
}
protocol LayersProtocol {
    associatedtype T: BinaryFloatingPoint&DifferentiableValue&ElementaryFunctions where T.RawSignificand: FixedWidthInteger
    associatedtype InputDimension
    associatedtype OutputDimension
    associatedtype Tail: LayersProtocol where T == Tail.T, Tail.OutputDimension == OutputDimension
    init<R: RandomNumberGenerator>(nin: UInt32, nouts: [UInt32], rng:inout R)
    func call(input: Vector<InputDimension, T>) -> Vector<OutputDimension, T>
    func call<D>(batch: Matrix<D, InputDimension, T>) -> Matrix<D, OutputDimension, T>
    func calld<D>(batch: Matrix<D, InputDimension, Differentiable<T>>) -> Matrix<D, OutputDimension, Differentiable<T>>
}
struct EndLayer<OutputDimension, T: BinaryFloatingPoint&DifferentiableValue&ElementaryFunctions>: LayersProtocol where T.RawSignificand: FixedWidthInteger {
    func call<D>(batch: Matrix<D, OutputDimension, T>) -> Matrix<D, OutputDimension, T> {
        batch
    }

    func call(input: Vector<OutputDimension, T>) -> Vector<OutputDimension, T> {
        input
    }
    func calld<D>(batch: Matrix<D, InputDimension, Differentiable<T>>) -> Matrix<D, OutputDimension, Differentiable<T>> {
        batch
    }
    typealias Tail = Self
    init<R>(nin: UInt32, nouts: [UInt32], rng: inout R) where R: RandomNumberGenerator {
        if !nouts.isEmpty {
            fatalError()
        }
    }
}
struct Layers<InputDimension, Tail: LayersProtocol>: LayersProtocol {
    typealias T = Tail.T
    typealias OutputDimension = Tail.OutputDimension
    var layer: Layer<InputDimension, Tail.InputDimension, T>
    var tail: Tail
    init<R: RandomNumberGenerator>(nin: UInt32, nouts: [UInt32], rng:inout R) {
        var nouts = nouts
        if let f = nouts.first {
            nouts.removeFirst()
            layer = .init(nin: nin, nout: f, rng: &rng)
            tail = .init(nin: f, nouts: nouts, rng: &rng)
        } else {
            fatalError()
        }
    }
    func call(input: Vector<InputDimension, T>) -> Vector<OutputDimension, T> {
        let r: Vector<Tail.InputDimension, T> = layer.call(input: input)
        let result: Vector<OutputDimension, T> = tail.call(input: r)
        return result
    }
    func call<D>(batch: Matrix<D, InputDimension, Tail.T>) -> Matrix<D, Tail.OutputDimension, Tail.T> {
        let r: Matrix<D, Tail.InputDimension, T> = layer.call(batch: batch)
        let result: Matrix<D, OutputDimension, T> = tail.call(batch: r)
        return result
    }
    func calld<D>(batch: Matrix<D, InputDimension, Differentiable<Tail.T>>) -> Matrix<D, Tail.OutputDimension, Differentiable<Tail.T>> {
        let r: Matrix<D, Tail.InputDimension, Differentiable<T>> = layer.calld(batch: batch)
        let result: Matrix<D, OutputDimension, Differentiable<T>> = tail.calld(batch: r)
        return result
    }
}
struct MCP<BatchDimension, Layers: LayersProtocol> where Layers.T: DifferentiableValue {
    var layers: Layers
    let lossFunction: (Matrix<BatchDimension, Layers.OutputDimension, Layers.T>) -> Layers.T = \.normSquared.value
    let lossFunction2: (Matrix<BatchDimension, Layers.OutputDimension, Differentiable<Layers.T>>) -> Differentiable<Layers.T> = \.normSquared.value
    func call(batch: Matrix<BatchDimension, Layers.InputDimension, Layers.T>, ys: Vector<BatchDimension, Layers.T>) -> Layers.T {
        let r: Matrix<BatchDimension, Layers.OutputDimension, Layers.T> = layers.call(batch: batch)
        return lossFunction(r-ys)
    }
    func calld(batch: Matrix<BatchDimension, Layers.InputDimension, Layers.T>, ys: Vector<BatchDimension, Layers.T>) -> Differentiable<Layers.T> {
        let r: Matrix<BatchDimension, Layers.OutputDimension, Differentiable<Layers.T>> = layers.calld(batch: batch.constant)
        return lossFunction2(r-ys.constant)
    }
}

struct GD<Parameters: MutableCollection> where Parameters.Element: TensorProtocol, Parameters.Element.T: DifferentiableProtocol, Parameters.Element.T.ValueType: Numeric, Parameters.Element.T: DifferentiableValue {
    var parameters: Parameters
    let objective: (Parameters)->Parameters.Element.T
    let count: Int
    init(parameters: Parameters, objective: @escaping (Parameters)->Parameters.Element.T) {
        self.parameters = parameters
        self.objective = objective
        count = parameters.reduce(0) {$0+$1.elementCount}
    }
    mutating func step(learningRate: Parameters.Element.T.ValueType, helper: ([Tensor<Parameters.Element.Types, Parameters.Element.IndexType, Undifferentiable<Parameters.Element.T.ValueType>>])->Undifferentiable<Parameters.Element.T.ValueType>) {
        print(parameters)
        let df = objective(parameters)
        print(df)
        let dx = TensorGradients(of: df, forTensors: parameters)
        print(dx)
        _ = dx.update(stepSize: learningRate, parameters: &parameters)
        print(parameters)
        let undiff = parameters.map(\.view)
        print(undiff)
        print(helper(undiff))
        print(objective(parameters))
    }
}
struct Model<T: DifferentiableValue&SignedNumeric&Comparable>:TensorModel, CustomStringConvertible where T.Magnitude == T {
    func objective() -> T {
        (x-c).normSquared.value
    }
    var x: Vector<R, T>
    var c: Vector<R, T>
    var description: String {
        "\(x) -> \(objective())"
    }
}
extension Model: DifferentiableTensorModel where T: DifferentiableProtocol, T.ValueType: SignedNumeric, T.ValueType.Magnitude==T.ValueType {

    var view: Model<Undifferentiable<T.ValueType>> {
        .init(x: x.view, c: c.view)
    }
}

final class NNTests: XCTestCase {

//    func testGradientDecent() {
//        let x: Vector<R, Float> = .init(shape: 3, initialValue: 1)
//        let y: Vector<R, Float> = .init(shape: 3, initialValue: 0.5)
//        var m = Model<Differentiable<Float>>(x: x.differentiable, c: y.constant)
//        _ = m.gradientDescent(maxIterations: 200, initialStepSize: -0.6, stepReductionFactor: 0.5, stepIncrementFactor: 1.01,c1:0.5, tolerance: Float.ulpOfOne*Float.ulpOfOne)
//        print(m.x)
//    }

    func testGD() {
        enum R {}
        enum C {}
        let x: Vector<R, Float> = [1, 2, 3]
        print(x.constant.view)
        let y: Vector<C, Float> = [3, 4, 5]
        var m: Matrix<R, C, Float> = .init(shape: [3, 3], uniformValuesIn: -1..<1)
//        struct Test:Objective {
//            func objective<T>(parameters: [Matrix<R,C,T>]) -> T where T : DifferentiableValue {
//                (parameters[0]*x-y).normSquared.value
//            }
//
//            var parameters:[Matrix<R, C, Float>]
//            var constants: [Vector<]
//            let x:Vector<R,Float>
//            let y:Vector<C,Float>
//        }
        print(m)
        func function<T: FloatingPoint&DifferentiableValue>(m: Matrix<R, C, T>, xx: Vector<R, T>, yy: Vector<C, T>) -> T {
            (m*xx-yy).normSquared.value
        }
        var gd=GD<[Matrix<R, C, Differentiable<Float>>]>(parameters: [m.differentiable]) { mm in function(m: mm[0], xx: x.constant, yy: y.constant)}
//        let f: (Vector<R, Float>, Vector<C, Float>) -> Scalar<Float> = {xx, yy in (m*xx-yy).normSquared}
//        print(f(x, y))
        gd.step(learningRate: -0.01, helper: { mm in function(m: mm[0], xx: x.constant.view, yy: y.constant.view)})
        return
        for _ in 0..<30 {
            let mm: Matrix<R, C, Differentiable<Float>> = m.differentiable
            let g = mm.gradients(of: function(m: mm, xx: x.constant, yy: y.constant), multipliedBy: -0.05)
//            print(g)
            m += g
//            print(m)
            print(function(m: m, xx: x, yy: y))
        }
        print(m)
        print(m*x as Vector<C, Float>)
    }
    func testNN1() {
        enum Layer1 {}
        enum Layer2 {}
        enum BatchDimension {}
        let xs: Matrix<BatchDimension, DataDimension, Float> = [[2, 3, -1], [3, -1, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
        let ys: Vector<BatchDimension, Float> = [1, -1, -1, 1]
        print(xs)
        var rng = SystemRandomNumberGenerator()
        let n: MCP<BatchDimension, Layers<DataDimension, Layers<Layer1, EndLayer<Layer2, Float>>>> = .init(layers: .init(nin: 3, nouts: [2, 1], rng: &rng))
        print(n)
        let fx = n.call(batch: xs, ys: ys)
        print(fx)
        //        let g:GradientDescent<Matrix<BatchDimension, DataDimension, Float>> = .init(function: {x in n.call(batch: x, ys: ys)}, diffFunction: {x in n.calld(batch: x, ys: ys)})
        //        g.step(steps: 1, stepSize: 0.01, parameters: &n.layers.layer)
        //        print(fx[0].value.gradient[xs.first!.first!.value])
        //        let fx:[Differentiable<Float>] = n(x:x)
        //        print(fx)
        //        print(fx.map{$0.gradient[x[0].value]})
        //        print(fx[0].graph)
        //        print(fx.gradient[x[0].value])
        //        print(n(x:[1,2,3,4] as Vector<Void,Float>))
    }

//    func testSeq() {
//        let a:Vector<R,Int> = [1,2,3]
//        let b:Matrix<R,R,Int> = [[3,4],[4,5]]
//        var ss = [a.cast(),b]
//        var s = ss.flattened
//        print(s)
//        print(s.count)
//        print(ss)
//        for i in s {
//            print(i, terminator: " ")
//        }
//        print()
//        for offset in 1...s.count {
//            var i = s.startIndex
//            while i < s.endIndex {
//                s[i] += 1
//                s[i] %= 10
//                print(s[i], terminator: " "+repeatElement("  ", count: offset-1).joined())
//                i = s.index(i, offsetBy: offset)
//            }
//            print()
//        }
//        print(s)
//        ss.update(from: s)
//        print(ss)
//    }

}
