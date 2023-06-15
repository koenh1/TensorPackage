//
//  GradientDescentTests.swift
//  TensorTests
//
//  Created by Koen Hendrikx on 10/06/2023.
//

import XCTest
import TensorPackage

struct Rosenbrock<T: FloatingPoint&DifferentiableValue>:TensorModel, CustomStringConvertible {
    var x: Vector<R, T>
    func objective() -> T {
        let r1: Vector<R, T> = x[0..<x.count-1]
        let r2: Vector<R, T> = x[1..<x.count]
        let r: Vector<R, T> = r2-r1*r1
        let m1: Vector<R, T> = r1 - 1
        let m2: Vector<R, T> = r*r*100+m1*m1
        return m2.elementSum
    }
    var description: String {
        "\(x) -> \(objective())"
    }
}
extension Rosenbrock: DifferentiableTensorModel where T: DifferentiableProtocol, T.ValueType: FloatingPoint&DifferentiableValue, T.ValueType.Stride==T.ValueType {
    var view: Rosenbrock<Undifferentiable<T.ValueType>> {
        .init(x: x.view)
    }
}

final class GradientDescentTests: XCTestCase {

    func testRosenBrock() {
        #if DEBUG
        print("debug mode")
        #endif
        let x: Vector<R, Float> = .init(shape: [50], uniformValuesIn: 10..<11)
        var r: Rosenbrock<Differentiable<Float>> = .init(x: x.differentiable)
//        print(r.objective())
        let time: ContinuousClock = .init()
        let t = time.now
        let c1 = r.gradientDescent(maxIterations: 50000, initialStepSize: -1, stepReductionFactor: 0.5, stepIncrementFactor: 1.1, tolerance: 1e-14)
        print(r.objective())
        print(r.x)
        print(time.now-t)
        print(c1)
//        r = .init(x: x.differentiable)
//        t = time.now
//        print(r.adaGrad(maxIterations: c1, learningRate: -1, tolerance: 1e-13))
//        print(r.objective())
//        print(r.x)
//        print(time.now-t)
    }

}
