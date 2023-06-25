//
//  GradientTests.swift
//  
//
//  Created by Koen Hendrikx on 16/06/2023.
//

import XCTest
import TensorPackage

struct Quadratic<T: FloatingPoint&DifferentiableValue>:TensorModel, CustomStringConvertible {
    let A: Matrix<R, R, T>
    var x: Vector<R, T>
    let x0: Vector<R, T>
    init(A: Matrix<R, R, T>, x0: Vector<R, T>, x: Vector<R, T>) {
        self.A = A
        self.x0 = x0
        self.x = x
    }
    func objective() -> T {
        let d: Vector<R, T> = x - x0
        let Ax: Vector<R, T> = A*d
        let r: Scalar<T> = Ax*d
        return r.value
    }
    var description: String {
        "\(A) \(x) \(x0) -> \(objective())"
    }
}
extension Quadratic: DifferentiableTensorModel where T: DifferentiableProtocol, T.ValueType: BinaryFloatingPoint&DifferentiableValue, T.ValueType.Stride==T.ValueType {
    init<RNG: RandomNumberGenerator>(size: Int, rng:inout RNG) where T.ValueType: BinaryFloatingPoint, T.ValueType.RawSignificand: FixedWidthInteger {
        var _a = Matrix<R, R, T.ValueType>(shape: [ScalarIndex(size), ScalarIndex(size)], uniformValuesIn: 0..<1, using: &rng)
        _a = _a ⨂ _a.Transposed
        let a: Matrix<R, R, T> = _a.constant as! Matrix<R, R, T>
        let _x0: Vector<R, T.ValueType> = .init(shape: [ScalarIndex(size)], uniformValuesIn: 0..<1, using: &rng)
        let _x: Vector<R, T.ValueType> = .init(shape: [ScalarIndex(size)], uniformValuesIn: 0..<1, using: &rng)
        let x0: Vector<R, T> = _x0.constant as! Vector<R, T>
        let x: Vector<R, T> = _x.differentiable as! Vector<R, T>
        self.init(A: a, x0: x0, x: x)
    }

    init<RNG: RandomNumberGenerator>(eigenvalues: [T.ValueType], rng:inout RNG) where T.ValueType: BinaryFloatingPoint, T.ValueType.RawSignificand: FixedWidthInteger {
        let N = ScalarIndex(eigenvalues.count)
        let u: Vector<R, T.ValueType> = .init(shape: [N], uniformValuesIn: 0..<1, using: &rng)
        let norm=u.normSquared
        let uu: Matrix<R, R, T.ValueType> = (u⨂u)*(2/norm.value)
        let U: Matrix<R, C, T.ValueType> = Matrix<R, C, T.ValueType>(diag: eigenvalues.map {_ in 1}) - uu.cast()
        let _A: Matrix<R, R, T.ValueType> = (U ⨂ Matrix<C, C, T.ValueType>(diag: eigenvalues) as Matrix<R, C, T.ValueType>) ⨂ U
        let A: Matrix<R, R, T> = _A.constant as! Matrix<R, R, T>
        let _x0: Vector<R, T.ValueType> = .init(shape: [N], uniformValuesIn: 0..<1, using: &rng)
        let _x: Vector<R, T.ValueType> = .init(shape: [N], uniformValuesIn: 0..<1, using: &rng)
        let x0: Vector<R, T> = _x0.constant as! Vector<R, T>
        let x: Vector<R, T> = _x.differentiable as! Vector<R, T>
        self.init(A: A, x0: x0, x: x)
    }

    var view: Quadratic<Undifferentiable<T.ValueType>> {
        .init(A: A.view, x0: x0.view, x: x.view)
    }
}

final class GradientTests: XCTestCase {

    func testQuadratics() {
        var rng: RandomNumberGenerator = NumPyRandomSource(seed: 2)
//        var f:Quadratic<Differentiable<Double>> = .init(eigenvalues:[10,1,0.1,0.01,0.001], rng: &rng)
        var f: Quadratic<Differentiable<Double>> = .init(eigenvalues: [1, 1, 1, 1, 0.0001], rng: &rng)
        let state = f.state
        print(f.x)
        var gradientEvals = 0
        for _ in 0..<20 {
            print(f.objective())
            let stepSize = f.findStepSize(alpha: 0.8, c1: 0.5)
            if let c1 = f.gradientDescent(maxIterations: 1000, stepSize: stepSize, tolerance: .ulpOfOne, gradientEvals: &gradientEvals) {
                if c1.steps < 1000 {
                    break
                }
            }
//            print(f.x-f.x0)
        }
        print(f.objective())
        print(gradientEvals)
        print(f.x-f.x0)
        f.state = state
        print(f.x-f.x0)
        print(f.x)
    }

    func testGradientPerformance() {
        let c: Vector<R, Double> = .init(shape: [100], initialValue: 1)
        let k: Differentiable<Double> = .init(value: 2)
        let x: Vector<R, Differentiable<Double>> = c.differentiable*k
        let half: Differentiable<Double> = .init(constant: 0.5)
        var xv = x.view
        var s = x
        while s.count > 5 {
            s = s[1...]*half + s[0..<s.count-1]*half
            print(s)
        }
        var y = s.elementSum
        print(y)
        let g = y.gradient
        print(g[k])
        print(g[x[0].value])
        xv[0].value += 10
    }

}
