//
//  ValueTests.swift
//  TensorTests
//
//  Created by Koen Hendrikx on 14/05/2023.
//

import XCTest
import TensorPackage
import RealModule
import RegexBuilder
import os
import os.log

enum R {}
enum C {}

struct T5<T: DifferentiableValue&Numeric>:TensorModel where T.Magnitude==T {
    let v: Vector<R, T>
    let w: Vector<C, T>
    func objective() -> T {
        v.normSquared.value+w.normSquared.value
    }
}
struct T6<T: DifferentiableValue&Numeric>:TensorModel where T.Magnitude==T {
    let t: T5<T>
    let w: Vector<C, T>
    func objective() -> T {
        t.objective()*w.elementSum.magnitude
    }
}

extension T5: DifferentiableTensorModel where T: DifferentiableProtocol, T.ValueType: SignedNumeric, T.ValueType.Magnitude==T.ValueType {
    var view: T5<Undifferentiable<T.ValueType>> {
        .init(v: v.view, w: w.view)
    }
}
extension T6: DifferentiableTensorModel where T: DifferentiableProtocol, T.ValueType: SignedNumeric, T.ValueType.Magnitude==T.ValueType {
    var view: T6<Undifferentiable<T.ValueType>> {
        .init(t: t.view, w: w.view)
    }
}

final class ValueTests: XCTestCase {
    static let logger = OSLog(subsystem: "test", category: .pointsOfInterest)

    func extrapolate(x: [Double], y: [Double], x0: Double) -> Double {
        var s: Double = 0
        for i in y.indices {
            var v = y[i]
            for j in y.indices {
                if i != j {
                    v = v * (x[j]-x0) / (x[j]-x[i])
                }
            }
            s += v
        }
        return s
    }
    func testUnaryFunction(f: (Differentiable<Double>)->Differentiable<Double>, range: Range<Double>) {
        var g = SystemRandomNumberGenerator()
        for _ in 0..<20 {
            let x: Double = .random(in: range, using: &g)
            let x2: Differentiable<Double> = .init(value: x)
            let y2 = f(x2)
            let df = y2.gradient[x2]
            var xy: [(Double, Double)] = []
            for step in 0..<100 {
                let dx = 0.1*Foundation.pow(2.0, -Double(step))
                if (x-dx).sign != (x+dx).sign {
                    continue
                }
                let x0: Differentiable<Double> = .init(value: x-dx)
                let x1: Differentiable<Double> = .init(value: x+dx)
                let y0 = f(x0)
                let y1 = f(x1)
                let dydx = (y1.value-y0.value)/(x1.value-x0.value)
                if dydx.isFinite {
                    xy.append((dx, dydx))
                    if xy.count == 20 {
                        break
                    }
                }
            }
            if xy.isEmpty {
                fatalError()
            }
            let extra = extrapolate(x: xy.map(\.0), y: xy.map(\.1), x0: 0)
//            print("\(xy) \(extra) \(df)")
//            print((extra-df).magnitude)
            XCTAssertEqual(df, extra, accuracy: Swift.max(1e-6, df.magnitude*1e-6))
        }
    }
    func testUnaryFunctions() {
        testUnaryFunction(f: Differentiable<Double>.acos, range: -1.0..<1.0)
        testUnaryFunction(f: Differentiable<Double>.erf, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.erfc, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.sin, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.cos, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.tan, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.sinh, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.cosh, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.tanh, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.exp, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.exp2, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.exp10, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.asin, range: -1.0..<1.0)
        testUnaryFunction(f: Differentiable<Double>.atan, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.asinh, range: -10.0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.acosh, range: 1..<10.0)
        testUnaryFunction(f: Differentiable<Double>.atanh, range: -1.0..<1.0)
        testUnaryFunction(f: Differentiable<Double>.expMinusOne, range: -1.0..<1.0)
        testUnaryFunction(f: Differentiable<Double>.log(_:), range: 0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.log2, range: 0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.log10, range: 0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.log(onePlus:), range: 0..<10.0)
        testUnaryFunction(f: Differentiable<Double>.sqrt(_:), range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, 1)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, -1)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, 2)}, range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, -2)}, range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, 3)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, -3)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, 1.0)}, range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, -1.0)}, range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, 2.0)}, range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, -2.0)}, range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, 3.0)}, range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow($0, -3.0)}, range: 0..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow(1.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow(2.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.pow(3.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: \.magnitude, range: -10..<10.0)
        testUnaryFunction(f: \.reciprocal!, range: -10..<10.0)
        testUnaryFunction(f: \.ulp, range: -10..<10.0)
        testUnaryFunction(f: \.nextUp, range: -10..<10.0)
        testUnaryFunction(f: \.nextDown, range: -10..<10.0)
        testUnaryFunction(f: \.significand, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: 1.0, x: $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: -1.0, x: $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: 2.0, x: $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: -2.0, x: $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: 3.0, x: $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: -3.0, x: $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: $0, x: 1.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: $0, x: -1.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: $0, x: 2.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: $0, x: -2.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: $0, x: 3.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.atan2(y: $0, x: -3.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot($0, 1.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot($0, -1.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot($0, 2.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot($0, -2.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot($0, 3.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot($0, -3.0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot(1.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot(-1.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot(2.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot(-2.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot(3.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>.hypot(-3.0, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd(3, 4, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd(3, $0, 4)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd($0, 3, 4)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd(3, -4, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd(3, $0, -4)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd($0, 3, -4)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd(-3, -4, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd(-3, $0, -4)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd($0, -3, -4)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd(-3, 4, $0)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd(-3, $0, 4)}, range: -10..<10.0)
        testUnaryFunction(f: {Differentiable<Double>._mulAdd($0, -3, 4)}, range: -10..<10.0)

    }
    func test<C: Collection, T: Strideable&Numeric>(f: (Differentiable<T>, Differentiable<T>)->Differentiable<T>, range: C) where C.Element == (T, T), T: BinaryFloatingPoint, T: ExpressibleByFloatLiteral, T.Stride == T {
        for i in range {
            let x = Differentiable<T>(value: i.0+0.0001)
            let y = Differentiable<T>(value: i.1+0.0001)
            let F = f(x, y)
            let g = F.gradient
            let dx = g[x]
            let dy = g[y]
            let x1 = (f(x+Differentiable<T>(value: 0.0001), y).value - F.value)*10000
            let y1 = (f(x, y+Differentiable<T>(value: 0.0001)).value - F.value)*10000
            XCTAssertEqual(x1, dx, accuracy: Swift.max(0.001, dx*0.001))
            XCTAssertEqual(y1, dy, accuracy: Swift.max(0.001, dy*0.001))
        }
    }

    func testInt() {
//        test(f: {x, y in x.truncatingRemainder(dividingBy: y)}, range: [(0.0, 1.0), (1.0, 1.0), (-5.0, -10.0), (-10.0, -5.0)])
        test(f: { x, y in x.distance(to: y)}, range: [(0, 1), (1, 1), (-5, 10), (-10, -5)])
    }
    func testTensor() {
        let a: Differentiable<Vector<R, Double>> = .init(value: Vector<R, Double>(arrayLiteral: 1, 2, 3))
        let b: Differentiable<Vector<R, Double>> = .init(value: Vector<R, Double>(arrayLiteral: 4, 5, 6))
        let c1: Differentiable<Vector<R, Double>> = b+1
        let c: Differentiable<Vector<R, Double>> = a*c1
        XCTAssertEqual(c.value, [5, 12, 21])
        XCTAssertEqual(a.value, c.gradient[b])
        XCTAssertEqual(b.value+1, c.gradient[a])
        let s: Scalar<Double> = a.value*b.value
        XCTAssertEqual(32, s)
        let ds: Differentiable<Scalar<Double>> = a⨂b
        XCTAssertEqual(32, ds)
        XCTAssertEqual(b.value, ds.gradient[a])
        XCTAssertEqual(a.value, ds.gradient[b])
        let s1: Differentiable<Scalar<Double>> = ((a*a).norm)
        XCTAssertEqual(9.899494936611665, s1.value)
        XCTAssertEqual([0.20203050891044214, 1.616244071283537, 5.454823740581938], s1.gradient[a])
        let s2: Differentiable<Scalar<Double>> = ((a*a).normSquared)
        XCTAssertEqual(98, s2.value)
        XCTAssertEqual([4.0, 32.0, 108.0], s2.gradient[a])
//        let d: Differentiable<Scalar<Double>> = a*b
    }
    func testTensor2() {
        let x: Differentiable<Double> = .init(value: 1)
        let a: Vector<R, Differentiable<Double>> = .init(fromArray: [x, x*2, x*3])
        let b: Vector<R, Differentiable<Double>> = [5, 6, 7]
        let c: Matrix<R, R, Differentiable<Double>> = a*b
        let d: Scalar<Differentiable<Double>> = a*b
        XCTAssertEqual([[5, 10, 15], [6, 12, 18], [7, 14, 21]], c)
        XCTAssertEqual(38, d)
        XCTAssertEqual(14, a.normSquared)
        XCTAssertEqual(14, a.norm.value.gradient[x]*a.norm.value.gradient[x], accuracy: 1e-10)
        XCTAssertEqual(38, d.value.gradient[x])
        let m: Matrix<R, C, Double> = [[1, 2], [3, 4]]

        var diff = m.differentiable * x
        XCTAssertEqual([[1.0, 2.0], [3.0, 4.0]], diff.gradients(x))
        diff[0][0].value = x*x+x
        XCTAssertEqual([[2.0, 2.0], [3.0, 4.0]], diff)
        XCTAssertEqual([[2.0, 2.0], [3.0, 4.0]], diff)
        XCTAssertEqual([[3.0, 2.0], [3.0, 4.0]], diff.gradients(x))
        // (30x^2+x^4+2x^3)^0.5 -> 0.5((30x^2+x^4+2x^3)^-0.5)*(60x = 35/sqrt(33)
        XCTAssertEqual(6.092717958449424, diff.norm.gradients(x), accuracy: 1e-10)
    }

//    func testCast() {
//        print(MemoryLayout<Differentiable<Float>>.size)
//        print(MemoryLayout<Undifferentiable<Float>>.size)
//        func myFunc<T: TensorProtocol>(x: T) -> T.T where T.T: FloatingPoint {
//            x.elements.enumerated().map {($0.element-T.T($0.offset))*$0.element}.reduce(0, +)
//        }
//        var g: GradientDescent<Vector<R, Double>> = .init(function: myFunc(x:), diffFunction: myFunc(x:))
//        var x: Vector<R, Double> = [1, 2, 3]
//        print(g.step(steps: 100, stepSize: -0.1, parameters: &x))
//        print(x)
//    }

    func testCast() {
        let a: Differentiable<Int> = .init(value: 3)
        let v: Differentiable<Double> = .init(value: 10)
        let v3: Differentiable<Double> = v*v*3
        let f: Differentiable<Float> = .init(value: v3)
        let b: Differentiable<Float> = .init(value: a*a+a)
        XCTAssertEqual(60, f.gradient[v])
        XCTAssertEqual(7, b.gradient[a])
    }

    func testConst() {
        let x: Differentiable<Double> = .init(value: 10)
        let y: Differentiable<Double> = .init(value: 2)
        let c: Differentiable<Double> = 3
        let d: Differentiable<Double> = 4
        XCTAssertTrue(c.isConstant)
        XCTAssertFalse(x.isConstant)
        XCTAssertTrue((c+d).isConstant)
        XCTAssertEqual(7, c+d)
        XCTAssertEqual(13, (c*d+1))
        XCTAssertTrue((c*d+1).isConstant)
        let z = x*y+c+d
        XCTAssertEqual(2, z.gradient[x])
        XCTAssertEqual(10, z.gradient[y])
        XCTAssertEqual(0, z.gradient[c])
        XCTAssertEqual(0, z.gradient[d])
        let t = x*c+d+d*x
        XCTAssertFalse(t.isConstant)
        XCTAssertEqual(74, t)
        XCTAssertEqual(7, t.gradient[x])
        XCTAssertEqual(0, t.gradient[c])
        XCTAssertEqual(0, t.gradient[d])
    }

    func computeMemoryUsed() -> Int {
        var info = task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout.size(ofValue: info)) / 4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: Int32.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_BASIC_INFO),
                          task_info_t($0),
                          &count)
            }
        }
        assert(kerr == KERN_SUCCESS)
        return Int(info.resident_size)
    }

    func testRange() {
        let N: UInt32 = 10

        let m: Vector<R, Differentiable<Double>> = Vector<R, Double>(shape: [N], function: {_ in 1}).differentiable
        let x: Differentiable<Double> = m[m.index(m.startIndex, offsetBy: 5)].value
        print(m[5].value)
        print(x)
        var m1 = m
        for _ in 0..<N-2 {
            m1 = m1[1...]+m1[0..<m1.count-1]
            print(m1)
//            print(m1.indices.map{m1[$0].value.gradient[x]})
//            print(m1.gradients(x))
            print(m.gradients(of: m1.first!.value))
//            print(m1[m1.startIndex].value.gradient[x])
        }
    }

    func testMemoryUse1() {
        let m1 = computeMemoryUsed()
        #if DEBUG
        let N = 100000
        #else
        let N = 10000000
        #endif
        for _ in 0..<10 {
            do {
                var m: [Differentiable<Double>] = (0..<N).map {.init(value: Double($0))}
                print((computeMemoryUsed()-m1)/Int(N))
                let a = m.enumerated().map {$0.element*m[($0.offset+1)%N]}
                print((computeMemoryUsed()-m1)/Int(N))
                m = a
                print((computeMemoryUsed()-m1)/Int(N))
            }
            print((computeMemoryUsed()-m1)/Int(N))
        }
    }
    func testMemoryUse() {
        let m1 = computeMemoryUsed()
        let N: UInt32 = 100
//        var m:[Differentiable<Double>] = (0..<N).map{.init(value:Double($0))}
        let v: Matrix<R, R, Differentiable<Double>> = .init(shape: [N, N], function: {x in .init(value: Double(x.y) - Double(x.x))})
//        print(v)
        print((computeMemoryUsed()-m1)/Int(N*N))
        let w: Matrix<R, R, Differentiable<Double>> = v⨂(v.transpose() as Matrix<R, R, Differentiable<Double>>)
        print((computeMemoryUsed()-m1)/Int(2*N*N))
        print(v.size)
        print(w)
//        print(v.gradients(of: w.norm.value))
    }

    func testSecondDerivative() {
        let d: Differentiable<Differentiable<Double>> = .init(value:.init(value: 5))
        let d2 = d*d
        XCTAssertEqual(25, d2.value)
        let d2g = d2.gradient
        print(d2g[d])
        XCTAssertEqual(10, d2g[d])
        let d2g2 = d2g[d].gradient
        XCTAssertEqual(2, d2g2[d.value])
        print(d2.graph)
        print(d2g[d].graph)
        let d3 = d2*d
        XCTAssertEqual(125, d3.value)
        let d3g = d3.gradient
        let d3g2 = d3g[d].gradient
        XCTAssertEqual(75, d3g[d])
        XCTAssertEqual(30, d3g2[d.value])
        let d4 = d2*d2
        XCTAssertEqual(625, d4.value)
        let d4g = d4.gradient
        let d4g2 = d4g[d].gradient
        XCTAssertEqual(4*125, d4g[d])
        XCTAssertEqual(12*25, d4g2[d.value])
        print(d4.graph)
        print(d4g[d].graph)
    }

    func testMagnitude() {
        let a: Differentiable<Float> = .init(value: -2)
        let b = a.magnitude
        XCTAssertEqual(-1, b.gradient[a])
        let c: Differentiable<Vector<R, Float>> = .init(value: [1, -2, 3])
        let m: Differentiable<Vector<R, Float>> = (c*c*c).magnitude
        XCTAssertEqual([1.0, 8.0, 27.0], m.value)
        XCTAssertEqual([3.0, -12.0, 27.0], m.gradient[c])
        print(m.graph)
    }

    func testTensorTuple() {
        let v: Vector<R, Float> = [1, 2, 3]
        let w: Vector<C, Float> = [1, 2, 3]
        var vv = v.differentiable
        var ww = w.differentiable
        let vv2: Scalar<Differentiable<Float>> = vv*(ww.cast() as Vector<R, Differentiable<Float>>)
        let s: Scalar<Differentiable<Float>> = vv.sum()+vv2
        print(s)
        var tensors = (a:vv, (b:ww, c:ww))
        let g: TensorGradients = .init(of: s.value, forTensors: tensors)
        print(g)
        _ = g.update(stepSize: -0.01, parameters: &tensors)
        print(tensors)
        let g2: TensorGradients = .init(of: s.value, forTensors: [vv, ww] as [any TensorProtocol])
        if let dv: Vector<R, Float> = g2["_0"] {
            print(dv*(0.001 as Vector<R, Float>))
        }
        print(g2)
        let g3: TensorGradients = .init(of: s.value, forTensors: vv[1...])
        if let dv3: Vector<R, Float> = g3[""] {
            print(dv3*(0.001 as Vector<R, Float>))
        }
        print(g3)
        var vv3 = vv
        vv3[1] += vv3[0]
        vv3[2] += vv3[1]
        vv3[0] += vv3[2]
        print(vv3)
        var g4: TensorGradients = .init(of: (vv3.sum() as Scalar<Differentiable<Float>>).value, forTensors: vv)
        var g4b: TensorGradients = .init(of: (vv3.sum() as Scalar<Differentiable<Float>>).value, forTensors: vv)
        if let dv4: Vector<R, Float> = g4[""] {
            print(dv4*(0.001 as Vector<R, Float>))
        }
        print(g4)
//        g4.updateGradients(factor: 1, factor2: 0.5, other: g4b)
        print(g4)
        vv.values = [1, 2, 3]
        ww.values = [4, 5, 6]
//        var t5: T5 = .init(v: vv, w: ww)
//        t5.gradientDescent(stepSize: -0.001)
//        print(t5)
//        t5.gradientDescent(stepSize: -0.001)
//        print(t5)
//        let w3: Vector<C, Float> = [1, 2, 3]
//        vv.values = [1, 2, 3]
//        ww.values = [4, 5, 6]
//        var t6: T6 = .init(t: .init(v: vv, w: ww), w: w3.differentiable)
//        t6.gradientDescent(stepSize: -0.00001)
//        print(t6)
//        t6.gradientDescent(stepSize: -0.0000001)
//        print(t6)
//        print(t6.objective())
    }

    func testUndifferentiable() {
        enum R {}
        #if DEBUG
        let list: [UInt32] = [100, 1000, 10000, 100000]
        #else
        let list: [UInt32] = [100, 1000, 10000, 100000, 1000000, 10000000]
        #endif
        for N: UInt32 in list {
            print("start \(N)")
            let v: Vector<R, Differentiable<Double>> = .init(shape: [N], uniformValuesIn: 0..<1)
            //        XCTAssertEqual(v.description, v.view.description)
            let v2 = v.view
            let clock = ContinuousClock()
            for _ in 0..<2 {
                let elapsed = clock.measure {
                    os_signpost(.begin, log: Self.logger, name: "testDifferentiable %d", "testDifferentiable %d", N)
                    print(v.sum() as Scalar<Differentiable<Double>>)
                    os_signpost(.end, log: Self.logger, name: "testDifferentiable %d", "testDifferentiable %d", N)
                }
                print(elapsed)
                let elapsed2 = clock.measure {
                    os_signpost(.begin, log: Self.logger, name: "testUndifferentiable %d", "testUndifferentiable %d", N)
                    print(v2.sum() as Scalar<Undifferentiable<Double>>)
                    os_signpost(.end, log: Self.logger, name: "testUndifferentiable %d", "testUndifferentiable %d", N)
                }
                print(elapsed2)
            }
        }
        print("done")
    }

    func testExpression() {
        let a: Differentiable<Float> = .init(value: 2)
        let b: Differentiable<Float> = .init(value: -3)
        let c: Differentiable<Float> = .init(value: 10)
        let bb = b+b
        XCTAssertEqual(2, bb.gradient[b])
        let bm1 = b.magnitude
        XCTAssertEqual(-1, bm1.gradient[b])
        let e = a*b
        let d = e+c
        let f: Differentiable<Float> = .init(value: -2)
        let L = f*d
        XCTAssertEqual(4, L.gradient[f])
        XCTAssertEqual(-2, L.gradient[e])
        XCTAssertEqual(-2, L.gradient[d])
        XCTAssertEqual(-2, L.gradient[c])
        XCTAssertEqual(-4, L.gradient[b])
        XCTAssertEqual(6, L.gradient[a])
        print(L.graph)
        // L = -2*(a*b+c) //-2(2*-3+10) = -2*(4)=8
        // dL/da = -2*b = -6
        // dl/db = -2*a = 4
        // dL/dc = -2
        let bm = b.magnitude*a.magnitude
        XCTAssertEqual(-2, bm.gradient[b])
        XCTAssertEqual(3, bm.gradient[a])
        XCTAssertEqual(0, bm.gradient[e])
        let xp1: Differentiable<Float> = Differentiable.pow(b, 4)
        print(xp1.value)
        print(xp1.gradient[b])
        let xdiv: Differentiable<Float> = c/a
        print(xdiv.gradient[c])
        print(xdiv.gradient[a])
        let diff = a - b
        let diff2 = b.distance(to: a)
        XCTAssertEqual(1, diff.gradient[a])
        XCTAssertEqual(-1, diff.gradient[b])
        XCTAssertEqual(diff, diff2)
        XCTAssertEqual(diff.gradient[a], diff2.gradient[a])
        XCTAssertEqual(diff.gradient[b], diff2.gradient[b])

//        let b:Expression<Int> = 3
//        print(a+b)
//        let c:Expression<UInt> = (a+b).magnitude
//        let d:Expression<Int> = c.distance(to: 4)
//        print(c)
//        print(d)
//        print(a+b == b+a)
//        XCTAssertTrue(a<b)
//        XCTAssertEqual(a+b, b+a)
//        XCTAssertEqual(a+b, b+a)
//        XCTAssertEqual(a.magnitude.magnitude,a.magnitude)
//        XCTAssertEqual(c.distance(to: 4), d)
//        XCTAssertTrue(a.magnitude<b.magnitude)
//        let map:[Expression<Int>:Int] = [a:1,a+b:2]
//        XCTAssertEqual(2,map[b+a])
//        let n: Expression<Int> = -b
//        XCTAssertEqual(b,-n)
//        print(n<<3+1|d)
//        print(c.eval)
//        print(d.eval)
    }

}
