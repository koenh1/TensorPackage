//
//  TensorTests.swift
//  TensorTests
//
//  Created by Koen Hendrikx on 16/04/2023.
//

import XCTest
import RealModule
import ComplexModule
import TensorPackage
// typealias Shape0 = Shape<NilTypeList, Int32>
// typealias Shape1<A> = Shape<TypeList<A, NilTypeList>, Int32>
// typealias Shape2<A, B> = Shape<TypeList<A, TypeList<B, NilTypeList>>, SIMD2<UInt32>>
// typealias Shape3<A, B, C> = Shape<TypeList<A, TypeList<B, TypeList<C, NilTypeList>>>, SIMD3<UInt32>>
// typealias Shape4<A, B, C, D> = Shape<TypeList<A, TypeList<B, TypeList<C, TypeList<D, NilTypeList>>>>, SIMD4<UInt32>>
// typealias Shape6<A, B, C, D, E> = Shape<TypeList<A, TypeList<B, TypeList<C, TypeList<D, TypeList<E, NilTypeList>>>>>, SIMD8<UInt32>>

final class TensorTests: XCTestCase {

//    func xtestMatching() {
//        //        let x=[1, 2, 1, 2]
//        //        let y=[1, 1, 2, 4, 2]
//        //        let tg = perfectMatch(a: x, b: y, f: {$0 < $1 }).map {(x[$0.key], y[$0.value])}
//        //        print(tg)
//        let a="AA".map {$0}
//        let b="AA".map {$0}
//        let c="A".map {$0}
//        let n = a.count+b.count+c.count+2
//        let source=n-2
//        let sink=n-1
//        var caps: [[Int8]] = .init(repeating: .init(repeating: 0, count: n), count: n)
//        for i in a.indices {
//            caps[source][i] = 1
//            for j in b.indices {
//                if a[i] == b[j] {
//                    caps[i][j+a.count] = 1
//                }
//            }
//            for j in c.indices {
//                if a[i] == c[j] {
//                    caps[i][j+a.count+b.count] = 1
//                }
//            }
//        }
//        for i in c.indices {
//            caps[i+a.count+b.count][sink] = 10
//            for j in b.indices {
//                if b[j] == c[i] {
//                    caps[j+a.count][i+a.count+b.count] = 1
//                }
//            }
//        }
//        for i in b.indices {
//            caps[source][a.count+i] = 1
//            if caps[i+a.count].allSatisfy({$0==0}) {
//                caps[i+a.count][sink] = 10
//            }
//        }
//        let orig = caps
//        print(dot(a: a, b: b, c: c, capacities: caps, source: source, sink: sink))
//        print(caps)
//        let f = maxFlow(capacities: &caps, source: source, sink: sink)
//        print(caps)
//        print(f)
//        var abmap: [Int: Int] = [:]
//        var acmap: [Int: Int] = [:]
//        var bcmap: [Int: Int] = [:]
//        for i in a.indices {
//            for j in b.indices {
//                if caps[i][j+a.count] < orig[i][j+a.count] {
//                    abmap[i] = j
//                }
//            }
//            for j in c.indices {
//                if caps[i][j+a.count+b.count] < orig[i][j+a.count+b.count] {
//                    acmap[i] = j
//                }
//            }
//        }
//        for i in b.indices {
//            for j in c.indices {
//                if caps[i+a.count][j+a.count+b.count] < orig[i+a.count][j+a.count+b.count] {
//                    bcmap[i] = j
//                }
//            }
//        }
//        print(abmap)
//        print(acmap)
//        print(bcmap)
//        for i in caps.indices {
//            for j in caps[i].indices {
//                caps[i][j] = Swift.max(0, orig[i][j] - caps[i][j])
//            }
//        }
//        print(caps)
//        print(dot(a: a, b: b, c: c, capacities: caps, source: source, sink: sink))
//    }

    // TODO: FloatingPoint
    // TODO:

    final class AnyTensor {
        static func + (lhs: AnyTensor, rhs: AnyTensor) -> AnyTensor {
            lhs.plus(rhs.value)
        }

        let value:any TensorProtocol
        let add:(any TensorProtocol) -> any TensorProtocol
        init(value:any TensorProtocol, add:@escaping (any TensorProtocol) -> any TensorProtocol) {
            self.value = value
            self.add = add
        }
        convenience init<L: TensorProtocol>(value: L) where L.T: AdditiveArithmetic {
            self.init(value: value, add: { rhs -> L in
                value.combine(rhs as! L, +)!
            })
        }
        func plus(_ rhs:any TensorProtocol) -> AnyTensor {
            .init(value: add(rhs), add: add)
        }
    }

//    func testAny() {
//        let v: Vector<R, Int> = [1, 2, 3]
//        let w: Scalar<Int> = 2
//        let v_:AnyTensor = .init(value:v)
//        let w_:AnyTensor = .init(value:w)
//        let z:AnyTensor = v_+w_
//    }
    func testOuter() {
        let v: Vector<R, Int> = [1, 2, 3]
        let w: Vector<C, Int> = [2, 3]
        var vw: Matrix<R, C, Int> = v.outer(rhs: w, *)
        vw[0][1].value += 1
        XCTAssertEqual([[2, 4], [4, 6], [6, 9]], vw)
        vw[0][1].value -= 1
        vw[0] = [1, 2]
        print(vw)
        XCTAssertEqual([3, 2], vw.size)
        XCTAssertEqual([[1, 2], [4, 6], [6, 9]], vw)
        let A: Matrix<R, C, Int> = [[1, 2, 3], [4, 5, 6]]
        let Aw: Tensor3<R, C, C, Int> = A.outer(rhs: w, *)
        XCTAssertEqual([2, 3, 2], Aw.size)
        let B: Matrix<T, R, Int> = [[7, 8, 9, 10, 11], [11, 12, 13, 14, 15], [14, 15, 16, 17, 18], [19, 20, 21, 22, 23]]
        let AB: Tensor4<R, C, T, R, Int> = A.outer(rhs: B, *)
        XCTAssertEqual([2, 3, 4, 5], AB.size)
        print(AB.shape)
        print(AB)
    }

    func testComplex() {
        let m: Matrix<R, C, Complex<Float32>> = [[-1, 2], [2, -3]]
        let s: Matrix<R, C, Complex<Float32>> = .sqrt(m)
        print(s)
        print(s*s)
    }

    func testSum() {
        let C: Tensor3<T, R, C, Int> = [[[0, 1, 2, 3], [ 4, 5, 6, 7], [ 8, 9, 10, 11]],
                                        [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        let Ct1: Tensor3<T, C, R, Int> = C.transpose()
        XCTAssertEqual([2, 4, 3], Ct1.size)
        let Ct2: Tensor3<R, T, C, Int> = C.transpose()
        let Ct3: Tensor3<R, C, T, Int> = C.transpose()
        let Ct4: Tensor3<C, T, R, Int> = C.transpose()
        let Ct5: Tensor3<C, R, T, Int> = C.transpose()
        let v1: Vector<T, Int>? = C.reduceTensor(zero: 0, op: +)
        let v2: Vector<T, Int>? = Ct5.reduceTensor(zero: 0, op: +)
        XCTAssertEqual([66, 210], v1)
        XCTAssertEqual([66, 210], v2)
        let tubesum: Matrix<R, C, Int>? = C.reduceTensor(zero: 0, op: +)
        XCTAssertEqual([[12, 14, 16, 18], [20, 22, 24, 26], [28, 30, 32, 34]], tubesum)
        let tubesum1: Matrix<R, C, Int>? = Ct1.reduceTensor(zero: 0, op: +)
        XCTAssertEqual(tubesum1, tubesum)
        let tubesum2: Matrix<R, C, Int>? = Ct2.reduceTensor(zero: 0, op: +)
        XCTAssertEqual(tubesum2, tubesum)
        let tubesum3: Matrix<R, C, Int>? = Ct3.reduceTensor(zero: 0, op: +)
        XCTAssertEqual(tubesum3, tubesum)
        let tubesum4: Matrix<R, C, Int>? = Ct4.reduceTensor(zero: 0, op: +)
        XCTAssertEqual(tubesum4, tubesum)
        let tubesum5: Matrix<R, C, Int>? = Ct5.reduceTensor(zero: 0, op: +)
        XCTAssertEqual(tubesum5, tubesum)
        let colsum3: Matrix<T, R, Int>? = C.reduceTensor(zero: 0, op: +)
        XCTAssertEqual([[6, 22, 38], [54, 70, 86]], colsum3)
        let rowsum3: Matrix<T, C, Int>? = C.reduceTensor(zero: 0, op: +)
        XCTAssertEqual([[12, 15, 18, 21], [48, 51, 54, 57]], rowsum3)
        let v: Vector<R, Int> = [1, 2, 3]
        let s: Scalar<Int>? = v.reduceTensor(zero: 0, op: +)
        XCTAssertEqual(6, s)
        let s2: Scalar<Int>? = C.reduceTensor(zero: 0, op: +)
        XCTAssertEqual(23*12, s2)
        let B: Matrix<R, C, Int> = [[1, 2], [3, 4], [5, 6]]
        let Bt: Matrix<C, R, Int> = B.transpose()
        let rowsum: Vector<R, Int>? = B.reduceTensor(zero: 0, op: +)
        let rowsum2: Vector<R, Int>? = Bt.reduceTensor(zero: 0, op: +)
        XCTAssertEqual([3, 7, 11], rowsum)
        XCTAssertEqual([3, 7, 11], rowsum2)
        let colsum: Vector<C, Int>? = B.reduceTensor(zero: 0, op: +)
        let colsum2: Vector<C, Int>? = Bt.reduceTensor(zero: 0, op: +)
        XCTAssertEqual([9, 12], colsum)
        XCTAssertEqual([9, 12], colsum2)
        let Av: Matrix<R, C, Int> = v.broadcast(to: B.shape)!
        let colsum4: Vector<C, Int> = Av.reduceTensor(zero: 0, op: +)!
        XCTAssertEqual([6, 6], colsum4)
        let rowsum4: Vector<R, Int> = Av.reduceTensor(zero: 0, op: +)!
        XCTAssertEqual([2, 4, 6], (rowsum4))
    }

    func testStride() {
        let v: Vector<R, UInt> = [1, 2, 3]
        let w: Vector<R, UInt> = v * [10]
        print(v)
        print(w)
        print(1.distance(to: 10))
        print(type(of: v.distance(to: w)))
        print(v.advanced(by: [1, 2, 3]))
    }

    func testSigned() {
        let v: Vector<R, Int> = [1, 2, 3]
        XCTAssertEqual([-1, -2, -3], -v)
        var w = v
        w.negate()
        XCTAssertEqual([-1, -2, -3], w)
        var A: Matrix<R, C, Int> = [[1, 2], [3, 4], [5, 6]] // 3x2
        var At: Matrix<C, R, Int> = A.transpose()
        XCTAssertEqual([[-1, -2], [-3, -4], [-5, -6]], -A)
        XCTAssertEqual([[-1, -3, -5], [-2, -4, -6]], -At)
        A.negate()
        At.negate()
        XCTAssertEqual([[-1, -2], [-3, -4], [-5, -6]], A)
        XCTAssertEqual([[-1, -3, -5], [-2, -4, -6]], At)
    }

    func testMultiply() {
        let v: Vector<R, Int> = [1, 2, 3]
        let w: Vector<C, Int> = [2, 3]
        let A: Matrix<R, C, Int> = [[1, 2], [3, 4], [5, 6]] // 3x2
        let At: Matrix<C, R, Int> = A.transpose()
        XCTAssertEqual([[1, 3, 5], [2, 4, 6]], At)
        let AA: Matrix<C, C, Int> = A * A
        XCTAssertEqual([2, 2], AA.size) // 2x3 X 3x2
        XCTAssertEqual([[35, 44], [44, 56]], AA)
        let vv: Matrix<R, R, Int> = v*v
        XCTAssertEqual([3, 3], vv.size) // 3x3
        XCTAssertEqual([[1, 2, 3], [2, 4, 6], [3, 6, 9]], vv)
        let vvv: Tensor3<R, R, R, Int> = vv * v
        XCTAssertEqual([3, 3, 3], vvv.size) // 3x3
        XCTAssertEqual([[[1, 2, 3], [2, 4, 6], [3, 6, 9]], [[2, 4, 6], [4, 8, 12], [6, 12, 18]], [[3, 6, 9], [6, 12, 18], [9, 18, 27]]], vvv)
        let Aw: Vector<R, Int> = A*w
        XCTAssertEqual([3], Aw.size) // 3x2 X 2x1 = 3x1
        XCTAssertEqual([8, 18, 28], Aw)
        let vw2: Scalar<Int> = v*v
        XCTAssertEqual([], vw2.size) // 3x2 X 2
        XCTAssertEqual(14, vw2)
        let A3: Matrix<R, C, Int> = A*3
        XCTAssertEqual([[3, 6], [9, 12], [15, 18]], A3)
        let At3: Matrix<C, R, Int> = At*Scalar<Int>(3)
        let At3b: Matrix<C, R, Int> = At*3
        let At3c: Matrix<C, R, Int> = A*Scalar<Int>(3)
        XCTAssertEqual([[3, 9, 15], [6, 12, 18]], At3)
        XCTAssertEqual(At3, At3b)
        XCTAssertEqual(At3, At3c)
        let vw: Matrix<R, C, Int> = v*w
        XCTAssertEqual([3, 2], vw.size) // 3x2
        XCTAssertEqual([[2, 3], [4, 6], [6, 9]], vw)
        // R(3)C(2) x C(2) -> R(3)
        let Atw: Vector<R, Int> = At*w
        XCTAssertEqual([3], Atw.size) // 3x2 X 2x1 = 3x1
        XCTAssertEqual([8, 18, 28], Atw)
        let AtA: Matrix<C, C, Int> = At*A
        XCTAssertEqual([2, 2], AtA.size) // 2x3 X 3x2
        XCTAssertEqual([[35, 44], [44, 56]], AtA)
        let AAt: Matrix<C, C, Int> = A*At
        XCTAssertEqual([2, 2], AAt.size) // 2x3 X 3x2
        XCTAssertEqual([[35, 44], [44, 56]], AAt)
        let AtAt: Matrix<C, C, Int> = At*At
        XCTAssertEqual([2, 2], AtAt.size) // 2x3 X 3x2
        XCTAssertEqual([[35, 44], [44, 56]], AtAt)
        let wv: Matrix<C, R, Int> = w*v
        XCTAssertEqual([2, 3], wv.size) // 3x2
        XCTAssertEqual([[2, 4, 6], [3, 6, 9]], wv)
        let ww: Matrix<C, C, Int> = w*w
        XCTAssertEqual([2, 2], ww.size) // 3x3
        XCTAssertEqual([[4, 6], [6, 9]], ww)
        let Avv: Matrix<R, C, Int> = A*vv
        XCTAssertEqual([3, 2], Avv.size) // 3x2
        XCTAssertEqual([[22, 28], [44, 56], [66, 84]], Avv)
        let Aww: Matrix<R, C, Int> = A*ww
        XCTAssertEqual([3, 2], Aww.size) // 3x2
        XCTAssertEqual([[16, 24], [36, 54], [56, 84]], Aww)
        let vvv2: Vector<R, Int> = vv*v
        XCTAssertEqual([3], vvv2.size) // 3x3
        XCTAssertEqual([14, 28, 42], vvv2)
        let Awv: Matrix<C, C, Int> = A*wv
        XCTAssertEqual([[44, 56], [66, 84]], Awv)
        let Atwv: Matrix<R, R, Int> = At*wv
        XCTAssertEqual([[8, 18, 28], [16, 36, 56], [24, 54, 84]], Atwv)
        let Atwv2: Matrix<R, R, Int> = wv*At
        XCTAssertEqual([[8, 16, 24], [18, 36, 54], [28, 56, 84]], Atwv2)
        let Atwvt: Matrix<R, R, Int> = Atwv.transpose()
        XCTAssertEqual(Atwv2, Atwvt)
    }

    func testMultiply2() {
        let rowvec2: Vector<C, Int> = [1, 2]
        let colvec3: Vector<R, Int> = [1, 2, 3]
        let A: Matrix<R, C, Int> = rowvec2 * colvec3
        print(A)
        XCTAssertEqual([[1, 2], [2, 4], [3, 6]], A)
        let B: Matrix<C, R, Int> = rowvec2 * colvec3
        print(B)
        XCTAssertEqual([[1, 2, 3], [2, 4, 6]], B)
    }
    func testAdditive() {
        var A: Matrix<R, C, Int> = [[1, 2], [3, 4], [5, 6]] // 3x2
        let rowvec2: Vector<C, Int> = [1, 2]
        let colvec3: Vector<R, Int> = [1, 2, 3]
        let scalar: Scalar<Int> = 1
        XCTAssertEqual([[2, 3], [4, 5], [6, 7]], A+scalar)
        XCTAssertEqual([[2, 3], [5, 6], [8, 9]], A+colvec3)
        XCTAssertEqual([[2, 4], [4, 6], [6, 8]], A+rowvec2)
        A += colvec3
        XCTAssertEqual([[2, 3], [5, 6], [8, 9]], A)
        A += rowvec2
        XCTAssertEqual([[3, 5], [6, 8], [9, 11]], A)
        A -= scalar
        XCTAssertEqual([[2, 4], [5, 7], [8, 10]], A)
        let B: Matrix<C, R, Int> = rowvec2.multiply(rhs: colvec3, zero: 0, multiplyOperator: *, sumOperator: +)!
        XCTAssertEqual([2, 3], B.size)
        XCTAssertEqual([[1, 2, 3], [2, 4, 6]], B)
        XCTAssertEqual([[3, 7, 11], [6, 11, 16]], B+A)
    }

    func testCompare() {
        let v: Vector<R, Int> = [1, 2, 3]
        let w: Vector<R, Int> = [2, 1, 4]
        let A: Matrix<R, C, Int> = [[1, 2], [3, 4], [5, 6]] // 3x2
        XCTAssertTrue(v<w)
        XCTAssertFalse(v>w)
        XCTAssertTrue(v*2>=w)
        XCTAssertTrue(A>=v.broadcast(to: A.shape)!)
    }

    func testScalar() {
        var s: Scalar<Double> = .init(value: 10)
        XCTAssertEqual("10.0", s.description)
        s=20
        s=20.0
        XCTAssertEqual(1, s.count)
        XCTAssertEqual("20.0", s.description)
        XCTAssertTrue(type(of: s) == type(of: s[0]))
        s[0]=10.0
        XCTAssertEqual("10.0", s.description)
        XCTAssertTrue(s == 10.0)
        XCTAssertTrue(s != 11.0)
        var x: Scalar<Int> = 10
        x += 20
        XCTAssertTrue(x == 30)
        XCTAssertTrue(x<40)
        XCTAssertEqual(30, x)
        XCTAssertTrue(Int.self==type(of: x.value))
        XCTAssertTrue(Double.self==type(of: s.value))
    }
    class TestClass: CustomStringConvertible {
        static var live = 0
        let v: String
        init(_ v: String) {
            self.v=v
            Self.live += 1
        }
        deinit {
            Self.live -= 1
        }
        var description: String {
            "(\(v))"
        }
    }
    func testClassVector() {
        do {
            var v: Vector<R, TestClass> = .init(shape: 4, initialValue: .init("x"))
            XCTAssertEqual("[(x), (x), (x), (x)]", v.description)
            v[3] = .init(.init("y"))
            print(v)
            v[2] = v[3]
            print(v)
            v[0] = v[0]
            print(v)
            XCTAssertEqual(2, TestClass.live)
            XCTAssertEqual("[(x), (x), (y), (y)]", v.description)
        }
        XCTAssertEqual(0, TestClass.live)
    }
    func testClassMatrix() {
        do {
            var v: Matrix<R, C, TestClass> = .init(shape: 3, 2, initialValue: .init("x"))
            XCTAssertEqual("[[(x), (x)], [(x), (x)], [(x), (x)]]", v.description)
            //            v[2] = [.init("y"), .init("z")]
            print(type(of: v[2][0]))
            v[2][0] = .init(.init("y"))
            v[2][1] = .init(.init("z"))
            v[1] = v[2]
            v[1] = v[1]
            XCTAssertEqual(3, TestClass.live)
            XCTAssertEqual("[[(x), (x)], [(y), (z)], [(y), (z)]]", v.description)
        }
        XCTAssertEqual(0, TestClass.live)
    }
    func testStringVector() {
        var v: Vector<R, String> = .init(shape: 4, initialValue: "x")
        v = ["x", "y", "z"]
        XCTAssertEqual("[x, y, z]", v.description)
        v[2] = "xx"
        XCTAssertEqual("[x, y, xx]", v.description)
        XCTAssertTrue(["x", "y", "xx"] == v)
        let r = v.applyBinary(rhs: v, +)
        XCTAssertEqual(["xx", "yy", "xxxx"], r)
    }
    func testIntVector() {
        var v: Vector<R, Int> = .init(shape: 4)
        v = [1, 2, 3]
        XCTAssertEqual("[1, 2, 3]", v.description)
        v[2] = 4
        XCTAssertEqual("[1, 2, 4]", v.description)
        XCTAssertEqual([1, 2, 4], v)
        let w = v.applyUnary(-)
        XCTAssertEqual([-1, -2, -4], w)
        v.apply({$0+=1})
        XCTAssertEqual([2, 3, 5], v)
    }
    func testOptionalIntVector() {
        var v: Vector<R, Int?> = .init(shape: 4, initialValue: nil)
        v = [.init(1), nil, .init(3)]
        XCTAssertEqual("[Optional(1), nil, Optional(3)]", v.debugDescription)
        v[1] = .init(4)
        v[0] = nil
        v[1].value = 5
        XCTAssertEqual("[nil, Optional(5), Optional(3)]", v.debugDescription)
        v[0].value = 10
        XCTAssertTrue([.init(10), .init(5), .init(3)] == v)
    }
    func testBoolVector() {
        var v: Vector<R, Bool> = .init(shape: 4, initialValue: false)
        XCTAssertEqual("[false, false, false, false]", v.description)
        v = [false, true, true]
        XCTAssertEqual("[false, true, true]", v.description)
        v[0] = true
        XCTAssertEqual("[true, true, true]", v.description)
        XCTAssertTrue([true, true, true] == v)
        let w: Vector<C, Bool> = v.cast()
        XCTAssertEqual(w.elements.map {$0}, v.elements.map {$0})
    }
    func testVector() {
        var v: Vector<R, Double> = .init(shape: 10)
        XCTAssertEqual(10, v.count)
        XCTAssertEqual(0.0, v.elements.reduce(0, +))
        v = [1, 2, 3]
        XCTAssertEqual(6.0, v.elements.reduce(0, +))
        XCTAssertEqual("[1.0, 2.0, 3.0]", v.description)
        XCTAssertTrue([1, 2, 3] == v)
        XCTAssertEqual(3, v.count)
        v[1] = 3
        XCTAssertEqual(1, v[1].count)
        XCTAssertEqual([3], v.size)
        XCTAssertEqual(7.0, v.elements.reduce(0, +))
        let w: Vector<R, Float> = v.mapTensor({Float($0)})
        print(w)
    }
    func testMatrix() {
        var m: Matrix<R, C, Double>
        m = [[1.0, 2.0, 3.0], [4, 5, 6]]
        let values: [Double] = m.elements.indices.map {m[$0]}
        XCTAssertEqual([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], values)
        let m1: Matrix<C, R, Double> = m.transpose()
        let values2: [Double] = m1.elements.indices.map {m1[$0]}
        XCTAssertEqual([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], values2)
        XCTAssertEqual([2, 3], m.size)
        XCTAssertEqual(21.0, m.elements.reduce(0, +))
        XCTAssertEqual(6, m.elementCount)
        XCTAssertEqual(3, m[1].elementCount)
        XCTAssertEqual("[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]", m.description)
        XCTAssertTrue([[1.0, 2.0, 3.0], [4, 5, 6]] == m)

        XCTAssertEqual("[1.0, 2.0, 3.0]", m[0].description)
        XCTAssertEqual("[4.0, 5.0, 6.0]", m[1].description)
        XCTAssertEqual([2, 3], m.size)
        m[1] = [3, 2, 1]
        XCTAssertEqual("[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]", m.description)
        m[1][2] = 3
        XCTAssertEqual("[[1.0, 2.0, 3.0], [3.0, 2.0, 3.0]]", m.description)
        m[1].apply({$0+=1})
        XCTAssertEqual([[1.0, 2.0, 3.0], [4.0, 3.0, 4.0]], m)
        m = 10
        XCTAssertEqual([1, 1], m.size)
        XCTAssertEqual("10.0", m.description)
        var mm: Matrix<R, C, Int> = [[1, 2], [0, 3], [3, 4]]
        mm.sort()
        XCTAssertEqual([[0, 3], [1, 2], [3, 4]], mm)
        let mms = mm.sorted(by: {$0.normSquared < $1.normSquared})
        XCTAssertEqual([[1, 2], [0, 3], [3, 4]], mms)

    }
    func testTensor3() {
        var t: Tensor3<T, R, C, Double> = .init(shape: 2, 3, 4)
        t[0] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        t[1] = [[10, 20, 30, 40], [50, 60, 70, 80], [90, 100, 110, 120]]
        print(t)
        XCTAssertEqual([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0], [90.0, 100.0, 110.0, 120.0]]], t)
        let v = t[1][2].elements.map {$0}
        XCTAssertEqual([90.0, 100.0, 110.0, 120.0], v)
        t[1][2] = [-1, -2, -3, -4]
        XCTAssertEqual([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], [[10.0, 20.0, 30.0, 40.0], [50.0, 60.0, 70.0, 80.0], [-1, -2, -3, -4]]], t)
    }

    func testSIMDIterator() {
#if DEBUG
#else
        typealias T = UInt32
        var reduceTensor: T = 0
        for _ in 0..<10000 {
            var it = SIMDIterator<SIMD4<T>>.init(size: [40, 30, 20, 10], stride: [1, 40, 1200, 24000])
            while let i = it.next() {
                reduceTensor &+= i
            }
        }
        print(reduceTensor)
#endif
    }

    enum T { }
    enum R { }
    enum C {}

    func testInit() {
        let v: Vector<R, Int> = .init(shape: [5]) {
            Int($0[0])
        }
        XCTAssertEqual([0, 1, 2, 3, 4], v)
        let m: Matrix<R, C, Int> = .init(shape: [2, 3]) {
            Int($0.x*10+$0.y)
        }
        XCTAssertEqual([[0, 10, 20], [1, 11, 21]], m)
        let mm: Tensor3<T, R, C, Int> = .init(shape: [2, 2, 2]) {
            Int($0.x*10+$0.y+$0.z*100)
        }
        XCTAssertEqual([[[0, 10], [1, 11]], [[100, 110], [101, 111]]], mm)
        let mm2: Tensor4<(T, R), T, R, C, Int> = .init(shape: [2, 2, 2, 2], function: {
            Int($0.x*10+$0.y+$0.z*100)+Int($0.w)*1000
        })
        XCTAssertEqual([[[[0, 10], [1, 11]], [[100, 110], [101, 111]]], [[[1000, 1010], [1001, 1011]], [[1100, 1110], [1101, 1111]]]], mm2)
    }

    func testOrder() {
        let a = [1, 2, 3, 2, 5, 6, 1]
        for _ in 0..<1000 {
            var b = a.shuffled()
            for i in stride(from: 0, to: a.count-1, by: 1) {
                if a[i] != b[i], let j = stride(from: i+1, to: a.count, by: 1).first(where: {b[$0]==a[i]}) {
                    b.swapAt(i, j)
                }
            }
            XCTAssertEqual(a, b)
        }
    }
    func testFlatten<X, Y, Z>(mm: Tensor3<T, R, C, Int>, x: X.Type, y: Y.Type, z: Z.Type) {
        let mmt: Tensor3<X, Y, Z, Int> = mm.transpose()
        let s3: Matrix<(T, C), R, Int> = mm.flatten()!
        let t3: Matrix<(T, C), R, Int> = mmt.flatten()!
        XCTAssertEqual([8, 3], s3.size)
        XCTAssertEqual([8, 3], t3.size)
        XCTAssertEqual(s3.description, t3.description)
        let s1: Matrix<T, (R, C), Int> = mm.flatten()!
        let t1: Matrix<T, (R, C), Int> = mmt.flatten()!
        let t1b: Matrix<T, (C, R), Int> = mmt.flatten()!
        XCTAssertEqual([2, 12], s1.size)
        XCTAssertEqual([2, 12], t1.size)
        XCTAssertEqual(s1.description, t1.description)
        XCTAssertFalse(t1.description == t1b.description)
        let s2: Matrix<(T, R), C, Int> = mm.flatten()!
        let t2: Matrix<(T, R), C, Int> = mmt.flatten()!
        let t2c: Matrix<(R, T), C, Int> = mm.flatten()!
        let t2b: Matrix<(R, T), C, Int> = mmt.flatten()!
        XCTAssertEqual([6, 4], s2.size)
        XCTAssertEqual([6, 4], t2.size)
        XCTAssertEqual([6, 4], t2b.size)
        XCTAssertFalse(t2.description == t2b.description)
        XCTAssertEqual(t2c, t2b)
        XCTAssertEqual(s2.description, t2.description)
        let s4: Vector<(T, R, C), Int> = mm.flatten()!
        let t4: Vector<(T, R, C), Int> = mmt.flatten()!
        XCTAssertEqual([24], s4.size)
        XCTAssertEqual([24], t4.size)
        XCTAssertEqual(s4.description, t4.description)
        let s5: Matrix<T, (R, C, R), Int>? = mm.flatten()
        XCTAssertNil(s5)
    }

    func testFlatten() {
        let mm: Tensor3<T, R, C, Int> = [[[0, 1, 2, 3], [ 4, 5, 6, 7], [ 8, 9, 10, 11]],
                                         [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]
        testFlatten(mm: mm, x: T.self, y: R.self, z: C.self)
        testFlatten(mm: mm, x: T.self, y: C.self, z: R.self)
        testFlatten(mm: mm, x: R.self, y: C.self, z: T.self)
        testFlatten(mm: mm, x: R.self, y: T.self, z: C.self)
        testFlatten(mm: mm, x: C.self, y: T.self, z: R.self)
        testFlatten(mm: mm, x: C.self, y: R.self, z: T.self)
        // XCTAssertEqual([4, 3, 2], Ct.size)
        let M: Matrix<R, C, Int> = [[0, 1], [2, 3]]
        let Mt: Matrix<C, R, Int> = M.transpose()
        let v1: Vector<(R, C), Int> = M.flatten()!
        let v1b: Vector<(R, C), Int> = Mt.flatten()!
        let v2: Vector<(C, R), Int> = M.flatten()!
        let v2b: Vector<(C, R), Int> = Mt.flatten()!
        XCTAssertEqual(v1, v1b)
        XCTAssertEqual(v2, v2b)
        XCTAssertFalse(v1.description == v2.description)
    }
    func testBroadcast() {
        let m: Matrix<R, C, Int> = [[1, 2, 3], [4, 5, 6]]
        XCTAssertEqual([[1, 2, 3], [4, 5, 6]], m.array as! [[Int]])
        let m2: Matrix<C, R, Int> = [[10, 11], [20, 21], [30, 31]]
        let m3: Matrix<C, C, Int> = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        let m4: Matrix<C, R, Int> = [[1, 2, 3], [4, 5, 6]]
        let m5: Matrix<C, C, Int> = [[1, 2, 3], [4, 5, 6]]
        let t: Vector<C, Int> = [1, 2, 3]
        let t2: Vector<C, Int> = [1, 2]
        XCTAssertEqual([[1, 2, 3], [1, 2, 3]], t.broadcast(to: m.shape)!)
        XCTAssertEqual([[1, 1], [2, 2], [3, 3]], t.broadcast(to: m2.shape)!)
        XCTAssertEqual([[1, 2, 3], [1, 2, 3], [1, 2, 3]], t.broadcast(to: m3.shape)!)
        XCTAssertEqual([[1, 2, 3], [1, 2, 3]], t.broadcast(to: m5.shape)!)
        XCTAssertEqual([[1, 1, 1], [2, 2, 2]], t2.broadcast(to: m5.shape)!)
        XCTAssertNil(t.broadcast(to: m4.shape))
    }

    func testRange() {
        var t: Vector<C, Int> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        var t1 = t[2..<5]
        XCTAssertEqual([3, 4, 5], t1)
        t1[0] += 1
        print(t)
        XCTAssertEqual([4, 4, 5], t1)
        XCTAssertEqual([3, 4, 5], t[2..<5])
        let t2 = t[2...5]
        XCTAssertEqual([3, 4, 5, 6], t2)
        let t3 = t[8...]
        XCTAssertEqual([9, 10], t3)
        let t4 = t[...4]
        XCTAssertEqual([1, 2, 3, 4, 5], t4)
        t[1...4] = [0, 0, 0, 0]
        XCTAssertEqual([1, 2, 3, 4, 5], t4)
        XCTAssertEqual([0, 0, 0, 0], t[1...4])
        print(t)
        t[1...4][2] += 1
        print(t)
        XCTAssertEqual([0, 0, 1, 0], t[1...4])
        XCTAssertEqual([1, 2, 3, 4, 5], t4)
        print(t.array)
    }

    func testNormalDist() {
        var n: NormalDistribution<Double, SystemRandomNumberGenerator> = .init(using: SystemRandomNumberGenerator())
        var sum = 0.0
        var sums = 0.0
        var count = 0
        var sigma = 0
        var sigma2 = 0
        var sigma3 = 0
        for _ in 0..<100000 {
            if let x = n.next() {
                if x.magnitude > 1 {
                    sigma += 1
                    if x.magnitude > 2 {
                        sigma2 += 1
                        if x.magnitude > 3 {
                            sigma3 += 1
                        }
                    }
                }
                sum += x
                sums += x*x
                count += 1
            }
        }
        let avg = sum/Double(count)
        print(avg)
        print((sums/Double(count)).squareRoot())
        print(Double(sigma)/Double(count))
        print(Double(sigma2)/Double(count))
        print(Double(sigma3)/Double(count))
    }

    func testBinaryFloatingPoint() {
        let m: Matrix<R, C, Double> = .random(in: [[1, 2], [3, 4]]..<[[5, 6], [7, 8]])
        let m2: Matrix<R, C, Double> = .init(shape: [3, 4], uniformValuesIn: 5..<10).rounded()
        XCTAssertTrue(m2.compareAll(Scalar<Double>(11), <)!)
        XCTAssertTrue(m2.compareAll(Scalar<Double>(4), >)!)
        XCTAssertTrue(m.compareAll(Matrix<R, C, Double>(arrayLiteral: [1, 2], [3, 4]), >=)!)
        XCTAssertTrue(m.compareAll(Matrix<R, C, Double>(arrayLiteral: [5, 6], [7, 8]), <)!)
        var rng = SystemRandomNumberGenerator()
        let m3: Matrix<R, C, Double> = .init(shape: [100, 100], mean: 10, stddev: 5, using: &rng)
        let sr: Vector<R, Double> = (m3-10).avg()
        let n = sr.norm.value / Double(sr.count)
        XCTAssertEqual(0, n, accuracy: 0.1)
        let sc: Vector<C, Double> = m3.avg()
        XCTAssertEqual(0, (sc-10).sumMagnitude/Double(sc.count), accuracy: 1)
        let stdr: Vector<R, Double> = m3.stddev()
        XCTAssertTrue(stdr.compareAll(Scalar<Double>(7), <)!)
        XCTAssertTrue(stdr.compareAll(Scalar<Double>(3), >)!)

    }

    func testBidirectional() {
        var m: Matrix<R, C, Int> = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
        m.reverse()
        for i in m.indices {
            print(m[i])
            //            m[i].removeLast()
            print(m[i])
            m[i].shuffle()
            //            m[i].reverse()
            print(m[i])
            print(m)
        }
        m.shuffle()
        print(m)
        //        m[0...1] = m[0...0]
        m.removeLast()
        m.removeFirst()
        print(m)
    }

    func testInit2() {
        let v1: Vector<C, Int> = [1, 2]
        let v2: Vector<C, Int> = [1, 2, 3]
        let v3: Vector<C, Int> = .init(concat: v1, v2)
        XCTAssertEqual([1, 2, 1, 2, 3], v3)
        let v22: Matrix<C, C, Int> = v2.outer(rhs: v2, -)
        XCTAssertEqual([[0, -1, -2], [1, 0, -1], [2, 1, 0]], v22)
        let m22: Matrix<C, C, Int> = .init(concat: v22.transpose(), v22)
        XCTAssertEqual([[0, 1, 2], [-1, 0, 1], [-2, -1, 0], [0, -1, -2], [1, 0, -1], [2, 1, 0]], m22)
        let v: Matrix<R, C, Int> = .init(v1, v2)
        let m: Matrix<C, R, Int> = [[4, 5, 6], [7, 8, 9]]
        let mm: Matrix<R, C, Int> = .init(concat: m.transpose(), m.transpose())
        XCTAssertEqual([[4, 7], [5, 8], [6, 9], [4, 7], [5, 8], [6, 9]], mm)
        let t: Tensor3<T, R, C, Int> = .init(v, m.transpose())
        XCTAssertEqual([[[2, 2, 1], [0, 3, 1], [0, 0, 0]], [[9, 4, 0], [7, 5, 0], [8, 6, 0]]], t)
    }

    func testParse() {
        let s: Scalar<Int>? = .init("10")
        XCTAssertEqual(10, s)
        let v: Vector<R, Int>? = .init("[1,2,3]")
        XCTAssertEqual([1, 2, 3], v)
        let m: Matrix<R, C, Int>? = .init("[[1,2],[3,4]]")
        XCTAssertEqual([[2, 1], [4, 3]], m)
    }

    func testBinaryInteger() {
        var m: Matrix<R, C, UInt> = Vector<R, UInt>(arrayLiteral: 1, 2, 3).outer(rhs: Vector<C, UInt>(arrayLiteral: 1, 2, 3), +)
        m <<= 10
        m += 1
        print(m)
        XCTAssertTrue(m.nonzeroBitCount > 0)
        while m != .zero.broadcast(to: m.shape)! {
            let i = m.trailingZeroBitCount
            m >>= i+1
        }
        XCTAssertEqual(64, m.leadingZeroBitCount)
    }

    func testVector2() throws {
        let v: Vector<R, Double> = .init(shape: [3]) { i in Double(i[0])}
        let v2: Vector<R, Double> = [1, 2, 3]
        for var i in v {
            i += 1
        }
        XCTAssertEqual([0, 1, 2], v)
        let z: Vector<R, Double> = .zeros(like: v)
        XCTAssertEqual([0, 0, 0], z)
        let v3 = v + v2
        XCTAssertEqual([1, 3, 5], v3)
        let v4 = v + 1.0
        XCTAssertEqual([1, 2, 3], v4)
        XCTAssertEqual(6.0, v4.elementSum)
    }
    func testSum2() {
        var ml: Matrix<R, C, Double> = [[1, 2], [3, 4], [5, 6]]
        XCTAssertEqual(21.0, ml.elementSum)
        XCTAssertEqual([9.0, 12.0] as Vector<C, Double>, ml.sum())
        XCTAssertEqual([3.0, 7.0, 11.0] as Vector<R, Double>, ml.sum())
    }
    func testMatrix2() throws {
        var ms: Matrix<R, C, Double> = [[1, 2], [3, 4]]
        ms += ms.Transposed.cast()
        XCTAssertEqual([[2, 5], [5, 8]], ms)
        var ml: Matrix<R, C, Double> = [[0, 1], [3, 4], [5, 6]]
        ml.Transposed[0][0] += 1
        XCTAssertEqual([[1, 1], [3, 4], [5, 6]], ml)
        ml.Transposed[0] = [2, 3, 4]
        XCTAssertEqual([[2, 1], [3, 4], [4, 6]], ml)
        ml.Transposed[0] = [0, 3, 5]
        XCTAssertEqual([[1, 2], [4, 5], [6, 7]], ml+1.0)
        for var v in ml.transpose() as Matrix<C, R, Double> {
            v[0] += 1
        }
        XCTAssertEqual( [[0, 1], [3, 4], [5, 6]], ml)
        for v in (ml.transpose() as Matrix<C, R, Double>).indices {
            ml[0][v] += 1
        }
        XCTAssertEqual( [[1, 2], [3, 4], [5, 6]], ml)
        let vr: Vector<R, Double> = [10, 12, 13]
        XCTAssertEqual([[11, 12], [15, 16], [18, 19]], ml+vr)
        let z: Matrix<R, C, Double> = .zeros(shape: 3, 2)
        XCTAssertEqual([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], z)
        let eye: Matrix<R, C, Int> = .eye(shape: 4, 3)
        XCTAssertEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], eye)
        let eye2: Tensor3<T, R, C, Int> = .eye(shape: 2, 2, 3)
        XCTAssertEqual([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]], eye2.transpose() as Tensor3<C, T, R, Int>)
        let m: Matrix<R, C, Double> = .init(shape: [3, 4]) {Double($0.x*4+$0.y)}
        XCTAssertEqual([[0.0, 4.0, 8.0, 12.0], [1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0]], m)
        let m2: Matrix<R, C, Int> = .init(fromArray: [[1+1, 2+2], [3, 4], [1]])
        XCTAssertEqual([[2, 4], [3, 4], [1, 0]], m2)
        let stringmatrix: Matrix<R, C, Character> = .init(fromArray: "ab cd ef".split(separator: " "))
        let stringmatrix2: Matrix<R, C, String> = [["a", "b"], ["c"]]
        XCTAssertEqual([["a", "b"], ["c", "d"], ["e", "f"]], stringmatrix)
        XCTAssertEqual([["a", "b"], ["c", ""]], stringmatrix2)
        let stringvector: Vector<C, String> = (stringmatrix.mapTensor({$0.description}) as Matrix<R, C, String>).reduceTensor(zero: "", op: +)!
        XCTAssertEqual(["ace", "bdf"], stringvector)
        let seq: AnySequence<StrideTo<Int>> = .init {
            var i = 4
            return AnyIterator<StrideTo<Int>> {
                defer {
                    i-=1
                }
                return i>0 ? Swift.stride(from: i, to: 4, by: 1) : nil
            }}
        let v: Matrix<R, R, Int> = .init(fromArray: seq)
        XCTAssertEqual([[0, 0, 0], [3, 0, 0], [2, 3, 0], [1, 2, 3]], v)
        var arraymatrix: Matrix<R, C, [Int]> = .init(fromArray: [[[1], [2, 3]], [[4, 5, 6], [7, 8, 9, 10], [1]]])
        XCTAssertEqual([2, 3], arraymatrix.size)
        arraymatrix[0][2].value.append(contentsOf: arraymatrix[1][2].value)
        XCTAssertEqual([[[1], [2, 3], [1]], [[4, 5, 6], [7, 8, 9, 10], [1]]], arraymatrix.array as! [[[Int]]])
        let arrayvector: Vector<C, [Int]> = arraymatrix.reduceTensor(zero: [], op: +)!
        XCTAssertEqual([[1, 4, 5, 6], [2, 3, 7, 8, 9, 10], [1, 1]], arrayvector.array as? [[Int]])
    }
    //    func testTensor() throws {
    //        let m: Tensor3<T, R, C, Double> = .init(tubes: 2, rows: 3, cols: 4, supplier: {Double($0*12+$1*4+$2)})
    //        let m1: Tensor3<T, R, C, Double> = [[[1, 2], [3, 4]], [[5, 6], [7, 8, 9], []], []]
    //        let mt: Tensor3<C, T, R, Double> = m.transpose()
    //        print(mt)
    //        for var v in mt {
    //            print(v)
    //            v[0, 0] += 1
    //            print(v)
    //        }
    //        print(mt)
    //        print(m1)
    //        print(m1.shape)
    //        print(m)
    //        var mi: Tensor3<T, R, C, Double> = .init(integerLiteral: 10)
    //        print(mi)
    //        var mm: Tensor3<T, C, R, Double> = m.transpose()
    //        print(mm)
    //        print(m.shape)
    //        print(mm.shape)
    //        mm[0, 1, 2] = -1
    //        print(mm)
    //        print(m)
    //
    //    }

}
