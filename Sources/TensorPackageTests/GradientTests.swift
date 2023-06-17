//
//  GradientTests.swift
//  
//
//  Created by Koen Hendrikx on 16/06/2023.
//

import XCTest
import TensorPackage
final class GradientTests: XCTestCase {

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
