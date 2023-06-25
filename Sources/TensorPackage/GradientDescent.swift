//
//  GradientDescent.swift
//  Tensor
//
//  Created by Koen Hendrikx on 08/06/2023.
//

import Foundation

public protocol TensorModel {
    associatedtype ValueType: DifferentiableValue
    func objective() -> ValueType
}
public protocol DifferentiableTensorModel: TensorModel where ValueType: DifferentiableProtocol {
    associatedtype ViewModel: TensorModel where ViewModel.ValueType == Undifferentiable<ValueType.ValueType>
    var view: ViewModel { get }
}

extension TensorModel {
    public var state: ModelState<Self> {
        get {
            .init(self)
        }
        set {
            newValue.update(&self)
        }
    }
}

public struct ModelState<Model>: CustomStringConvertible {
    var result: [String: Any]
    static func visitParameters(parameters: Any, key: String, _ f: (String, inout any TensorProtocol) -> Void) {
        let m = Mirror(reflecting: parameters)
        if var mm = parameters as? any TensorProtocol {
            f("", &mm)
        }
        for mc in m.children.enumerated() {
            let k = "\(key)\(mc.element.label ?? "_\(mc.offset)")"
            if var mm = mc.element.value as? any TensorProtocol {
                f(k, &mm)
            } else {
                visitParameters(parameters: mc.element.value, key: "\(k).", f)
            }
        }
    }
    func update(_ model:inout Model) {
        func gather<S: TensorProtocol>(_ key: String, _ s: inout S) {
            if let k = result[key] as? Tensor<S.Types, S.IndexType, S.T> {
                print("\(key):\(s)\n\(k)")
                s.buffer.transfer(range: 0..<s.elementCount, from: k.buffer)
                print("\(s)")
            } else {
                fatalError()
            }
        }
        Self.visitParameters(parameters: model, key: "") { n, s in
            gather(n, &s)
        }
    }
    init(_ model: Model) {
        result = [:]
        func gather<S: TensorProtocol>(_ key: String, _ s: S) {
            result[key] = S(buffer: s.buffer.copy, shape: s.shape)
        }
        Self.visitParameters(parameters: model, key: "") { n, s in
            gather(n, s)
        }
    }
    public var description: String {
        "\(result)"
    }
}

extension DifferentiableTensorModel where ValueType.ValueType: FloatingPoint {

    public mutating func rprop(maxIterations: Int, stepSize: ValueType.ValueType, stepSizeRange: Range<ValueType.ValueType>, beta: ValueType.ValueType, tolerance: ValueType.ValueType, gradientEvals:inout Int) -> (steps: Int, relativeReduction: ValueType.ValueType)? {
        guard stepSize < 0 else { fatalError() }
        var view = view
        var stepSize = stepSize
        var gradientSquares = TensorGradients<Self, ValueType.ValueType>(forTensors: self)
        var last: ValueType.ValueType = .zero
        var gradientValue = objective()
        for iter in 0..<maxIterations {
            var gradients = TensorGradients<Self, ValueType.ValueType>(of: gradientValue, forTensors: self)
            gradientSquares.updateGradients(other: gradients, f: {$0 = beta*$0+(1-beta)*$1*$1})
            gradients.updateGradients(other: gradientSquares, f: {$0 = $0/$1.squareRoot()})
            let normSquared = gradients.update(stepSize: stepSize, views: &view)
            gradientEvals += 1
            let previous = gradientValue.value
            gradientValue = objective()
            last = (gradientValue.value-previous)/normSquared
            if normSquared.magnitude <= tolerance*ValueType.ValueType(gradientSquares.count) {
                return (iter, last)
            }
        }
        return (maxIterations, last)
    }

    public mutating func gradientDescent(maxIterations: Int, stepSize: ValueType.ValueType, tolerance: ValueType.ValueType, gradientEvals:inout Int) -> (steps: Int, relativeReduction: ValueType.ValueType)? {
        guard stepSize < 0 else { fatalError() }
        var gradientValue = objective()
        let firstValue = gradientValue.value
        var view = view
        var last: ValueType.ValueType = .zero
        for iter in 0..<maxIterations {
            let gradients: TensorGradients = .init(of: gradientValue, forTensors: self)
            let normSquared = gradients.update(stepSize: stepSize, views: &view)
            gradientEvals += 1
            let previous = gradientValue.value
            gradientValue = objective()
            last = (gradientValue.value-previous)/normSquared
            if normSquared.magnitude <= tolerance*ValueType.ValueType(gradients.count) {
                return (iter, last)
            }
            if gradientValue.value > previous {
                _ = gradients.update(stepSize: -stepSize, views: &view)
                if gradientValue.value > firstValue {
                    return nil
                } else {
                    break
                }
            }
        }
        return (maxIterations, last)
    }
    public func findStepSize(alpha: ValueType.ValueType, c1: ValueType.ValueType) -> ValueType.ValueType {
        guard alpha > .zero && alpha < 1 else { fatalError() }
        let gradientValue = objective()
        let gradients: TensorGradients = .init(of: gradientValue, forTensors: self)
        var stepSize: ValueType.ValueType = -1
        var view = view
        var d = gradients.update(stepSize: stepSize, views: &view)
        while view.objective().value > gradientValue.value + c1*d {
            let newStepSize = stepSize * alpha
            d = gradients.update(stepSize: -stepSize, views: &view)
            d = gradients.update(stepSize: newStepSize, views: &view)
            if d == .zero {
                break
            }
            stepSize = newStepSize
//            print(view.objective().value)
        }
        d = gradients.update(stepSize: -stepSize, views: &view)
        return stepSize
    }
    public mutating func gradientDescent(maxIterations: Int, stepSize: inout ValueType.ValueType, stepReductionFactor: ValueType.ValueType, stepIncrementFactor: ValueType.ValueType, c1: ValueType.ValueType, tolerance: ValueType.ValueType) -> Int {
        guard stepSize < 0 else { fatalError() }
        var gradientValue = objective()
        var previousValue = gradientValue.value
        var view = view
        let reportSteps = maxIterations/100
        var gradientEvals = 1
        var evals = 0
        var averageFunctionChange: ValueType.ValueType = 0
        defer {
            print("\(gradientValue)  #evals=\(evals) #gradient evals=\(gradientEvals) step=\(stepSize)")
        }

        for iter in 0..<maxIterations {
//            print(self)
            let gradients: TensorGradients = .init(of: gradientValue, forTensors: self)
//            print(gradients)
            let normSquared = gradients.update(stepSize: stepSize, views: &view)
            if normSquared <= tolerance*ValueType.ValueType(gradients.count) {
                print("stopped when step \(stepSize) resulted in \(normSquared)")
                return iter
            }
//            print(self)
            var newGradientValue = objective()
            gradientEvals += 1
            var newValue = newGradientValue.value
            averageFunctionChange = averageFunctionChange*stepIncrementFactor + (newValue-previousValue).magnitude
            if averageFunctionChange < tolerance {
                print("stopped when step \(stepSize) resulted in \(newValue-previousValue) function change")
                return iter
            }
            if newValue > previousValue {
                while newValue > previousValue {
                    let newStepSize = stepSize*stepReductionFactor
                    let n = gradients.update(stepSize: -stepSize+newStepSize, views: &view)
                    newValue = view.objective().value
                    evals += 1
                    stepSize = newStepSize
                    if n <= tolerance*ValueType.ValueType(gradients.count) {
                        break
                    }
                }
                newGradientValue = objective()
                gradientEvals += 1
                newValue = newGradientValue.value
            } else if iter & 7 == 0 {
                let expectedDifference = normSquared.squareRoot()
                let actualDifference = previousValue - newValue
//                print("expected \(expectedDifference) actual \(actualDifference)")
                if actualDifference < c1*expectedDifference {
                    var actualStepSize = stepSize
                    let increment = stepIncrementFactor*stepSize
                    while true {
                        _ = gradients.update(stepSize: increment, views: &view)
                        let nv = view.objective().value
                        evals += 1
                        if nv >= newValue {
                            _ = gradients.update(stepSize: -increment, views: &view)
                            break
                        }
                        newValue = nv
                        actualStepSize += increment
                    }
                    if actualStepSize != stepSize {
                        stepSize = actualStepSize
                        newGradientValue = objective()
                        gradientEvals += 1
                        newValue = newGradientValue.value
                    }
                } else {
                    stepSize += stepSize*stepIncrementFactor
                }
            }
            previousValue = newValue
            gradientValue = newGradientValue
            if iter % reportSteps == 0 {
                print("\(iter) \(newValue) #evals=\(evals) #gradient evals=\(gradientEvals) step=\(stepSize)")
            }
        }
        return maxIterations
    }
}

class LBFGS<Model: DifferentiableTensorModel> where Model.ValueType.ValueType: FloatingPoint {
    let maxHistorySize: Int
    let maxIterations: Int
    var model: Model
    init(model: Model, maxHistorySize: Int, maxIterations: Int) {
        self.model = model
        self.maxHistorySize = maxHistorySize
        self.maxIterations = maxIterations
    }
    var inputDiffs: [TensorGradients<Model, Model.ValueType.ValueType>] = []
    var derivDiffs: [TensorGradients<Model, Model.ValueType.ValueType>] = []
    func getInitialInverseHessianDiagonal(count: Int) -> TensorGradients<Model, Model.ValueType.ValueType> {
        var result: TensorGradients<Model, Model.ValueType.ValueType> = .init(forTensors: model)
        var scale: Model.ValueType.ValueType = 1
        if !inputDiffs.isEmpty {
            let lastDerivativeDifference = derivDiffs.first!
            let lastInputDifference = inputDiffs.first!
            let prod: Model.ValueType.ValueType = lastDerivativeDifference.inner(lastInputDifference)
            let norm: Model.ValueType.ValueType = lastDerivativeDifference.inner(lastDerivativeDifference)
            scale = prod / norm
        }
        result.updateGradients(op: {_, i in i = scale})
        return result
    }
    func implicitMultiply(initialInverseHessianDiagonal: TensorGradients<Model, Model.ValueType.ValueType>, derivative: TensorGradients<Model, Model.ValueType.ValueType>) -> TensorGradients<Model, Model.ValueType.ValueType> {
        var rho: [Model.ValueType.ValueType] = .init(repeating: .zero, count: inputDiffs.count)
        var alpha: [Model.ValueType.ValueType] = .init(repeating: .zero, count: inputDiffs.count)
        var right = derivative
        for i in rho.indices.reversed() {
            let inputDifference = inputDiffs[i]
            let derivativeDifference = derivDiffs[i]
            rho[i] = inputDifference.inner(derivativeDifference)
            if rho[i] == .zero {
                fatalError("Curvature problem.")
            }
            alpha[i] = inputDifference.inner(right) / rho[i]
            right.updateGradients(other: derivativeDifference, f: { $0 -= $1*alpha[i]})
        }
        right.updateGradients(other: initialInverseHessianDiagonal, f: {$0 *= $1})
        for i in rho.indices {
            let inputDifference = inputDiffs[i]
            let derivativeDifference = derivDiffs[i]
            let beta = derivativeDifference.inner(right) / rho[i]
            right.updateGradients(other: inputDifference, f: {$0 += (alpha[i] - beta)*$1})
        }
        return right
    }
    func getSearchDirection(gradient: TensorGradients<Model, Model.ValueType.ValueType>) -> TensorGradients<Model, Model.ValueType.ValueType> {
        let initialInverseHessianDiagonal = getInitialInverseHessianDiagonal(count: gradient.count)
        return implicitMultiply(initialInverseHessianDiagonal: initialInverseHessianDiagonal, derivative: gradient)
    }
    func minimize() {
        var cur = model.objective()
        let gradient: TensorGradients<Model, Model.ValueType.ValueType> = .init(of: cur, forTensors: model)
        for iter in 0..<maxIterations {
            let dir = getSearchDirection(gradient: gradient)
        }
    }
    func lineSearch(current: Model.ValueType, gradient: TensorGradients<Model, Model.ValueType.ValueType>, direction: TensorGradients<Model, Model.ValueType.ValueType>, maxIterations: Int) -> Bool {
        var stepSize: Model.ValueType.ValueType = 1
//        var guess =
        for iter in 0..<maxIterations {
//            let guess =
        }
        fatalError()
    }
}
extension DifferentiableTensorModel where ValueType.ValueType: FloatingPoint {
    func lbfgs(maxHistorySize: Int, maxIterations: Int, stepSize: ValueType.ValueType, tolerance: ValueType.ValueType, gradientEvals:inout Int) -> (steps: Int, relativeReduction: ValueType.ValueType)? {
        fatalError()
    }
}
