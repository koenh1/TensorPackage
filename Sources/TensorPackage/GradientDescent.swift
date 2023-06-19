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

extension DifferentiableTensorModel where ValueType.ValueType: FloatingPoint {
    
    public mutating func rprop(maxIterations:Int,stepSize:ValueType.ValueType,stepSizeRange:Range<ValueType.ValueType>,beta:ValueType.ValueType, tolerance: ValueType.ValueType, gradientEvals:inout Int) -> (steps: Int, relativeReduction: ValueType.ValueType)?{
        guard stepSize < 0 else { fatalError() }
        var view = view
        var stepSize = stepSize
        var gradientSquares = TensorGradients<Self,ValueType.ValueType>(forTensors: self)
        var last: ValueType.ValueType = .zero
        var gradientValue = objective()
        for iter in 0..<maxIterations {
            var gradients = TensorGradients<Self,ValueType.ValueType>(of: gradientValue, forTensors: self)
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
    public mutating func findStepSize(alpha: ValueType.ValueType, c1: ValueType.ValueType) -> ValueType.ValueType {
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
    public func increaseStepSize(stepSize:inout ValueType.ValueType) {
        var view = view
        let gradientValue = objective()
        var best = gradientValue.value
        let gradients: TensorGradients = .init(of: gradientValue, forTensors: self)
        var factor: ValueType.ValueType = 2
        var total: ValueType.ValueType = factor
        _ = gradients.update(stepSize: stepSize*factor, views: &view)
        var v = view.objective().value
        while v < best {
            best = v
            factor += 2
            _ = gradients.update(stepSize: stepSize*factor, views: &view)
            v = view.objective().value
            if v < best {
                total += factor
            } else {
                _ = gradients.update(stepSize: -stepSize*factor, views: &view)
            }
        }
        print("\(self) \(view.objective().value)")
        stepSize *= total
    }
    public func decreaseStepSize(stepSize:inout ValueType.ValueType) {
        var view = view
        let gradientValue = objective()
        let gradients: TensorGradients = .init(of: gradientValue, forTensors: self)
        var factor: ValueType.ValueType = 1/2
        _ = gradients.update(stepSize: stepSize, views: &view)
        var best = Swift.min(gradientValue.value, view.objective().value)
        _ = gradients.update(stepSize: -stepSize*factor, views: &view)
        var v = view.objective().value
        while v < best {
            best = v
            factor /= 2
            _ = gradients.update(stepSize: -stepSize*factor, views: &view)
            v = view.objective().value
            if v > best {
                _ = gradients.update(stepSize: stepSize*factor, views: &view)
            }
        }
        print("\(self) \(view.objective().value)")
        stepSize *= factor
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
