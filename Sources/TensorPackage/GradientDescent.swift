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
    public mutating func gradientDescent(maxIterations:Int,initialStepSize: ValueType.ValueType,stepReductionFactor:ValueType.ValueType,stepIncrementFactor: ValueType.ValueType,tolerance: ValueType.ValueType) -> Int {
        var stepSize = initialStepSize
        var gradientValue = objective()
        var previousValue = gradientValue.value
        var view = view
        let reportSteps = maxIterations/10
        for iter in 0..<maxIterations {
//            print(self)
            let gradients: TensorGradients = .init(of: gradientValue, forTensors: self)
//            print(gradients)
            let normSquared = gradients.update(stepSize: stepSize, views: &view)
            if normSquared <= tolerance*ValueType.ValueType(gradients.count) {
                return iter
            }
//            print(self)
            var newGradientValue = objective()
            var newValue = newGradientValue.value
            if (newValue-previousValue).magnitude < tolerance {
                return iter
            }
            if (newValue - previousValue > .zero) != (stepSize > .zero) {
                while (newValue - previousValue > .zero) != (stepSize > .zero) {
                    let newStepSize = stepSize*stepReductionFactor
                    let n = gradients.update(stepSize: -stepSize+newStepSize, views: &view)
                    newValue = view.objective().value
                    stepSize = newStepSize
                    if n <= tolerance*ValueType.ValueType(gradients.count) {
                        break
                    }
                }
                newGradientValue = objective()
                newValue = newGradientValue.value
            } else {
                stepSize *= stepIncrementFactor
            }
            previousValue = newValue
            gradientValue = newGradientValue
            if iter % reportSteps == 0 {
                print("\(iter) \(newValue)")
            }
        }
        return maxIterations
    }
}

