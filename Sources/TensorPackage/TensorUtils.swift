//
//  File.swift
//  
//
//  Created by Koen Hendrikx on 11/06/2023.
//

import Foundation
import SwiftUI
import Charts

@available(macOS 13.0, *)
extension Tensor:CustomReflectable where Types: NonEmptyList {
    public var customMirror: Mirror {
        array.customMirror
    }
}

extension Differentiable: CustomReflectable {
    public var customMirror: Mirror {
        .init(self, children: [(label:String?,value:Any)].init(arrayLiteral:("value",value),("header",header)))
    }
}

public struct SimplePlot: View {
    var xrange:ClosedRange<Double>
    @State var selected:ClosedRange<Double>? = nil
    var function:(Double)->Double
    public init(xrange: ClosedRange<Double>, function: @escaping (Double) -> Double) {
        self.xrange = xrange
        self.function = function
    }
    public var body: some View {
        let points:[(Int,Double)] = Array(stride(from:xrange.lowerBound,through: xrange.upperBound,by:(xrange.upperBound-xrange.lowerBound)/100.0).enumerated())
        return Chart{
            ForEach(points,id:\.0) {
                LineMark(
                    x: .value("x", $0.1),
                    y: .value("y", function($0.1))
                )
            }
            if let selected = selected {
                RectangleMark(
                    xStart: .value("Selection Start", selected.lowerBound),
                    xEnd: .value("Selection End", selected.upperBound)
                )
                .foregroundStyle(.white.opacity(0.2))
            }
        }.chartOverlay { proxy in
            GeometryReader { nthGeoItem in
                Rectangle().fill(.clear).contentShape(Rectangle())
                    .gesture(DragGesture()
                        .onChanged { value in
                            // Find the x-coordinates in the chart’s plot area.
                            let xStart = value.startLocation.x - nthGeoItem[proxy.plotAreaFrame].origin.x
                            let xCurrent = value.location.x - nthGeoItem[proxy.plotAreaFrame].origin.x
                            // Find the date values at the x-coordinates.
                            if let start: Double = proxy.value(atX: xStart),
                               let current: Double = proxy.value(atX: xCurrent) {
                                if start < current {
                                    selected = start...current
                                }
                            }
                        }
                        .onEnded { _ in selected = nil } // Clear the state on gesture end.
                    )
            }
        }.frame(minWidth: 200,idealWidth: 400,minHeight: 100, idealHeight: 200,alignment: .top)
    }
}

public struct ParameterPlot: View {
    var xrange:ClosedRange<Double>
    var trange:ClosedRange<Double>
    var function:(Double,Double)->Double
    @State var t: Double
    public init(xrange: ClosedRange<Double>,trange:ClosedRange<Double>, function: @escaping (Double,Double) -> Double) {
        self.xrange = xrange
        self.trange = trange
        self.function = function
        t = trange.lowerBound
    }
    public var body: some View {
        let points:[(Int,Double)] = Array(stride(from:xrange.lowerBound,through: xrange.upperBound,by:(xrange.upperBound-xrange.lowerBound)/100.0).enumerated())
        return VStack {
            Chart(points,id:\.0) {
                LineMark(
                    x: .value("x", $0.1),
                    y: .value("y", function($0.1,t))
                )
            }
            Slider(value: $t,in: trange,label: {Text("t")}, minimumValueLabel: {Text("\(trange.lowerBound)")}, maximumValueLabel: {Text("\(trange.upperBound)")})
            Text("\(t)")
        }.frame(minWidth: 200,idealWidth: 400,minHeight: 100, idealHeight: 200,alignment: .top)
    }
}

public struct GridPlot: View {
    var xrange:ClosedRange<Double>
    var yrange:ClosedRange<Double>
    var function:(Double,Double)->Double
    public init(xrange: ClosedRange<Double>,yrange:ClosedRange<Double>, function: @escaping (Double,Double) -> Double) {
        self.xrange = xrange
        self.yrange = yrange
        self.function = function
    }
    public var body: some View {
        let xs = stride(from:xrange.lowerBound,through: xrange.upperBound,by:(xrange.upperBound-xrange.lowerBound)/5.0)
        let points:[(Int,(Double,Double))] = Array(zip(xs,xs.dropFirst()).enumerated())
        let ys = stride(from:yrange.lowerBound,through: yrange.upperBound,by:(yrange.upperBound-yrange.lowerBound)/5.0)
        return Chart(points,id:\.0) { xpoint in
            let column:[(Int,(Double,Double))] = Array(zip(ys,ys.dropFirst()).enumerated())
            ForEach(column,id:\.0) { ypoint in
                RectangleMark(xStart: .value("x",xpoint.1.0), xEnd: .value("y",ypoint.1.0)).opacity(0.2).symbol(symbol: {Text(".").foregroundColor(Color.white)})
//                RectangleMark(x: .value("x",xpoint.1.1),y: .value("y", ypoint.1.1),width: .fixed(xpoint.1.1-xpoint.1.0),height: .fixed(ypoint.1.1-ypoint.1.0))
            }
        }.frame(minWidth: 300,idealWidth: 400,minHeight: 200, idealHeight: 200,alignment: .top)
    }
}
