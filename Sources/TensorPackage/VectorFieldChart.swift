//
// Copyright © 2022 Swift Charts Examples.
// Open Source - MIT License

import SwiftUI
import Charts

struct Point: Hashable, Identifiable {
    let id = UUID()
    let x: Double
    let y: Double
    let dx: Double
    let dy: Double
    func angle() -> Double {
        let t = atan2(dy, dx)
        return t
    }
    func length() -> Double {
        hypot(dx, dy)
    }
    var description: String {
        "(\(x),\(y)):(\(dx),\(dy))"
    }
}

public struct GradientField<Model: DifferentiableTensorModel>: View where Model.ValueType.ValueType==Double {
    let model: Model
    let xrange: ClosedRange<Model.ValueType.ValueType>
    let yrange: ClosedRange<Model.ValueType.ValueType>
    let x: WritableKeyPath<Model, Model.ValueType>
    let y: WritableKeyPath<Model, Model.ValueType>
    let xv: WritableKeyPath<Model.ViewModel, Model.ViewModel.ValueType>
    let yv: WritableKeyPath<Model.ViewModel, Model.ViewModel.ValueType>
    public init(model: Model, x: WritableKeyPath<Model, Model.ValueType>, xv: WritableKeyPath<Model.ViewModel, Model.ViewModel.ValueType>, y: WritableKeyPath<Model, Model.ValueType>, yv: WritableKeyPath<Model.ViewModel, Model.ViewModel.ValueType>, xrange: ClosedRange<Model.ValueType.ValueType>, yrange: ClosedRange<Model.ValueType.ValueType>) {
        self.model = model
        self.x = x
        self.y = y
        self.xv = xv
        self.yv = yv
        self.xrange = xrange
        self.yrange = yrange
    }
    public var body: some View {
        var v = model.view
        let x = model[keyPath: x]
        let y = model[keyPath: y]
        defer {
            v[keyPath: xv].value = x.value
            v[keyPath: yv].value = y.value
        }
        let eval: (Model.ValueType.ValueType, Model.ValueType.ValueType) -> (Model.ValueType.ValueType, Model.ValueType.ValueType) = {
            v[keyPath: xv].value = $0
            v[keyPath: yv].value = $1
            let g = model.objective().gradients(for: [x, y])
            return (g.result[0], g.result[1])
        }
        return VectorField(numRows: 50, numCols: 50, xrange: xrange, yrange: yrange, function: eval)
    }
}

public struct DimensionGrid: View {
    let dimensionCount: Int
    @State var selectedX: Int = -1
    @State var selectedY: Int = -1
    @State var hoverX: Int = -1
    @State var hoverY: Int = -1
    let builder: (Int, Int) -> any View
    public init(dimensionCount: Int, builder:@escaping (Int, Int) -> any View = {_, _ in Rectangle().foregroundColor(.blue)}) {
        self.dimensionCount = dimensionCount
        self.builder = builder
    }
    func makeView(_ row: Int, _ col: Int) -> any View {
        row == col ? Rectangle().opacity(0) : builder(row, col)
    }
    public var body: some View {
        Grid(horizontalSpacing: 0, verticalSpacing: 0) {
            ForEach(0..<dimensionCount, id: \.self) { row in
                GridRow {
                    Text("\(row)").opacity( row == hoverY ? 1 : 0.5 )
                    ForEach(0..<dimensionCount, id: \.self) { col in
                        AnyView(makeView(row, col))
                            .border(col == selectedX && row == selectedY ? Color.white : Color.black, width: 2)
                            .onTapGesture {
                            selectedX = col
                            selectedY = row
                        }.onHover {
                            if $0 {
                                hoverX = col
                                hoverY = row
                            } else {
                                hoverX = -1
                                hoverY = -1
                            }
                        }

                    }
                }
            }
            GridRow {
                Text("")
                ForEach(0..<dimensionCount, id: \.self) {
                    Text("\($0)").opacity( $0 == hoverX ? 1 : 0.5 )
                }
            }
        }
    }
}

public struct VectorField: View {
    public init(numRows: Int, numCols: Int, xrange: ClosedRange<Double>, yrange: ClosedRange<Double>, function:@escaping (Double, Double) -> (Double, Double)) {
        self.numRows = numRows
        self.numCols = numCols
        self.function = function
        self.xrange = xrange
        self.yrange = yrange
        self.dx = (xrange.upperBound-xrange.lowerBound)/Double(numCols)
        self.dy = (yrange.upperBound-yrange.lowerBound)/Double(numRows)
        let xs = stride(from: xrange.lowerBound+dx/2, through: xrange.upperBound-dx/2, by: dx)
        let ys = stride(from: yrange.lowerBound+dy/2, through: yrange.upperBound-dy/2, by: dy)
        for y in ys {
            for x in xs {
                let r = function(x, y)
                let pt: Point = .init(x: x, y: y, dx: r.0, dy: r.1)
                points.append(pt)
                maxLength = Swift.max(maxLength, pt.length())
                minLength = Swift.min(minLength, pt.length())
            }
        }
    }
    let xrange: ClosedRange<Double>
    let yrange: ClosedRange<Double>
    let dx: Double
    let dy: Double
    let function: (Double, Double) -> (Double, Double)
    let numRows: Int
    let numCols: Int
    var minLength: Double = .infinity
    var maxLength: Double = .zero
    var points = [Point]()

    private var opacity = 0.7

    var gradient: LinearGradient {
        LinearGradient(colors: Array(stride(from: minLength, to: maxLength, by: (maxLength-minLength)/10).map {Color(hue: $0/maxLength, saturation: 1, brightness: 1)}), startPoint: .leading, endPoint: .trailing)
    }

    public var body: some View {
        VStack {
            Spacer()
            GeometryReader { geo in
                let sx = geo.size.width/CGFloat(2*numCols)
                let sy = geo.size.height/CGFloat(2*numRows)
                Chart(points) { point in
                        Plot {
                            PointMark(x: .value("x", point.x),
                                      y: .value("y", point.y))
                            .symbol(Arrow2(angle: CGFloat(point.angle()), dx: sx, dy: sy))
                            .symbolSize(pow(Swift.min(sx, sy), 2))
                            .foregroundStyle(Color(hue: point.length()/maxLength, saturation: 1, brightness: 1))
                            .opacity(opacity)
                        }
                        .accessibilityLabel("Point: \(point.description)")
                    }
                .chartXScale(domain: xrange.lowerBound...xrange.upperBound)
                .chartYScale(domain: yrange.lowerBound...yrange.upperBound)
                .chartYAxis(.automatic)
                .chartXAxis(.automatic)
//                .aspectRatio(contentMode: .fit)
            }.frame(minWidth: 400, idealWidth: 600, minHeight: 300, idealHeight: 400, alignment: .top)
            if maxLength > minLength {
                Rectangle().fill(gradient).frame(height: 12)
                HStack {
                    Text(String(format: "%5.2g", minLength))
                    Spacer()
                    Text(String(format: "%5.2g", maxLength))
                }
            }
        }
    }

}

struct Arrow: ChartSymbolShape {
    let angle: CGFloat
    let size: CGFloat

    func path(in rect: CGRect) -> Path {
        let w = rect.width * size * 0.05 + 0.6
        var path = Path()
        path.move(to: CGPoint(x: 0, y: 1))
        path.addLine(to: CGPoint(x: -0.2, y: -0.5))
        path.addLine(to: CGPoint(x: 0.2, y: -0.5))
        path.closeSubpath()
        return path.applying(.init(rotationAngle: angle))
            .applying(.init(scaleX: w, y: w))
            .applying(.init(translationX: rect.midX, y: rect.midY))
    }

    var perceptualUnitRect: CGRect {
        return CGRect(x: 0, y: 0, width: 1, height: 1)
    }
}

struct Arrow2: ChartSymbolShape {
    let angle: CGFloat
    let dx: CGFloat
    let dy: CGFloat
    func path(in rect: CGRect) -> Path {
        let w = Swift.min(rect.width, rect.height)
        var path = Path()
        path.move(to: CGPoint(x: -w, y: 0))
        path.addLine(to: CGPoint(x: w, y: 0))
        return path.applying(.init(rotationAngle: -angle))
            .applying(.init(translationX: rect.midX, y: rect.midY))
            .strokedPath(StrokeStyle())
    }

    var perceptualUnitRect: CGRect {
        return CGRect(x: 0, y: 0, width: 1, height: 1)
    }
}

extension AnyView: CustomPlaygroundDisplayConvertible {
    @MainActor
    public var playgroundDescription: Any {
        let renderer = ImageRenderer(content: self)
        return renderer.cgImage!
    }

}
