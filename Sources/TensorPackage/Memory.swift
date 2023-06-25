//
//  Memory.swift
//  Tensor
//
//  Created by Koen Hendrikx on 29/04/2023.
//

import Foundation

final class MemoryPtr<Element> {
    let address: UnsafeMutableRawBufferPointer
    let owned: Bool
    init(shared address: UnsafeMutableRawBufferPointer) {
        self.address = address
        owned = false
    }
    init(capacity: Int, initialValue: Element) {
        self.address = .allocate(byteCount: capacity*MemoryLayout<Element>.stride, alignment: MemoryLayout<Element>.alignment)
        owned=true
        fill(initialValue)
    }
    var count: Int {
        address.count / MemoryLayout<Element>.stride
    }
    init(capacity: Int) {
        self.address = .allocate(byteCount: capacity*MemoryLayout<Element>.stride, alignment: MemoryLayout<Element>.alignment)
        owned=true
    }
    func fill(_ value: Element) {
        self.address.initializeMemory(as: Element.self, repeating: value)
    }
    deinit {
        if owned {
            _ = address.withMemoryRebound(to: Element.self) {$0.deinitialize()}
            address.deallocate()
        }
    }
    func copy(range: Range<Int>) -> MemoryPtr<Element> {
        let r: MemoryPtr<Element> = .init(capacity: range.count)
        let buf = address.assumingMemoryBound(to: Element.self)[range]
        _ = r.address.initializeMemory(as: Element.self, from: buf)
        return r
    }
    func view<R>(as type: R.Type) -> MemoryPtr<R> {
        .init(shared: address)
    }
}

public struct Buffer<Element> {
    private var mptr: MemoryPtr<Element>
    private var offset: Int
    private let count: Int
    private let shared: Bool
    private var retaining: AnyObject?
    init(owner: Buffer<Element>, offset: Int, count: Int, shared: Bool) {
        self.mptr = owner.mptr
        self.offset = offset
        self.count = count
        self.shared = shared
    }
    init(capacity: Int, initialValue: Element) {
        mptr = .init(capacity: capacity, initialValue: initialValue)
        offset = 0
        count = capacity
        shared = false
    }
    init(address: MemoryPtr<Element>, shared: Bool) {
        mptr = address
        self.shared = shared
        offset = 0
        count = address.count
    }
    init<X>(address: MemoryPtr<Element>, retaining: MemoryPtr<X>) {
        mptr = address
        shared = true
        offset = 0
        count = address.count
        self.retaining = retaining
    }
    init(capacity: Int) {
        mptr = .init(capacity: capacity)
        offset = 0
        count = capacity
        shared = false
    }

    public var copy: Self {
        let mem: MemoryPtr<Element> = offset != 0 ? mptr.copy(range: offset..<offset+count) : mptr
        return .init(address: mem, shared: false)
    }

    func apply<R>(_ body: (UnsafePointer<Element>) -> R) -> R {
        body(readable)
    }

    func view<R>(as type: R.Type) -> Buffer<R> {
        .init(address: mptr.view(as: type), retaining: mptr)
    }

    subscript(index: Range<Int>, shared: Bool) -> Self {
        .init(owner: self, offset: offset+index.lowerBound, count: index.count, shared: shared)
    }
    mutating func transfer(range: Range<Int>, from: Self) {
        if mptr.address.baseAddress! == from.mptr.address.baseAddress! {
            return
        }
        apply { optr in
            from.apply { iptr in
                optr.advanced(by: range.lowerBound).update(from: iptr, count: range.count)
            }
        }
    }
    public var readable: UnsafePointer<Element> {
        .init(mptr.address.baseAddress!.assumingMemoryBound(to: Element.self).advanced(by: offset))
    }
    public var writable: UnsafeMutablePointer<Element> {
        mutating get {
            if !shared && !isKnownUniquelyReferenced(&mptr) {
//                print("copying \(count) elements")
                self.mptr = self.mptr.copy(range: offset..<(offset+count))
                self.offset = 0
            }
            return mptr.address.baseAddress!.assumingMemoryBound(to: Element.self).advanced(by: offset)
        }
    }
    public var sharedWritable: UnsafeMutablePointer<Element> {
        return mptr.address.baseAddress!.assumingMemoryBound(to: Element.self).advanced(by: offset)
    }
    mutating func apply<R>(_ body: (UnsafeMutablePointer<Element>) -> R) -> R {
        body(writable)
    }
//    mutating func applyShared<R>(_ body: (UnsafeMutablePointer<Element>) -> R) -> R {
//        body(sharedWritable)
//    }
}

public struct BufferIndexingIterator<IT: IteratorProtocol, Element>: IteratorProtocol where IT.Element: FixedWidthInteger {
    private var it: IT
    private var mem: UnsafePointer<Element>
    init(it: IT, mem: UnsafePointer<Element>) {
        self.it = it
        self.mem = mem
    }
    mutating public func next() -> Element? {
        if let index = it.next() {
            return mem[Int(index)]
        }
        return nil
    }
}

struct Zip3Sequence<A, B, C>: Sequence where A: Sequence, B: Sequence, C: Sequence {
    let a: A
    let b: B
    let c: C
    struct Iterator: IteratorProtocol {
        var ita: A.Iterator
        var itb: B.Iterator
        var itc: C.Iterator
        mutating func next() -> (A.Element, B.Element, C.Element)? {
            if let a=ita.next(), let b = itb.next(), let c = itc.next() {
                return (a, b, c)
            }
            return nil
        }
    }
    var underestimatedCount: Int {
        Swift.min(Swift.min(a.underestimatedCount, b.underestimatedCount), c.underestimatedCount)
    }
    func makeIterator() -> Iterator {
        .init(ita: a.makeIterator(), itb: b.makeIterator(), itc: c.makeIterator())
    }
}

func maxFlow<T: FixedWidthInteger>(capacities:inout [[T]], source: Int, sink: Int) -> T {
    let n = capacities.count
    var visited: [T] = .init(repeating: 0, count: n)
    var maxFlow: T = 0
    var visitedToken: T = 1
    func dfs(node: Int, flow: T) -> T {
        if node == sink {
            return flow
        }
        visited[node] = visitedToken
        var flow = flow
        for i in  0..<n {
            if visited[i] != visitedToken && capacities[node][i] > 0 {
                if capacities[node][i] < flow {
                    flow = capacities[node][i]
                }
                let dfsFlow = dfs(node: i, flow: flow)
                if dfsFlow > 0 {
                    capacities[node][i] -= dfsFlow
                    capacities[i][node] += dfsFlow
                    return dfsFlow
                }
            }
        }
        return 0
    }
    while true {
        let flow = dfs(node: source, flow: T.max)
        if flow == 0 {
            return maxFlow
        }
        maxFlow += flow
        visitedToken += 1
    }
}

func dot<S, T: FixedWidthInteger>(a: [S], b: [S], c: [S], capacities: [[T]], source: Int, sink: Int) -> String {
    var r = "digraph G {\n"
    var j = 0
    r+="n\(source)[label=\"source\"];\n"
    r+="n\(sink)[label=\"sink\"];\n"
    for i in [("a", a), ("b", b), ("c", c)].enumerated() {
        r+="subgraph cluster_\(i.offset) {\n"
        r+="label=\"\(i.element.0)\";\n"
        for k in i.element.1.enumerated() {
            r+="n\(k.offset+j) [label=\"\(k.element)\"];\n"
        }
        j += i.element.1.count
        r+="}\n"
    }
    for i in capacities.indices {
        for j in capacities[i].indices {
            if capacities[i][j] > 0 {
                r += "n\(i)->n\(j) [label=\"\(capacities[i][j])\"];\n"
            } else if capacities[i][j] < 0 {
                r += "n\(j)->n\(i);\n"
            }
        }
    }
    r += "}"
    return r
}
struct SequencePair<A: Sequence, B: Sequence>: Sequence where A.Element == B.Element {
    let a: A
    let b: B
    struct Iterator<A: IteratorProtocol, B: IteratorProtocol>: IteratorProtocol where A.Element == B.Element {
        var a: A?
        var b: B?
        mutating func next() -> A.Element? {
            if let n = a?.next() {
                return n
            } else {
                a = nil
                if let n = b?.next() {
                    return n
                }
                b = nil
            }
            return nil
        }
    }
    func makeIterator() -> Iterator<A.Iterator, B.Iterator> {
        .init(a: a.makeIterator(), b: b.makeIterator())
    }
}

public struct NormalDistribution<T: BinaryFloatingPoint, R: RandomNumberGenerator>: IteratorProtocol where T.RawSignificand: FixedWidthInteger {
    var rng: R
    var n1: T = .nan
    var n2: T = .nan
    public init(using rng: R) {
        self.rng=rng
    }
    public mutating func next() -> T? {
        if n2.isNaN {
            var u1: Double = Double.random(in: 0..<1, using: &rng)
            var u2: Double = Double.random(in: 0..<1, using: &rng)
            while u1 <= Double.leastNonzeroMagnitude || u2 <= Double.leastNonzeroMagnitude {
                u1 = Double.random(in: 0..<1, using: &rng)
                u2 = Double.random(in: 0..<1, using: &rng)
            }
            n1 = T((-2*log(u1)).squareRoot() * cos(2*Double.pi*u2))
            n2 = T((-2*log(u1)).squareRoot() * sin(2*Double.pi*u2))
        }
        if n1.isNaN {
            defer { n2 = .nan }
            return n2
        } else {
            defer { n1 = .nan}
            return n1
        }
    }
}
