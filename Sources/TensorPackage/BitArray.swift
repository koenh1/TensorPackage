import Foundation

public struct BitArray: RandomAccessCollection, Hashable, CustomStringConvertible {
    public typealias Element = Bool
    public typealias Index = Int
    public typealias SubSequence = Self
    public typealias Indices = Range<Int>
    public let indices: Range<Int>
    fileprivate var data: [UInt64]

    public var endIndex: Int {
        indices.endIndex
    }
    public var startIndex: Int {
        indices.startIndex
    }

    public init(_ indices: Range<Int>) {
        self.indices = indices
        let wordcount = (indices.count+63)/64
        data = [UInt64](repeating: 0, count: wordcount)
    }
    public init(indices: Range<Int>, words: [UInt64]) {
        self.data = words
        self.indices = indices
    }
    public init(_ s: Self) {
        self.indices = s.indices
        data = s.data
    }
    public mutating func withMutableWords(f:(inout UnsafeMutableBufferPointer<UInt64>) -> Void) {
        data.withUnsafeMutableBufferPointer(f)
    }
    fileprivate var wordCount: Int {
        (indices.count+63)/64
    }
    public func xor(_ b: Self) -> Self {
        guard b.indices == indices else { fatalError() }
        var result = data
        for i in data.indices {
            result[i] ^= b.data[i]
        }
        return .init(indices: indices, words: result)
    }
    public subscript(range: Range<Int>) -> Self {
        get {
            assert(indices.contains(range.startIndex)&&indices.contains(range.endIndex-1))
            var r: Self = Self(range)
            for i in range {
                r[i] = self[i]
            }
            return r
        }
        set {
            assert(newValue.indices.count==range.count)
            for i in range {
                self[i] = newValue[i]
            }
        }
    }
    public subscript(position: Int) -> Bool {
        get {
            let pos = position - startIndex
            let index = pos >> 6
            return data[index] & (1 << (UInt64(pos & 63))) != 0
        }
        set(newValue) {
            let pos = position - startIndex
            let index = pos >> 6
            if newValue {
                data[index] |= 1 << (UInt64(pos & 63))
            } else {
                data[index] &= ~(1 << (UInt64(pos & 63)))
            }
        }
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(data)
    }

    public var description: String {
        var r: String = "{"
        var sep: String = ""
        for i in onSet {
            r += sep
            r += i.description
            sep = ", "
        }
        return r+"}"
    }

    public var onSet: BitSetSequence {
        BitSetSequence(self)
    }

    // proxy for "count"
    public var cardinality: Int {
        var sum: Int = 0
        let wordcount = (endIndex+63)/64
        for index in 0..<wordcount {
            let word = data[index]
            sum = sum &+ word.nonzeroBitCount
        }
        return sum
    }

    // check whether the value is empty
    public var isEmpty: Bool {
        for index in 0..<wordCount {
            let word = data[index]
            if word != 0 { return false; }
        }
        return true
    }

}
public struct BitSetSequence: Sequence {
    public typealias Element = Int
    public init(_ v: BitArray) {
        bits=v
    }
    let bits: BitArray
    public func makeIterator() -> BitArrayIterator {
        return BitArrayIterator(bits)
    }

}
public struct BitArrayIterator: IteratorProtocol {
    let bitset: BitArray
    let startIndex: Int
    var value: Int = -1

    init(_ bitset: BitArray) {
        self.bitset = bitset
        self.startIndex = bitset.startIndex
    }

    public mutating func next() -> Int? {
        value = value &+ 1
        var index = value >> 6
        if index >= bitset.wordCount {
            return nil
        }
        var word = bitset.data[index]
        word >>= UInt64(value & 63)
        if word != 0 {
            value = value &+ word.trailingZeroBitCount
            return value + startIndex
        }
        index = index &+ 1
        while index < bitset.wordCount {
            let word = bitset.data[index]
            if word != 0 {
                value = index &* 64 &+ word.trailingZeroBitCount
                return value + startIndex
            }
            index = index &+ 1
        }
        return nil
    }
}
