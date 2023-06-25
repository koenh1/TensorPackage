//
//  File.swift
//  
//
//  Created by Koen Hendrikx on 19/06/2023.
//

import Foundation
import RegexBuilder

struct Graph<ValueType: DifferentiableValue>: CustomStringConvertible {
    var nodes:[(id: ObjectIdentifier, gradient: ValueType)] = []
    let generation: UInt32
    var links:[(parent: ObjectIdentifier, child: ObjectIdentifier)] = []
    init(_ header: any HeaderProtocol) {
        var list: [any HeaderProtocol] = []
        globalGeneration += 1
        self.generation = globalGeneration
        func listSorted(_ header: any HeaderProtocol) {
            if header.generation != generation {
                header.clearGradient(generation: generation)
                header.applyLeft(visitor: listSorted)
                header.applyRight(visitor: listSorted)
                list.append(header)
            }
        }
        listSorted(header)
        for i in list {
            i.clearGradient(generation: generation)
        }
        header.unitGradient()
        for x in list.reversed() {
            x.updateGrad()
        }
        func getInfo<T: HeaderProtocol>(x: T) -> (id: ObjectIdentifier, gradient: ValueType) {
            (ObjectIdentifier(x), x.grad as! ValueType)
        }
        for node in list {
            let info = getInfo(x: node)
            node.applyLeft(visitor: {h in links.append((parent:ObjectIdentifier(node), child:ObjectIdentifier(h)))})
            node.applyRight(visitor: {h in links.append((parent:ObjectIdentifier(node), child:ObjectIdentifier(h)))})
            nodes.append(info)
        }
    }
    var description: String {
        if #available(macOS 13.0, *) {
            let regex = Regex {
                "ObjectIdentifier(0"
                Capture {
                    "x"
                    OneOrMore {CharacterClass(.anyOf(")")).inverted}
                }
                ")"
            }
            func rep(_ id: ObjectIdentifier) -> String {
                id.debugDescription.replacing(regex, with: \.1)
            }

            return "digraph G {\nrankdir=LR\n"
            +
            nodes.map {"\(rep($0.id)) [shape=record, label=\"{{\($0.gradient)}}\"];"}.joined(separator: "\n")
            + "\n" +
            links.map {"\(rep($0.parent))->\(rep($0.child));"}.joined(separator: "\n")
            + "\n}"
        } else {
            return ""
        }
    }
}

extension Differentiable {
    public var graph: String {
        Graph<ValueType>(header).description
    }
}
