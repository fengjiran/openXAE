//
// Created by richard on 8/6/24.
//

#ifndef OPENXAE_PASS_LEVEL2_H
#define OPENXAE_PASS_LEVEL2_H

#include "Graph.h"

namespace pnnx {

class GraphRewriterPass {
public:
    virtual ~GraphRewriterPass() = default;

    virtual std::string MatchPatternGraph() const = 0;
    virtual std::string ReplacePatternGraph() const {
        return {};
    }

    virtual std::string TypeStr() const {
        std::cerr << "GraphRewriterPass TypeStr() should be implemented\n";
        return "unk";
    }

    virtual std::string NameStr() const {
        return TypeStr();
    }

    virtual bool Match(const std::map<std::string, Parameter>& capturedParams) const {
        return true;
    }

    virtual bool Match(const std::map<std::string, Parameter>& capturedParams,
                       const std::map<std::string, Attribute>& capturedAttrs) const {
        return Match(capturedParams);
    }

    virtual bool Match(const std::map<std::string, const std::shared_ptr<Operator>>&,
                       const std::map<std::string, Parameter>& capturedParams,
                       const std::map<std::string, Attribute>& capturedAttrs) const {
        return Match(capturedParams, capturedAttrs);
    }

    virtual void Write(const std::shared_ptr<Operator>& op,
                       const std::map<std::string, Parameter>& capturedParams) const;

    virtual void Write(const std::shared_ptr<Operator>& op,
                       const std::map<std::string, Parameter>& capturedParams,
                       const std::map<std::string, Attribute>& capturedAttrs) const;

    virtual void Write(const std::map<std::string, std::shared_ptr<Operator>>& ops,
                       const std::map<std::string, Parameter>& capturedParams) const;

    virtual void Write(const std::map<std::string, std::shared_ptr<Operator>>& ops,
                       const std::map<std::string, Parameter>& capturedParams,
                       const std::map<std::string, Attribute>& capturedAttrs) const;
};

class GraphRewriterPassRegistry {
public:
    static GraphRewriterPassRegistry& GetInstance() {
        static GraphRewriterPassRegistry inst;
        return inst;
    }

    void Register(const std::shared_ptr<GraphRewriterPass>& pass, int priority) {
        if (passes_.find(priority) == passes_.end()) {
            passes_[priority] = std::vector<std::shared_ptr<GraphRewriterPass>>();
        }
        passes_[priority].push_back(pass);
    }

    const auto& GetGlobalPNNXGraphRewriterPass() const {
        return passes_;
    }

private:
    GraphRewriterPassRegistry() = default;
    std::map<int, std::vector<std::shared_ptr<GraphRewriterPass>>> passes_{};
};

template<typename Pass, int Priority>
class GraphRewriterPassRegEntry {
public:
    GraphRewriterPassRegEntry() {
        GraphRewriterPassRegistry::GetInstance().Register(
                std::make_shared<Pass>(), Priority);
    }
};

#define REGISTER_PNNX_GRAPH_REWRITER_PASS(PASS, PRIORITY) \
    static GraphRewriterPassRegEntry<PASS, PRIORITY> CONCAT_STR(pnnx_graph_rewriter_pass_, PASS) = GraphRewriterPassRegEntry<PASS, PRIORITY>()

}// namespace pnnx

#endif//OPENXAE_PASS_LEVEL2_H
