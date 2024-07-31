//
// Created by richard on 7/10/24.
//

#ifndef OPENXAE_PASS_LEVEL1_H
#define OPENXAE_PASS_LEVEL1_H

#include "Graph.h"

#include <torch/csrc/jit/api/module.h>
#include <torch/script.h>

namespace pnnx {

class FuseModulePass {
public:
    virtual ~FuseModulePass() = default;
    virtual std::string MatchTypeStr() const = 0;
    virtual std::string TypeStr() const = 0;
    virtual void Write(const std::shared_ptr<Operator>& op, const std::shared_ptr<torch::jit::Graph>& graph) const {}
    virtual void Write(const std::shared_ptr<Operator>& op,
                       const std::shared_ptr<torch::jit::Graph>& graph,
                       const torch::jit::Module& mod) const {
        Write(op, graph);
    }
};

class FuseModulePassRegistry {
public:
    static FuseModulePassRegistry& GetInstance() {
        static FuseModulePassRegistry inst;
        return inst;
    }

    void Register(const std::shared_ptr<FuseModulePass>& pass) {
        passes_.push_back(pass);
    }

    const auto& GetGlobalPNNXFuseModulePass() const {
        return passes_;
    }

private:
    FuseModulePassRegistry() = default;
    std::vector<std::shared_ptr<FuseModulePass>> passes_{};
};


template<typename Pass>
class FuseModulePassRegEntry {
public:
    explicit FuseModulePassRegEntry() {
        FuseModulePassRegistry::GetInstance().Register(std::make_shared<Pass>());
    }
};

#define REGISTER_PNNX_FUSE_MODULE_PASS(PASS) \
    static FuseModulePassRegEntry<PASS> CONCAT_STR(pnnx_fuse_module_pass_, PASS) = FuseModulePassRegEntry<PASS>()

void pass_level1(const torch::jit::Module& mod,
                 const std::shared_ptr<torch::jit::Graph>& g,
                 const std::vector<std::string>& moduleOperators,
                 Graph& pg);

}// namespace pnnx

#endif//OPENXAE_PASS_LEVEL1_H
