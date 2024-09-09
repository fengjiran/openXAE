//
// Created by richard on 9/9/24.
//
#include "pnnx/converter/include/torch/pass_level2.h"

namespace pnnx {

class F_relu : public GraphRewriterPass {
public:
    std::string MatchPatternGraph() const {
        return R"PNNXIR(7767517
3 2
pnnx.Input              input       0 1 input
aten::relu              op_0        1 1 input out
pnnx.Output             output      1 0 out
)PNNXIR";
    }

    std::string TypeStr() const {
        return "F.relu";
    }
};

REGISTER_PNNX_GRAPH_REWRITER_PASS(F_relu, 10);

}// namespace pnnx