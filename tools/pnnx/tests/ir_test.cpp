//
// Created by fengj on 2024/6/1.
//

//#include "glog/logging.h"
#include "Graph.h"
#include "pnnx/converter/include/torch/torch2pnnx.h"

#include <gtest/gtest.h>
#include <torch/csrc/jit/frontend/tracer.h>

namespace pnnx {

TEST(IRTEST, type_check) {
    static_assert(std::is_integral_v<long>);
    static_assert(std::is_integral_v<uint8_t>);
    static_assert(std::is_integral_v<short>);
    static_assert(std::is_integral_v<long long>);
    static_assert(std::is_integral_v<int64_t>);
    static_assert(std::is_integral_v<bool>);
    static_assert(std::is_floating_point_v<float>);
    static_assert(std::is_floating_point_v<double>);
    static_assert(!std::is_same_v<std::decay_t<const char*>, std::string>);
    static_assert(std::is_convertible_v<const char*, std::string>);
    static_assert(is_string_v<const char*>);
    static_assert(is_string_v<std::decay_t<const char*>>);

    static_assert(is_std_vector_int_v<std::vector<int>>);
    static_assert(is_std_vector_float_v<std::vector<float>>);
    static_assert(is_std_vector_string_v<std::vector<std::string>>);
    static_assert(is_std_vector_complex_v<std::vector<std::complex<float>>>);
    static_assert(is_std_vector_int_v<std::initializer_list<int>>);
    static_assert(is_std_vector_float_v<std::initializer_list<float>>);
    static_assert(is_std_vector_string_v<std::initializer_list<std::string>>);
    static_assert(is_std_vector_complex_v<std::initializer_list<std::complex<float>>>);

    static_assert(GetParameterType<void*>() == ParameterType::kParameterUnknown);
    static_assert(GetParameterType<bool>() == ParameterType::kParameterBool);
    static_assert(GetParameterType<int>() == ParameterType::kParameterInt);
    static_assert(GetParameterType<long>() == ParameterType::kParameterInt);
    static_assert(GetParameterType<long long>() == ParameterType::kParameterInt);
    static_assert(GetParameterType<char>() == ParameterType::kParameterInt);
    static_assert(GetParameterType<float>() == ParameterType::kParameterFloat);
    static_assert(GetParameterType<double>() == ParameterType::kParameterFloat);
    static_assert(GetParameterType<std::string>() == ParameterType::kParameterString);
    static_assert(GetParameterType<const char*>() == ParameterType::kParameterString);
    static_assert(GetParameterType<std::complex<float>>() == ParameterType::kParameterComplex);
    static_assert(GetParameterType<std::vector<int>>() == ParameterType::kParameterArrayInt);
    static_assert(std::is_convertible_v<int&&, int>);

    int keys[] = {1, 3, 7, 9, 11, 22, 35};
    static_assert(GetArraySize(keys) == 7);
}

TEST(IRTEST, Parameter) {
    //    GTEST_SKIP();
    // null parameter
    {
        Parameter p0;
        EXPECT_FALSE(p0.has_value());
        EXPECT_EQ(p0.type(), ParameterType::kParameterUnknown);
        EXPECT_EQ(p0.toString(), "None");
    }

    // bool parameter
    {
        Parameter p1(true);
        EXPECT_EQ(p1.type(), ParameterType::kParameterBool);
        EXPECT_EQ(p1.toString(), "True");
        EXPECT_EQ(p1.toValue<bool>(), true);
        p1 = false;
        EXPECT_EQ(p1.toString(), "False");
    }

    // int parameter
    {
        Parameter p2(-10);
        EXPECT_EQ(p2.type(), ParameterType::kParameterInt);
        EXPECT_EQ(p2.toString(), "-10");
        EXPECT_EQ(p2.toValue<int>(), -10);

        p2 = (int64_t) 20;
        EXPECT_EQ(p2.toValue<int>(), 20);

        p2.toValue<int>() = 10;
        EXPECT_EQ(p2.toValue<int>(), 10);
    }

    // float parameter
    {
        Parameter p3(0.3141592657);
        EXPECT_EQ(p3.type(), ParameterType::kParameterFloat);
        EXPECT_EQ(p3.toString(), "3.141593e-01");
        EXPECT_FLOAT_EQ(p3.toValue<float>(), 0.3141592657);
        p3 = 3.14;
        EXPECT_EQ(p3.type(), ParameterType::kParameterFloat);
        EXPECT_EQ(p3.toString(), "3.140000e+00");
        EXPECT_FLOAT_EQ(p3.toValue<float>(), 3.14);
    }

    // string parameter
    {
        Parameter p4("pnnx");
        EXPECT_EQ(p4.type(), ParameterType::kParameterString);
        EXPECT_EQ(p4.toString(), "pnnx");
        EXPECT_EQ(p4.toValue<std::string>(), "pnnx");
        p4 = "tvm";
        EXPECT_EQ(p4.toValue<std::string>(), "tvm");
        p4.SetOtherType();
        EXPECT_EQ(p4.type(), ParameterType::kParameterOther);
    }

    // array int parameter
    {
        Parameter p5 = {1, 2, 3, 4, -5};
        EXPECT_EQ(p5.type(), ParameterType::kParameterArrayInt);
        EXPECT_EQ(p5.toString(), "(1,2,3,4,-5)");
        EXPECT_EQ(p5.toValue<std::vector<int>>(), std::vector<int>({1, 2, 3, 4, -5}));
    }

    // array int parameter
    {
        Parameter p6(std::vector<int>({1, 2, 3, 4, 5}));
        EXPECT_EQ(p6.type(), ParameterType::kParameterArrayInt);
        EXPECT_EQ(p6.toString(), "(1,2,3,4,5)");
        EXPECT_EQ(p6.toValue<std::vector<int>>(), std::vector<int>({1, 2, 3, 4, 5}));
        p6 = {1, 2, 3};
        EXPECT_EQ(p6.toValue<std::vector<int>>(), std::vector<int>({1, 2, 3}));
    }

    // array float parameter
    {
        Parameter p7 = {1.0, 0.112, -3.14};
        EXPECT_EQ(p7.type(), ParameterType::kParameterArrayFloat);
        EXPECT_EQ(p7.toString(), "(1.000000e+00,1.120000e-01,-3.140000e+00)");
        EXPECT_EQ(p7.toValue<std::vector<float>>(), std::vector<float>({1.0, 0.112, -3.14}));
        p7 = {1.0, 2.0, 3.0};
        EXPECT_EQ(p7.toValue<std::vector<float>>(), std::vector<float>({1.0, 2.0, 3.0}));
    }

    // array string parameter
    {
        Parameter p = {"open", "XAE"};
        EXPECT_EQ(p.type(), ParameterType::kParameterArrayString);
        EXPECT_EQ(p.toString(), "(open,XAE)");
        EXPECT_EQ(p.toValue<std::vector<std::string>>(), std::vector<std::string>({"open", "XAE"}));
    }

    // complex parameter
    {
        Parameter p8(std::complex<float>(2, 3));
        EXPECT_EQ(p8.type(), ParameterType::kParameterComplex);
        EXPECT_EQ(p8.toString(), "2.000000e+00+3.000000e+00i");
    }
}

TEST(IRTEST, libtorch) {
    GTEST_SKIP();
    auto options = torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .layout(torch::kStrided)
                           .device(torch::kCPU)
                           .requires_grad(false);
    auto a = torch::rand({2, 3}, options);
    std::cout << a.sizes() << std::endl;
    std::cout << a.dtype() << std::endl;
    std::cout << a.device() << std::endl;
    std::cout << a.layout() << std::endl;
    std::cout << a.requires_grad() << std::endl;
    ASSERT_EQ(a.dtype().name(), "float");
}

TEST(IRTEST, Attribute) {
    GTEST_SKIP();
    at::IntArrayRef shape = {2, 3};

    auto t = torch::rand(shape);
    t.contiguous();

    std::cout << "stride0: " << t.stride(0) << std::endl;
    std::cout << "stride1: " << t.stride(1) << std::endl;

    for (int64_t i = 0; i < shape[0]; ++i) {
        for (int64_t j = 0; j < shape[1]; ++j) {
            int64_t idx = i * t.stride(0) + j;
            ASSERT_FLOAT_EQ(t[i][j].item().toFloat(), *(t.data_ptr<float>() + idx));
        }
    }

    Attribute weight({2, 3}, std::vector<float>(t.data_ptr<float>(), t.data_ptr<float>() + t.numel()));
    std::cout << "elem size: " << weight.GetElemSize() << std::endl;
    std::cout << "elem count: " << weight.size() << std::endl;

    std::string x = "#input=(1,3,10,10)f32";
    std::cout << x.substr(x.find_last_of(')') + 1) << std::endl;
    std::cout << x.substr(1, x.find_last_of(')')) << std::endl;
}

TEST(IRTEST, pnnx_graph_load) {
    //    GTEST_SKIP();
    std::string param_path = "test_linear.pnnx.param";
    std::string bin_path = "test_linear.pnnx.bin";
    Graph graph;
    int status_load = graph.load(param_path, bin_path);
    ASSERT_EQ(status_load, 0);

    int status_save = graph.save("test_linear1.pnnx.param", "test_linear1.pnnx.bin");
    ASSERT_EQ(status_save, 0);
}

TEST(IRTEST, create_pnnx_graph) {
    //    GTEST_SKIP();
    Graph graph;
    auto t1 = graph.CreateOperator("pnnx.Input", "pnnx_input_0",
                                   {}, {}, {}, {},
                                   "0", DataType::kDataTypeFloat32, {1, 32});//{1,32}

    auto bias = torch::rand({128});
    auto weight = torch::rand({128, 32});
    bias.contiguous();
    weight.contiguous();

    auto t2 = graph.CreateOperator("nn.Linear", "linear",
                                   {{"bias", std::make_shared<Parameter>(true)},
                                    {"in_features", std::make_shared<Parameter>(32)},
                                    {"out_features", std::make_shared<Parameter>(128)}},
                                   {{"bias", std::make_shared<Attribute>(std::vector<int>{128}, std::vector<float>(bias.data_ptr<float>(), bias.data_ptr<float>() + bias.numel()))},
                                    {"weight", std::make_shared<Attribute>(std::vector<int>{128, 32}, std::vector<float>(weight.data_ptr<float>(), weight.data_ptr<float>() + weight.numel()))}},
                                   {t1}, {},
                                   "1", DataType::kDataTypeFloat32, {1, 128});

    auto t3 = graph.CreateOperator("F.sigmoid", "F.sigmoid_0",
                                   {}, {}, {t2}, {"input"},
                                   "2", DataType::kDataTypeFloat32, {1, 128});

    auto t4 = graph.CreateOperator("pnnx.Output", "pnnx_output_0",
                                   {}, {}, {t3}, {}, {}, {}, {});
    ASSERT_TRUE(!t4);
    graph.save("linear.pnnx.param", "linear.pnnx.bin");

    Operator x("pnnx.Output", "pnnx_output_0");
    auto iter = x.GetParameters().find("value");
    if (iter == x.GetParameters().end()) {
        x.GetParameters().insert({"value", std::make_shared<Parameter>()});
    }

    const auto& p = x.GetParameters()["value"];
    std::cout << p->toString() << std::endl;
}

TEST(IRTEST, torch2pnnx) {
    //    GTEST_SKIP();
    std::string pt = "test_nn_Conv2d.pt";
    Graph g;
    std::set<std::string> foldConstants;
    //    std::string pt = "test_inline_block.pt";
    torch2pnnx(pt, g, "cpu", {}, {},
               {}, {}, {}, {},
               "", foldConstants);
}

TEST(IRTEST, create_parameter_from_torch_node) {
    //    GTEST_SKIP();
    auto g = torch::jit::Graph();

    auto node1 = g.createNone();
    EXPECT_TRUE(node1->output()->type()->kind() == c10::TypeKind::NoneType);
    Parameter p1 = CreateParameterFromTorchNode(node1);
    std::cout << "p1: " << p1.toString() << std::endl;
//    p1 = node1;
//    std::cout << (int) p1.type() << std::endl;

    torch::jit::Node* node2 = g.create(c10::prim::Constant);
    node2->output()->setType(c10::IntType::get());
    EXPECT_TRUE(node2->output()->type()->kind() == c10::TypeKind::IntType);
    node2->i_(torch::jit::attr::value, 10);
    Parameter p2 = CreateParameterFromTorchNode(node2);
    std::cout << "p2: " << p2.toString() << std::endl;

    auto node3 = g.create(c10::prim::Constant);
    node3->output()->setType(c10::BoolType::get());
    EXPECT_TRUE(node3->output()->type()->kind() == c10::TypeKind::BoolType);
    node3->i_(torch::jit::attr::value, false);
    auto p3 = CreateParameterFromTorchNode(node3);
    std::cout << "p3: " << p3.toString() << std::endl;

    auto node4 = g.create(c10::prim::Constant);
    node4->output()->setType(c10::FloatType::get());
    EXPECT_TRUE(node4->output()->type()->kind() == c10::TypeKind::FloatType);
    node4->f_(torch::jit::attr::value, 20);
    auto p4 = CreateParameterFromTorchNode(node4);
    std::cout << "p4: " << p4.toString() << std::endl;

    auto node5 = g.create(c10::prim::Constant);
    node5->output()->setType(c10::StringType::get());
    EXPECT_TRUE(node5->output()->type()->kind() == c10::TypeKind::StringType);
    node5->s_(torch::jit::attr::value, "hello, OpenXAE");
    auto p5 = CreateParameterFromTorchNode(node5);
    std::cout << "p5: " << p5.toString() << std::endl;

    auto node6 = g.create(c10::prim::Constant);
    node6->output()->setType(c10::ComplexType::get());
    EXPECT_TRUE(node6->output()->type()->kind() == c10::TypeKind::ComplexType);
    node6->c_(torch::jit::attr::value, c10::complex<float>(2, 3));
    auto p6 = CreateParameterFromTorchNode(node6);
    std::cout << "p6: " << p6.toString() << std::endl;

    auto node7 = g.create(c10::prim::Constant);
    EXPECT_TRUE(node7->output()->type()->kind() == c10::TypeKind::TensorType);
    node7->t_(torch::jit::attr::value, torch::rand({2, 3}));
    auto p7 = CreateParameterFromTorchNode(node7);
    std::cout << "p7: " << p7.toString() << std::endl;

    auto node8 = g.create(c10::prim::Constant);
    node8->output()->setType(c10::ListType::get("List<T>", c10::IntType::get()));
    EXPECT_TRUE(node8->output()->type()->kind() == c10::TypeKind::ListType);
    node8->ival_(torch::jit::attr::value, std::vector<int64_t>{1, 2, 3});
    auto p8 = CreateParameterFromTorchNode(node8);
    std::cout << "p8: " << p8.toString() << std::endl;

    auto node9 = g.create(c10::prim::Constant);
    node9->output()->setType(c10::ListType::get("List<T>", c10::FloatType::get()));
    EXPECT_TRUE(node9->output()->type()->kind() == c10::TypeKind::ListType);
    node9->ival_(torch::jit::attr::value, std::vector<double>{1., 2., 3.});
    auto p9 = CreateParameterFromTorchNode(node9);
    std::cout << "p9: " << p9.toString() << std::endl;
}

}// namespace pnnx