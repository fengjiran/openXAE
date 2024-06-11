//
// Created by fengj on 2024/6/1.
//

//#include "glog/logging.h"
#include "pnnx/src/ir.h"
#include "torch/torch.h"
#include "gtest/gtest.h"

namespace pnnx {

TEST(IRTEST, type_check) {
    static_assert(std::is_integral_v<long>);
    static_assert(std::is_integral_v<uint8_t>);
    static_assert(std::is_integral_v<short>);
    static_assert(std::is_integral_v<long long>);
    static_assert(std::is_integral_v<bool>);
    static_assert(std::is_floating_point_v<float>);
    static_assert(std::is_floating_point_v<double>);
    static_assert(std::is_same_v<std::decay_t<const char*>, std::string> || std::is_convertible_v<const char*, std::string>);

    static_assert(is_std_vector_int_v<std::vector<int>>);
    static_assert(is_std_vector_float_v<std::vector<float>>);
    static_assert(is_std_vector_string_v<std::vector<std::string>>);
    static_assert(is_std_vector_complex_v<std::vector<std::complex<float>>>);

    static_assert(is_std_vector_int_v<std::initializer_list<int>>);
    static_assert(is_std_vector_float_v<std::initializer_list<float>>);
    static_assert(is_std_vector_string_v<std::initializer_list<std::string>>);
    static_assert(is_std_vector_complex_v<std::initializer_list<std::complex<float>>>);

    EXPECT_TRUE(get_parameter_type<bool>() == ParameterType::kParameterBool);
    EXPECT_TRUE(get_parameter_type<int>() == ParameterType::kParameterInt);
    EXPECT_TRUE(get_parameter_type<long>() == ParameterType::kParameterInt);
    EXPECT_TRUE(get_parameter_type<long long>() == ParameterType::kParameterInt);
    EXPECT_TRUE(get_parameter_type<float>() == ParameterType::kParameterFloat);
    EXPECT_TRUE(get_parameter_type<double>() == ParameterType::kParameterFloat);
    EXPECT_TRUE(get_parameter_type<std::complex<float>>() == ParameterType::kParameterComplex);
    EXPECT_TRUE(get_parameter_type<const char*>() == ParameterType::kParameterString);
    EXPECT_TRUE(get_parameter_type<std::string>() == ParameterType::kParameterString);
    EXPECT_TRUE(get_parameter_type<std::vector<int>>() == ParameterType::kParameterArrayInt);
    EXPECT_TRUE(get_parameter_type<std::vector<int64_t>>() == ParameterType::kParameterArrayInt);
    EXPECT_TRUE(get_parameter_type<std::initializer_list<int>>() == ParameterType::kParameterArrayInt);
    EXPECT_TRUE(get_parameter_type<std::vector<float>>() == ParameterType::kParameterArrayFloat);
    EXPECT_TRUE(get_parameter_type<std::vector<double>>() == ParameterType::kParameterArrayFloat);
    EXPECT_TRUE(get_parameter_type<std::vector<std::string>>() == ParameterType::kParameterArrayString);
    EXPECT_TRUE(get_parameter_type<std::vector<const char*>>() == ParameterType::kParameterArrayString);
    EXPECT_TRUE(get_parameter_type<std::vector<std::complex<float>>>() == ParameterType::kParameterArrayComplex);
}

TEST(IRTEST, Parameter) {
    // null parameter
    Parameter p_null;
    ASSERT_EQ(p_null.type(), ParameterType::kParameterUnknown);
    ASSERT_EQ(Parameter::Encode2String(p_null), "None");
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_null)) == p_null);

    // bool parameter
    Parameter p_bool1(true);
    Parameter p_bool2(false);
    ASSERT_EQ(p_bool1.type(), ParameterType::kParameterBool);
    ASSERT_EQ(Parameter::Encode2String(p_bool1), "True");
    ASSERT_EQ(Parameter::Encode2String(p_bool2), "False");
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_bool1)) == p_bool1);
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_bool2)) == p_bool2);

    // int parameter
    Parameter p_int1(-10ll);// 10, 10l
    std::cout << Parameter::Encode2String(p_int1) << std::endl;
    ASSERT_EQ(p_int1.type(), ParameterType::kParameterInt);
    ASSERT_EQ(Parameter::Encode2String(p_int1), "-10");
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_int1)) == p_int1);

    // float parameter
    Parameter p_float(0.3141592657);
    std::cout << Parameter::Encode2String(p_float) << std::endl;
    ASSERT_EQ(p_float.type(), ParameterType::kParameterFloat);
    ASSERT_EQ(Parameter::Encode2String(p_float), "3.141593e-01");
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_float)) == p_float);

    // string parameter
    Parameter p_str("pnnx");
    std::cout << Parameter::Encode2String(p_str) << std::endl;
    ASSERT_EQ(p_str.type(), ParameterType::kParameterString);
    ASSERT_EQ(Parameter::Encode2String(p_str), "pnnx");
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_str)) == p_str);

    // array int parameter
    Parameter p_ai1 = {1, 2, 3, 4, -5};
    ASSERT_EQ(p_ai1.type(), ParameterType::kParameterArrayInt);
    ASSERT_EQ(Parameter::Encode2String(p_ai1), "(1,2,3,4,-5)");
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_ai1)) == p_ai1);

    Parameter p_ai2(std::vector<int>({1, 2, 3, 4, 5}));
    ASSERT_EQ(p_ai2.type(), ParameterType::kParameterArrayInt);
    ASSERT_EQ(Parameter::Encode2String(p_ai2), "(1,2,3,4,5)");
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_ai2)) == p_ai2);

    // array float parameter
    Parameter p_af1 = {1.0, 0.112, -3.14};
    ASSERT_EQ(p_af1.type(), ParameterType::kParameterArrayFloat);
    ASSERT_EQ(Parameter::Encode2String(p_af1), "(1.000000e+00,1.120000e-01,-3.140000e+00)");
    ASSERT_TRUE(Parameter::ParseFromString(Parameter::Encode2String(p_af1)) == p_af1);

    // complex parameter
    Parameter p_c(std::complex<float>(2, 3));
    ASSERT_EQ(p_c.type(), ParameterType::kParameterComplex);
    ASSERT_EQ(Parameter::Encode2String(p_c), "2.000000e+00+3.000000e+00i");
}

TEST(IRTEST, new_parameter) {
    Parameter_ p1(true);
    static_assert(std::is_same_v<decltype(p1)::value_type, bool>);
    EXPECT_EQ(p1.type(), ParameterType::kParameterBool);
    EXPECT_TRUE(p1.toValue());
    p1.SetValue(false);
    EXPECT_FALSE(p1.toValue());

    Parameter_ p2(10);
    static_assert(std::is_same_v<decltype(p2)::value_type, int>);
    EXPECT_EQ(p2.type(), ParameterType::kParameterInt);
    EXPECT_EQ(p2.toValue(), 10);
    p2.SetValue(20);
    EXPECT_EQ(p2.toValue(), 20);

    Parameter_ p3(2.5);
    static_assert(std::is_same_v<decltype(p3)::value_type, double>);
    EXPECT_EQ(p3.type(), ParameterType::kParameterFloat);
    EXPECT_EQ(p3.toValue(), 2.5);
    p3.SetValue(5.0);
    EXPECT_EQ(p3.toValue(), 5.0);

    Parameter_ p4("hello");
    static_assert(std::is_same_v<decltype(p4)::value_type, std::string>);
    EXPECT_EQ(p4.type(), ParameterType::kParameterString);
    EXPECT_EQ(p4.toValue(), "hello");
    p4.SetValue("world");
    EXPECT_EQ(p4.toValue(), "world");

    Parameter_ p5(std::complex<float>(2, 3));
    static_assert(std::is_same_v<decltype(p5)::value_type, std::complex<float>>);
    EXPECT_EQ(p5.type(), ParameterType::kParameterComplex);
    EXPECT_EQ(p5.toValue(), std::complex<float>(2, 3));
    p5.SetValue(std::complex<float>(3, 4));
    EXPECT_EQ(p5.toValue(), std::complex<float>(3, 4));

    Parameter_ p6(std::vector<int>{1, 2, 3});
    static_assert(std::is_same_v<decltype(p6)::value_type, std::vector<int>>);
    EXPECT_EQ(p6.type(), ParameterType::kParameterArrayInt);
    EXPECT_EQ(p6.toValue(), (std::vector<int>{1, 2, 3}));
    p6.SetValue({4, 5, 6});
    EXPECT_EQ(p6.toValue(), (std::vector<int>{4, 5, 6}));
    p6.AddElemToArray(7);
    EXPECT_EQ(p6.toValue(), (std::vector<int>{4, 5, 6, 7}));

    Parameter_ p7({1, 2, 3});
    static_assert(std::is_same_v<decltype(p7)::value_type, std::vector<int>>);
    EXPECT_EQ(p7.type(), ParameterType::kParameterArrayInt);
    EXPECT_EQ(p7.toValue(), (std::vector<int>{1, 2, 3}));
    p7.SetValue({4, 5, 6});
    EXPECT_EQ(p7.toValue(), (std::vector<int>{4, 5, 6}));
    p7.AddElemToArray(7);
    EXPECT_EQ(p7.toValue(), (std::vector<int>{4, 5, 6, 7}));

    Parameter_ p8({"hello", "world"});
    static_assert(std::is_same_v<decltype(p8)::value_type, std::vector<std::string>>);
    EXPECT_EQ(p8.type(), ParameterType::kParameterArrayString);
    EXPECT_EQ(p8.toValue(), (std::vector<std::string>{"hello", "world"}));
    p8.SetValue({"Effective", "Modern", "C++"});
    EXPECT_EQ(p8.toValue(), (std::vector<std::string>{"Effective", "Modern", "C++"}));
    p8.AddElemToArray("Scott Meyers");
    EXPECT_EQ(p8.toValue(), (std::vector<std::string>{"Effective", "Modern", "C++", "Scott Meyers"}));
}

TEST(IRTEST, libtorch) {
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
    std::string param_path = "test_linear.pnnx.param";
    std::string bin_path = "test_linear.pnnx.bin";
    Graph graph;
    int status_load = graph.load(param_path, bin_path);
    ASSERT_EQ(status_load, 0);

    int status_save = graph.save("test_linear1.pnnx.param", "test_linear1.pnnx.bin");
    ASSERT_EQ(status_save, 0);
}

TEST(IRTEST, create_pnnx_graph) {
    Graph graph;
    auto t1 = graph.CreateOperator("pnnx.Input", "pnnx_input_0",
                                   {}, {}, {}, {},
                                   "0", DataType::kDataTypeFloat32, {1, 32});

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
    graph.save("linear.pnnx.param", "linea.pnnx.bin");
}

}// namespace pnnx