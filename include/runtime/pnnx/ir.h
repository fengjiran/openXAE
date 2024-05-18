//
// Created by fengj on 2024/5/20.
//

#ifndef OPENXAE_IR_H
#define OPENXAE_IR_H

namespace pnnx {

class Parameter {
public:
    Parameter() : type(0) {}
    explicit Parameter(bool b_) : type(1), b(b_) {}
    explicit Parameter(int i_) : type(2), i(i_) {}
    explicit Parameter(long l_) : type(2), i((int) l_) {}
    explicit Parameter(long long l_) : type(2), i((int) l_) {}
    explicit Parameter(float f_) : type(3), f(f_) {}
    explicit Parameter(double d_) : type(3), f((float) d_) {}
    explicit Parameter(const char* s_) : type(4), s(s_) {}
    explicit Parameter(const std::string& s_) : type(4), s(s_) {}
    explicit Parameter(const std::initializer_list<int>& ai_) : type(5), ai(ai_) {}
    explicit Parameter(const std::initializer_list<int64_t>& ai_) : type(5) {
        for (const auto& x: ai_) {
            ai.push_back((int) x);
        }
    }

    explicit Parameter(const std::vector<int>& ai_) : type(5), ai(ai_) {}
    explicit Parameter(const std::initializer_list<float>& af_) : type(6), af(af_) {}
    explicit Parameter(const std::initializer_list<double>& af_) : type(6) {
        for (const auto& x: af_) {
            af.push_back((float) x);
        }
    }

    explicit Parameter(const std::vector<float>& af_) : type(6), af(af_) {}
    explicit Parameter(const std::initializer_list<const char*>& as_) : type(7) {
        for (const auto& x: as_) {
            as.push_back(std::string(x));
        }
    }

    explicit Parameter(const std::initializer_list<std::string>& as_) : type(7), as(as_) {}
    explicit Parameter(const std::vector<std::string>& as_) : type(7), as(as_) {}

    static Parameter parse_from_string(const std::string& value);


    /**
     * @brief Parameter type
     *
     * 0 = null \n
     * 1 = bool \n
     * 2 = int \n
     * 3 = float \n
     * 4 = string \n
     * 5 = array int \n
     * 6 = array float \n
     * 7 = array string \n
     * 8 = others
     */
    int type;

    // value
    bool b;
    int i;
    float f;
    std::vector<int> ai;
    std::vector<float> af;
    std::string s;
    std::vector<std::string> as;
};

bool operator==(const Parameter& lhs, const Parameter& rhs);

}// namespace pnnx

#endif//OPENXAE_IR_H
