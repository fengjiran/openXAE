//
// Created by richard on 6/14/24.
//

#include "Graph.h"
#include "storezip.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace pnnx {

static std::string SanitizeIdentifier(const std::string& s) {
    std::string ss = s;
    for (char& c: ss) {
        if (c == '.' || c == ':' || c == '/') {
            c = '_';
        }
    }
    return ss;
}

static std::string ExpandExpression(const std::shared_ptr<Operator>& op) {
    std::string expr = op->GetParameters().at("expr")->toValue<std::string>();
    // split into tokens
    std::vector<std::string> tokens;
    {
        std::string t;
        for (char ch: expr) {
            if (ch == '[') {// list
                t += ch;
                tokens.push_back(t);
                t.clear();
            } else if (ch == '(' || ch == ')' || ch == ',' || ch == ']') {
                if (!t.empty()) {
                    tokens.push_back(t);
                    t.clear();
                }
            } else {
                t += ch;
            }
        }

        if (!t.empty()) {
            tokens.push_back(t);
        }
    }

    // scan and stack
    std::stack<std::string> stk;
    for (int i = (int) tokens.size() - 1; i >= 0; --i) {
        const auto& t = tokens[i];
        if (t == "size") {
            std::string a = stk.top();
            stk.pop();
            if (stk.empty()) {
                std::string r = a + ".shape";
                stk.push(r);
            } else {
                std::string b = stk.top();
                stk.pop();

                std::string r = a + ".size(" + b + ")";
                stk.push(r);
            }
        } else if (t == "int" ||
                   t == "abs" ||
                   t == "acos" ||
                   t == "acosh" ||
                   t == "asin" ||
                   t == "asinh" ||
                   t == "atan" ||
                   t == "atanh" ||
                   t == "ceil" ||
                   t == "cos" ||
                   t == "cosh" ||
                   t == "exp" ||
                   t == "floor" ||
                   t == "log" ||
                   t == "log10" ||
                   t == "neg" ||
                   t == "reciprocal" ||
                   t == "round" ||
                   t == "rsqrt" ||
                   t == "sign" ||
                   t == "sin" ||
                   t == "sinh" ||
                   t == "sqrt" ||
                   t == "square" ||
                   t == "tan" ||
                   t == "tanh" ||
                   t == "trunc" ||
                   t == "torch.bool" ||
                   t == "torch.float" ||
                   t == "torch.long") {
            std::string unaryOp = t;
            if (t == "int") unaryOp = "int";
            if (t == "abs") unaryOp = "torch.abs";
            if (t == "acos") unaryOp = "torch.acos";
            if (t == "acosh") unaryOp = "torch.acosh";
            if (t == "asin") unaryOp = "torch.asin";
            if (t == "asinh") unaryOp = "torch.asinh";
            if (t == "atan") unaryOp = "torch.atan";
            if (t == "atanh") unaryOp = "torch.atanh";
            if (t == "ceil") unaryOp = "torch.ceil";
            if (t == "cos") unaryOp = "torch.cos";
            if (t == "cosh") unaryOp = "torch.cosh";
            if (t == "exp") unaryOp = "torch.exp";
            if (t == "floor") unaryOp = "torch.floor";
            if (t == "log") unaryOp = "torch.log";
            if (t == "log10") unaryOp = "torch.log10";
            if (t == "neg") unaryOp = "-";
            if (t == "reciprocal") unaryOp = "torch.reciprocal";
            if (t == "round") unaryOp = "torch.round";
            if (t == "rsqrt") unaryOp = "torch.rsqrt";
            if (t == "sign") unaryOp = "torch.sign";
            if (t == "sin") unaryOp = "torch.sin";
            if (t == "sinh") unaryOp = "torch.sinh";
            if (t == "sqrt") unaryOp = "torch.sqrt";
            if (t == "square") unaryOp = "torch.square";
            if (t == "tan") unaryOp = "torch.tan";
            if (t == "tanh") unaryOp = "torch.tanh";
            if (t == "trunc") unaryOp = "torch.trunc";

            std::string a = stk.top();
            stk.pop();

            std::string r = unaryOp + "(" + a + ")";
            stk.push(r);
        } else if (t == "atan2" ||
                   t == "fmod" ||
                   t == "max" ||
                   t == "maximum" ||
                   t == "min" ||
                   t == "minimum" ||
                   t == "pow") {
            std::string binaryOp;
            if (t == "atan2") binaryOp = "torch.atan2";
            if (t == "fmod") binaryOp = "torch.fmod";
            if (t == "max") binaryOp = "torch.max";
            if (t == "maximum") binaryOp = "torch.maximum";
            if (t == "min") binaryOp = "torch.min";
            if (t == "minimum") binaryOp = "torch.minimum";
            if (t == "pow") binaryOp = "torch.pow";

            std::string a = stk.top();
            stk.pop();
            std::string b = stk.top();
            stk.pop();

            std::string r = binaryOp + "(" + a + ", " + b + ")";
            stk.push(r);
        } else if (t == "add" ||
                   t == "sub" ||
                   t == "mul" ||
                   t == "div" ||
                   t == "floor_divide" ||
                   t == "remainder" ||
                   t == "and" ||
                   t == "or" ||
                   t == "xor" ||
                   t == "lshift" ||
                   t == "rshift") {
            std::string binaryOp;
            if (t == "add") binaryOp = "+";
            if (t == "sub") binaryOp = "-";
            if (t == "mul") binaryOp = "*";
            if (t == "div") binaryOp = "/";
            if (t == "floor_divide") binaryOp = "//";
            if (t == "remainder") binaryOp = "%";
            if (t == "and") binaryOp = "&";
            if (t == "or") binaryOp = "|";
            if (t == "xor") binaryOp = "^";
            if (t == "lshift") binaryOp = "<<";
            if (t == "rshift") binaryOp = ">>";

            std::string a = stk.top();
            stk.pop();
            std::string b = stk.top();
            stk.pop();

            std::string r = "(" + a + " " + binaryOp + " " + b + ")";
            stk.push(r);
        } else if (t == "[") {// list
            std::vector<std::string> elements;
            while (!stk.empty()) {
                std::string a = stk.top();
                stk.pop();
                elements.push_back(a);
            }

            std::string r = "[";
            size_t size = elements.size();
            for (const auto& elem: elements) {
                r += (elem + (--size ? ", " : ""));
            }
            r += "]";

            stk.push(r);
        } else if (t[0] == '@') {
            int input_index = std::stoi(t.substr(1));
            std::string varid = std::string("v_") + SanitizeIdentifier(op->GetInputOperands()[input_index]->name());
            stk.push(varid);
        } else {
            // literal
            if (t[t.size() - 1] == 'j') {
                // complex
                std::string r = std::string("(") + t + ")";
                stk.push(r);
            } else {
                stk.push(t);
            }
        }
    }

    std::string r = stk.top();
    stk.pop();

    return r;
}

static std::string MakeSliceExpression(const std::shared_ptr<Operator>& op) {
    const auto& params = op->GetParameters();
    std::vector<int> dims;
    if (op->HasParam("dims")) {
        dims = params.at("dims")->toValue<std::vector<int>>();
    } else {
        dims.push_back(params.at("dim")->toValue<int>());
    }

    std::string pr;
    std::string nr;
    int lastDim = -1;
    const int ndim = (int) dims.size();
    for (int i = 0; i < ndim; ++i) {
        int dim = dims[i];
        std::string& r = dim < 0 ? nr : pr;
        for (int j = lastDim + 1; j < dim; ++j) {
            r += ":,";
        }
        lastDim = dim;

        bool isSelect = false;
        if (op->HasParam("select")) {
            int select = params.at("select")->toValue<int>();
            if (select != std::numeric_limits<int>::max()) {
                r += std::to_string(select);
                isSelect = true;
            }
        }

        if (op->HasParam("selects")) {
            std::vector<int> selects = params.at("selects")->toValue<std::vector<int>>();
            int select = selects[i];
            if (select != std::numeric_limits<int>::max()) {
                r += std::to_string(select);
                isSelect = true;
            }
        }

        if (op->HasInput("select")) {
            r += "v_" + SanitizeIdentifier(op->GetNamedInput("select")->name());
            isSelect = true;
        }

        if (op->HasInput("selects")) {
            // must be pnnx.SliceIndexes
            const auto& opSliceIndexes = op->GetNamedInput("selects")->GetProducer();
            const std::string& index = opSliceIndexes->GetParameters().at("indexes")->toValue<std::vector<std::string>>()[i];
            if (index[0] == '@') {
                int selecti = std::stoi(index.substr(1));
                r += "v_" + SanitizeIdentifier(opSliceIndexes->GetInputOperands()[selecti]->name());
                isSelect = true;
            } else {
                int select = std::stoi(index);
                if (select != std::numeric_limits<int>::max()) {
                    r += std::to_string(select);
                    isSelect = true;
                }
            }
        }

        if (isSelect) {
            if (i + 1 != ndim) {
                r += ',';
            }
            continue;
        }

        if (op->HasParam("start")) {
            int start = params.at("start")->toValue<int>();
            if (start != 0) {
                r += std::to_string(start);
            }
        } else if (op->HasParam("starts")) {
            std::vector<int> starts = params.at("starts")->toValue<std::vector<int>>();
            int start = starts[i];
            if (start != 0) {
                r += std::to_string(start);
            }
        } else if (op->HasInput("start")) {
            r += "v_" + SanitizeIdentifier(op->GetNamedInput("start")->name());
        } else {
            // must be pnnx.SliceIndexes
            const auto& opSliceIndexes = op->GetNamedInput("starts")->GetProducer();
            const std::string& index = opSliceIndexes->GetParameters().at("indexes")->toValue<std::vector<std::string>>()[i];
            if (index[0] == '@') {
                int starti = std::stoi(index.substr(1));
                r += "v_" + SanitizeIdentifier(opSliceIndexes->GetInputOperands()[starti]->name());
            } else {
                int start = std::stoi(index);
                if (start != 0) {
                    r += std::to_string(start);
                }
            }
        }

        r += ':';

        if (op->HasParam("end")) {
            int end = params.at("end")->toValue<int>();
            if (end != std::numeric_limits<int>::max()) {
                r += std::to_string(end);
            }
        } else if (op->HasParam("ends")) {
            std::vector<int> ends = params.at("ends")->toValue<std::vector<int>>();
            int end = ends[i];
            if (end != std::numeric_limits<int>::max()) {
                r += std::to_string(end);
            }
        } else if (op->HasInput("end")) {
            r += "v_" + SanitizeIdentifier(op->GetNamedInput("end")->name());
        } else {
            // must be pnnx.SliceIndexes
            const auto& opSliceIndexes = op->GetNamedInput("ends")->GetProducer();
            const std::string& index = opSliceIndexes->GetParameters().at("indexes")->toValue<std::vector<std::string>>()[i];
            if (index[0] == '@') {
                int endi = std::stoi(index.substr(1));
                r += "v_" + SanitizeIdentifier(opSliceIndexes->GetInputOperands()[endi]->name());
            } else {
                int end = std::stoi(index);
                if (end != std::numeric_limits<int>::max()) {
                    r += std::to_string(end);
                }
            }
        }

        if (op->HasParam("step")) {
            int step = params.at("step")->toValue<int>();
            if (step != 1) {
                r += ':';
                r += std::to_string(step);
            }
        } else if (op->HasParam("steps")) {
            std::vector<int> steps = params.at("steps")->toValue<std::vector<int>>();
            int step = steps[i];
            if (step != 1) {
                r += ':';
                r += std::to_string(step);
            }
        } else if (op->HasInput("step")) {
            r += ':';
            r += "v_" + SanitizeIdentifier(op->GetNamedInput("step")->name());
        } else {
            // must be pnnx.SliceIndexes
            const auto& opSliceIndexes = op->GetNamedInput("steps")->GetProducer();
            const std::string& index = opSliceIndexes->GetParameters().at("indexes")->toValue<std::vector<std::string>>()[i];
            if (index[0] == '@') {
                int stepi = std::stoi(index.substr(1));
                r += ':';
                r += "v_" + SanitizeIdentifier(opSliceIndexes->GetInputOperands()[stepi]->name());
            } else {
                int step = std::stoi(index);
                if (step != 1) {
                    r += ':';
                    r += std::to_string(step);
                }
            }
        }

        if (i + 1 != ndim) {
            r += ',';
        }
    }

    if (!pr.empty() && !nr.empty()) {
        return pr + "...," + nr;
    }

    if (pr.empty() && !nr.empty()) {
        return "...," + nr;
    }

    return pr + nr;
}

static std::string MakeIndexExpression(const std::shared_ptr<Operator>& op) {
    std::cerr << "make index expression: " << op->name() << std::endl;
    std::string idxExpr = op->GetParameters().at("expr")->toValue<std::string>();
    idxExpr = idxExpr.substr(1, idxExpr.size() - 2);// strip out-most [] pair

    // None,None,  ->  ...,
    bool leadingNone = false;
    while (idxExpr.substr(0, 5) == "None,") {
        leadingNone = true;
        idxExpr = idxExpr.substr(5);
    }

    if (leadingNone) {
        idxExpr = "...," + idxExpr;
    }
    return idxExpr;
}

static Parameter CreateParameterFromString(const std::string& value) {
    // string type
    if (value.find('%') != std::string::npos) {
        return Parameter(value);
    }

    // null type
    if (value == "None" || value == "()" || value == "[]") {
        return {};
    }

    // bool type
    if (value == "True" || value == "False") {
        return Parameter(value == "True");
    }

    // array
    if (value[0] == '(' || value[0] == '[') {
        bool isArrayInt = false;
        bool isArrayFloat = false;
        bool isArrayString = false;

        std::vector<int> pArrayInt;
        std::vector<float> pArrayFloat;
        std::vector<std::string> pArrayString;

        std::string lc = value.substr(1, value.size() - 2);
        std::istringstream lcss(lc);
        while (!lcss.eof()) {
            std::string elem;
            std::getline(lcss, elem, ',');
            if ((elem[0] != '-' && (elem[0] < '0' || elem[0] > '9')) || (elem[0] == '-' && (elem[1] < '0' || elem[1] > '9'))) {
                // array string
                isArrayString = true;
                pArrayString.push_back(elem);
            } else if (elem.find('.') != std::string::npos || elem.find('e') != std::string::npos) {
                // array float
                isArrayFloat = true;
                pArrayFloat.push_back(std::stof(elem));
            } else {
                // array integer
                isArrayInt = true;
                pArrayInt.push_back(std::stoi(elem));
            }
        }

        if (isArrayInt && !isArrayFloat && !isArrayString) {
            return Parameter(pArrayInt);
        }

        if (!isArrayInt && isArrayFloat && !isArrayString) {
            return Parameter(pArrayFloat);
        }

        if (!isArrayInt && !isArrayFloat && isArrayString) {
            return Parameter(pArrayString);
        }

        return {};
    }

    // string
    if ((value[0] != '-' && (value[0] < '0' || value[0] > '9')) || (value[0] == '-' && (value[1] < '0' || value[1] > '9'))) {
        return Parameter(value);
    }

    // float
    if (value.find('.') != std::string::npos || value.find('e') != std::string::npos) {
        return Parameter(std::stof(value));
    }

    // integer
    return Parameter(std::stoi(value));
}

static void LoadParameter(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    op->GetParameters()[key] = std::make_shared<Parameter>(CreateParameterFromString(value));
}

static void LoadInputName(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    op->GetInputNames().resize(op->GetInputOperands().size());
    for (size_t i = 0; i < op->GetInputOperands().size(); ++i) {
        const auto& operand = op->GetInputOperands()[i];
        if (operand->name() == value) {
            op->GetInputNames()[i] = key;
            break;
        }
    }
}

static void LoadOperand(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value) {
    std::shared_ptr<Operand> operand;
    for (const auto& r: op->GetInputOperands()) {
        if (r->name() == key) {
            operand = r;
            break;
        }
    }

    if (!operand) {
        for (const auto& r: op->GetOutputOperands()) {
            if (r->name() == key) {
                operand = r;
                break;
            }
        }
    }

    if (!operand) {
        std::cerr << "operator " << op->name() << " has no such operand! " << key << std::endl;
        return;
    }

    // parse data type, e.g. #input=(1,3,10,10)f32
    std::string str = value.substr(value.find_last_of(')') + 1);
    operand->SetType(String2Type(str));

    // parse shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);
    operand->GetShape().clear();
    while (!lcss.eof()) {
        std::string elem;
        std::getline(lcss, elem, ',');
        if (elem == "?") {
            operand->GetShape().push_back(DimUnknownTag);
        } else if (elem[0] == '%') {
            // this shape is a variable,
            // encode %abc as symbolic tag
            operand->GetShape().push_back(DimVariableTag);
            size_t index = operand->GetShape().size() - 1;
            std::string s = elem.substr(1);
            operand->GetParams()[std::string("__shape__") + std::to_string(index)] = std::make_shared<Parameter>(s);
        } else {
            operand->GetShape().push_back(std::stoi(elem));
        }
    }
}

static void LoadAttribute(const std::shared_ptr<Operator>& op, const std::string& key, const std::string& value, StoreZipReader& szr) {
    // parse attribute data type
    std::string str = value.substr(value.find_last_of(')') + 1);
    DataType type = String2Type(str);
    if (type == DataType::kDataTypeUnknown) {
        return;
    }

    op->GetAttributes()[key] = std::make_shared<Attribute>();
    std::shared_ptr<Attribute>& attr = op->GetAttributes()[key];

    attr->SetType(type);

    // parse shape
    std::string lc = value.substr(1, value.find_last_of(')') - 1);
    std::istringstream lcss(lc);
    attr->GetShape().clear();
    while (!lcss.eof()) {
        std::string elem;
        std::getline(lcss, elem, ',');
        attr->GetShape().push_back(std::stoi(elem));
    }

    if (attr->GetShape().empty()) {
        return;
    }

    // parse data
    size_t sizeInByte =
            std::accumulate(attr->GetShape().begin(), attr->GetShape().end(), 1, std::multiplies<>()) * SizeOf(type);

    std::string filename = op->name() + "." + key;
    size_t filesize = szr.get_file_size(filename);
    if (filesize == 0) {
        // no such file
        return;
    }

    if (filesize != sizeInByte) {
        std::cerr << "file size not match, expect " << sizeInByte << " but got " << filesize << std::endl;
    }

    attr->GetRawData().resize(sizeInByte);
    szr.read_file(filename, (char*) attr->GetRawData().data());
}

int Graph::save(const std::string& paramPath, const std::string& binPath) {
    std::ofstream paramFile(paramPath, std::ios::out | std::ios::binary);
    if (!paramFile.is_open()) {
        std::cerr << "param file " << paramPath << " open failed!\n";
        return -1;
    }

    StoreZipWriter szw;
    if (szw.open(binPath) != 0) {
        std::cerr << "bin file " << binPath << " open failed!\n";
        return -1;
    }

    // magic number
    paramFile << "7767517" << std::endl;

    // op number and operand number
    paramFile << static_cast<int>(ops_.size()) << " "
              << static_cast<int>(operands_.size()) << std::endl;

    // dump op info
    for (const auto& op: ops_) {
        paramFile << std::left << std::setw(24) << op->type() << " "
                  << std::left << std::setw(24) << op->name() << " "
                  << static_cast<int>(op->GetInputOperands().size()) << " "
                  << static_cast<int>(op->GetOutputOperands().size());

        // dump op input operand name
        for (const auto& operand: op->GetInputOperands()) {
            paramFile << " " << operand->name();
        }

        // dump op output operand name
        for (const auto& operand: op->GetOutputOperands()) {
            paramFile << " " << operand->name();
        }

        // dump op param info
        for (const auto& it: op->GetParameters()) {
            paramFile << " " << it.first << "=" << it.second->toString();
        }

        // dump op attrs info
        for (const auto& it: op->GetAttributes()) {
            paramFile << " @" << it.first << "=(";
            const auto& attr = it.second;
            size_t size = attr->GetShape().size();
            for (const auto& x: attr->GetShape()) {
                paramFile << x << (--size ? "," : "");
            }
            paramFile << ")" << DataType2String(attr->type());

            std::string filename = op->name() + "." + it.first;
            szw.write_file(filename, attr->GetRawData().data(), attr->GetRawData().size());
        }

        if (op->GetInputNames().size() == op->GetInputOperands().size()) {
            for (size_t i = 0; i < op->GetInputOperands().size(); ++i) {
                if (op->GetInputNames()[i].empty()) {
                    continue;
                }
                const auto& operand = op->GetInputOperands()[i];
                paramFile << " $" << op->GetInputNames()[i] << "=" << operand->name();
            }
        }

        // dump input operands
        for (const auto& operand: op->GetInputOperands()) {
            if (operand->GetShape().empty()) {
                continue;
            }

            paramFile << " #" << operand->name() << "=(";
            size_t size = operand->GetShape().size();
            for (const auto& x: operand->GetShape()) {
                if (x == DimUnknownTag) {
                    paramFile << (--size ? "?," : "?");
                } else if (x == DimVariableTag) {
                    // %abc, shape variable
                    size_t idx = operand->GetShape().size() - size;
                    std::string key = "__shape__" + std::to_string(idx);
                    paramFile << "%" << operand->GetParams()[key]->toString() << (--size ? "," : "");
                } else {
                    paramFile << x << (--size ? "," : "");
                }
            }

            paramFile << ")" << DataType2String(operand->type());
        }

        // dump output operands
        for (const auto& operand: op->GetOutputOperands()) {
            if (operand->GetShape().empty()) {
                continue;
            }

            paramFile << " #" << operand->name() << "=(";
            size_t size = operand->GetShape().size();
            for (const auto& x: operand->GetShape()) {
                if (x == DimUnknownTag) {
                    paramFile << (--size ? "?," : "?");
                } else if (x == DimVariableTag) {
                    // %abc, shape variable
                    size_t idx = operand->GetShape().size() - size;
                    std::string key = "__shape__" + std::to_string(idx);
                    paramFile << "%" << operand->GetParams()[key]->toString() << (--size ? "," : "");
                } else {
                    paramFile << x << (--size ? "," : "");
                }
            }

            paramFile << ")" << DataType2String(operand->type());
        }
        paramFile << std::endl;
    }
    paramFile.close();
    return 0;
}

int Graph::load(const std::string& paramPath, const std::string& binPath) {
    std::ifstream paramFile(paramPath, std::ios::in | std::ios::binary);
    if (!paramFile.is_open()) {
        std::cerr << "param file " << paramPath << " open failed!\n";
        return -1;
    }

    StoreZipReader szr;
    if (szr.open(binPath) != 0) {
        std::cerr << "bin file " << binPath << " open failed!\n";
        return -1;
    }

    // parse the first line, magic number
    int magicNum = 0;
    {
        std::string line;
        std::getline(paramFile, line);
        std::istringstream iss(line);
        iss >> magicNum;
    }

    // parse the second line, operator number and operand number
    int operatorNum = 0;
    int operandNum = 0;
    {
        std::string line;
        std::getline(paramFile, line);
        std::istringstream iss(line);
        iss >> operatorNum >> operandNum;
    }

    for (int i = 0; i < operatorNum; ++i) {
        std::string line;
        std::getline(paramFile, line);
        std::istringstream iss(line);

        std::string type;
        std::string name;
        int inputOperandNum = 0;
        int outputOperandNum = 0;

        iss >> type >> name >> inputOperandNum >> outputOperandNum;

        const auto& op = CreateOperator(type, name);
        for (int j = 0; j < inputOperandNum; ++j) {
            std::string operandName;
            iss >> operandName;
            const auto& r = GetOperand(operandName);
            r->AddConsumer(op);
            op->AddInputOperand(r);
        }

        for (int j = 0; j < outputOperandNum; ++j) {
            std::string operandName;
            iss >> operandName;
            const auto& r = CreateOperand(operandName);
            r->SetProducer(op);
            op->AddOutputOperand(r);
        }

        while (!iss.eof()) {
            std::string param;
            iss >> param;

            std::string key;
            std::string value;
            std::istringstream pss(param);
            std::getline(pss, key, '=');
            std::getline(pss, value);
            if (key[0] == '@') {
                // load attribute raw data, shape and data type
                LoadAttribute(op, key.substr(1), value, szr);
            } else if (key[0] == '$') {
                // operand input key
                LoadInputName(op, key.substr(1), value);
            } else if (key[0] == '#') {
                // load operand shape and data type
                LoadOperand(op, key.substr(1), value);
            } else {
                // load parameter
                LoadParameter(op, key, value);
            }
        }
    }

    return 0;
}

int Graph::python(const std::string& pyPath, const std::string& binPath) {
    std::ofstream pyfp(pyPath, std::ios::out | std::ios::binary);
    if (!pyfp.is_open()) {
        std::cerr << "python file " << pyPath << " open failed!\n";
        return -1;
    }

    pyfp << "import os\n";
    pyfp << "import numpy as np\n";
    pyfp << "import tempfile, zipfile\n";
    pyfp << "import torch\n";
    pyfp << "import torch.nn as nn\n";
    pyfp << "import torch.nn.functional as F\n";
    pyfp << "try:\n";
    pyfp << "    import torchvision\n";
    pyfp << "except:\n";
    pyfp << "    pass\n\n";

    pyfp << "class Model(nn.Module):\n";
    pyfp << "    def __init__(self):\n";
    pyfp << "        super(Model, self).__init__()\n\n";

    // module
    {
        for (const auto& op: GetOperators()) {
            if (op->type().substr(0, 3) != "nn." &&
                op->type().substr(0, 16) != "torchvision.ops.")
                continue;
            pyfp << "        self." << SanitizeIdentifier(op->name()) << " = " << op->type() << "(";
            size_t paramCnt = op->GetParameters().size();
            if (op->type() == "nn.quantized.Conv2d" || op->type() == "nn.quantized.Linear") {
                paramCnt -= 2;// ignore scale and zero_point
            }

            int paramIdx = 0;
            for (const auto& it: op->GetParameters()) {
                if (op->type() == "nn.quantized.Conv2d" || op->type() == "nn.quantized.Linear") {
                    if (it.first == "scale" || it.first == "zero_point") {
                        continue;
                    }
                }

                pyfp << it.first << "=";
                const auto& param = it.second;
                if (param->type() == ParameterType::kParameterUnknown) {
                    pyfp << "None";
                }

                if (param->type() == ParameterType::kParameterBool) {
                    if (param->toValue<bool>()) {
                        pyfp << "True";
                    } else {
                        pyfp << "False";
                    }
                }

                if (param->type() == ParameterType::kParameterInt) {
                    pyfp << param->toValue<int>();
                }

                if (param->type() == ParameterType::kParameterFloat) {
                    pyfp << param->toValue<float>();
                }

                if (param->type() == ParameterType::kParameterString) {
                    if (param->toValue<std::string>().substr(0, 6) == "torch.") {
                        pyfp << param->toValue<std::string>();
                    } else {
                        pyfp << "\'" << param->toValue<std::string>() << "\'";
                    }
                }

                if (param->type() == ParameterType::kParameterArrayInt) {
                    pyfp << "(";
                    const size_t size = param->toValue<std::vector<int>>().size();
                    for (size_t i = 0; i < size; ++i) {
                        const auto& elem = param->toValue<std::vector<int>>()[i];
                        if ((op->type() == "nn.AdaptiveAvgPool2d" ||
                             op->type() == "nn.AdaptiveAvgPool3d" ||
                             op->type() == "nn.AdaptiveMaxPool2d" ||
                             op->type() == "nn.AdaptiveMaxPool3d") &&
                            it.first == "output_size" && elem == 0) {
                            pyfp << "None";
                        } else {
                            pyfp << elem;
                        }

                        if (i + 1 != size || size == 1) {
                            pyfp << ",";
                        }
                    }
                    pyfp << ")";
                }

                if (param->type() == ParameterType::kParameterArrayFloat) {
                    pyfp << "(";
                    const size_t size = param->toValue<std::vector<float>>().size();
                    for (size_t i = 0; i < size; ++i) {
                        const auto& elem = param->toValue<std::vector<float>>()[i];
                        pyfp << elem;
                        if (i + 1 != size || size == 1) {
                            pyfp << ",";
                        }
                    }
                    pyfp << ")";
                }

                if (param->type() == ParameterType::kParameterArrayString) {
                    pyfp << "(";
                    const size_t size = param->toValue<std::string>().size();
                    for (size_t i = 0; i < size; ++i) {
                        const auto& elem = param->toValue<std::vector<std::string>>()[i];
                        if (elem.substr(0, 6) == "torch.") {
                            pyfp << elem;
                        } else {
                            pyfp << "\'" << elem << "\'";
                        }
                        if (i + 1 != size || size == 1) {
                            pyfp << ",";
                        }
                    }
                    pyfp << ")";
                }
                paramIdx++;
                if (paramIdx != paramCnt) {
                    pyfp << ", ";
                }
            }
            pyfp << ")\n";
        }
    }

    pyfp << "\n";

    // load weight
    {
        pyfp << "        archive = zipfile.ZipFile(" << "\'" << binPath << "\'" << ", \'r\')\n";
        for (const auto& op: GetOperators()) {
            if (op->type().substr(0, 3) != "nn." &&
                op->type().substr(0, 16) != "torchvision.ops.")
                continue;

            if (op->type() == "nn.quantized.Conv2d" || op->type() == "nn.quantized.Linear") {
                for (const auto& it: op->GetAttributes()) {
                    if (it.first == "weight" || it.first == "bias") {
                        pyfp << "        self_" + SanitizeIdentifier(op->name()) + "_" + it.first
                             << " = self.load_pnnx_bin_as_parameter(archive, \'" + op->name() + "." + it.first + "\', (";
                    } else {
                        // unknown attr
                        continue;
                    }
                    const auto& attr = it.second;
                    size_t size = attr->GetShape().size();
                    for (const auto& elem: attr->GetShape()) {
                        pyfp << elem << (--size ? "," : "");
                    }
                    pyfp << "), \'" + DataType2NumpyString(attr->type()) + "\', requires_grad=False)\n";
                }

                pyfp << "        self." + SanitizeIdentifier(op->name()) + ".set_weight_bias(self_"
                     << SanitizeIdentifier(op->name()) + "_weight, self_" + SanitizeIdentifier(op->name()) + "_bias)\n";
                pyfp << "        self." + SanitizeIdentifier(op->name()) + ".scale = "
                     << op->GetParameters()["scale"]->toValue<float>() << std::endl;
                pyfp << "        self." + SanitizeIdentifier(op->name()) + ".zero_point = "
                     << op->GetParameters()["zero_point"]->toValue<int>() << std::endl;
                continue;
            }

            for (const auto& it: op->GetAttributes()) {
                if (it.first == "running_mean" || it.first == "running_var") {
                    pyfp << "        self." + SanitizeIdentifier(op->name()) + "." + it.first
                         << " = self.load_pnnx_bin_as_tensor(archive, \'" + op->name() + "." + it.first + "\', (";
                } else {
                    pyfp << "        self." + SanitizeIdentifier(op->name()) + "." + it.first
                         << " = self.load_pnnx_bin_as_parameter(archive, \'" + op->name() + "." + it.first + "\', (";
                }
                const auto& attr = it.second;
                size_t size = attr->GetShape().size();
                for (const auto& elem: attr->GetShape()) {
                    pyfp << elem << (--size ? "," : "");
                }

                if (attr->type() == DataType::kDataTypeFloat32 ||
                    attr->type() == DataType::kDataTypeFloat64 ||
                    attr->type() == DataType::kDataTypeFloat16) {
                    pyfp << "), \'" + DataType2NumpyString(attr->type()) + "\')\n";
                } else {
                    pyfp << "), \'" + DataType2NumpyString(attr->type()) + "\', requires_grad=False)\n";
                }
            }
        }

        for (const auto& op: GetOperators()) {
            if (op->type() != "pnnx.Attribute") {
                continue;
            }

            const auto& key = op->GetAttributes().begin()->first;
            const auto& attr = op->GetAttributes().begin()->second;
            bool is_running_mean_var = false;
            {
                const auto& r = op->GetOutputOperands()[0];
                if (r->GetConsumers().size() == 1) {
                    const auto& op2 = r->GetConsumers()[0];
                    if (op2->type() == "F.batch_norm" || op2->type() == "F.instance_norm") {
                        if (r == op2->GetInputOperands()[1] || r == op2->GetInputOperands()[2]) {
                            is_running_mean_var = true;
                        }
                    }
                }
            }

            bool is_empty = false;
            for (const auto& elem: attr->GetShape()) {
                if (elem == 0) {
                    is_empty = true;
                }
            }

            if (is_empty) {
                pyfp << "        self." + SanitizeIdentifier(op->name()) + "_" + SanitizeIdentifier(key)
                     << " = torch.from_numpy(np.empty((";
                for (const auto& elem: attr->GetShape()) {
                    pyfp << elem << ",";
                }
                pyfp << "), dtype=\'" + DataType2NumpyString(attr->type()) + "\'))\n";
            } else {
                if (is_running_mean_var) {
                    pyfp << "        self." + SanitizeIdentifier(op->name()) + "_" + SanitizeIdentifier(key)
                         << " = self.load_pnnx_bin_as_tensor(archive, \'" + op->name() + "." + key + "\', (";
                } else {
                    pyfp << "        self." + SanitizeIdentifier(op->name()) + "." + SanitizeIdentifier(key)
                         << " = self.load_pnnx_bin_as_parameter(archive, \'" + op->name() + "." + key + "\', (";
                }
                for (const auto& elem: attr->GetShape()) {
                    pyfp << elem << ",";
                }
                if (attr->type() == DataType::kDataTypeFloat32 ||
                    attr->type() == DataType::kDataTypeFloat64 ||
                    attr->type() == DataType::kDataTypeFloat16) {
                    pyfp << "), \'" + DataType2NumpyString(attr->type()) + "\')\n";
                } else {
                    pyfp << "), \'" + DataType2NumpyString(attr->type()) + "\', requires_grad=False)\n";
                }
            }
        }

        pyfp << "        archive.close()\n";
    }

    pyfp << std::endl;

    // utility function
    {
        pyfp << "    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):\n";
        pyfp << "        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)\n";
        pyfp << "\n";
        pyfp << "    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):\n";
        pyfp << "        fd, tmppath = tempfile.mkstemp()\n";
        pyfp << "        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:\n";
        pyfp << "            tmpf.write(keyfile.read())\n";
        pyfp << "        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()\n";
        pyfp << "        os.remove(tmppath)\n";
        pyfp << "        return torch.from_numpy(m)\n";
    }

    pyfp << "\n";

    // def forward
    {
        pyfp << "    def forward(self";
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Input") {
                pyfp << ", v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name());
            }
        }
        pyfp << "):\n";
    }

    // forward body
    {
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Input" || op->type() == "pnnx.Output") {
                continue;
            }

            if (op->type() == "pnnx.SliceIndexes") {
                continue;
            }

            pyfp << "        ";

            if (op->type() == "pnnx.Expression") {
                // expr
                size_t size = op->GetOutputOperands().size();
                for (const auto& elem: op->GetOutputOperands()) {
                    pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                }

                std::string expr = ExpandExpression(op);
                pyfp << " = " << expr << std::endl;
            } else if (op->type() == "pnnx.Attribute") {
                const auto& key = op->GetAttributes().begin()->first;
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name()) << " = self."
                     << SanitizeIdentifier(op->name()) << "_" << SanitizeIdentifier(key) << std::endl;
            } else if (op->type() == "Tensor.slice") {
                // slice expr
                std::string expr = MakeSliceExpression(op);
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                     << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << "[" << expr << "]\n";
            } else if (op->type() == "Tensor.slice_copy") {
                std::string expr = MakeSliceExpression(op);
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                     << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << std::endl;
                pyfp << "        v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                     << "[" << expr << "]" << " = v_" << SanitizeIdentifier(op->GetInputOperands()[1]->name());
            } else if (op->type() == "Tensor.index") {
                // index expr
                if (op->GetInputOperands().size() == 2) {
                    std::string expr = ExpandExpression(op->GetInputOperands()[1]->GetProducer());
                    pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                         << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << "[" << expr << "]\n";
                } else {
                    std::string expr = MakeIndexExpression(op);
                    pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                         << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << "[" << expr << "]\n";
                }
            } else if (op->type() == "Tensor.expand") {
                // expand
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                     << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << "." << op->type().substr(7) << "(";
                if (op->GetInputOperands().size() == 2) {
                    pyfp << "*v_" << SanitizeIdentifier(op->GetInputOperands()[1]->name());
                } else {
                    const auto& shape = op->GetParameters().at("shape")->toValue<std::vector<int>>();
                    size_t size = shape.size();
                    for (const auto& elem: shape) {
                        pyfp << elem << (--size ? ", " : "");
                    }
                }
                pyfp << ")\n";
            } else if (op->type() == "Tensor.view" || op->type() == "Tensor.reshape") {
                // view reshape
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                     << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << "." << op->type().substr(7) << "(";
                if (op->GetInputOperands().size() == 2) {
                    pyfp << "*v_" << SanitizeIdentifier(op->GetInputOperands()[1]->name());
                } else {
                    const auto& shape = op->GetParameters().at("shape")->toValue<std::vector<int>>();
                    size_t size = shape.size();
                    for (const auto& elem: shape) {
                        pyfp << elem << (--size ? ", " : "");
                    }
                }
                pyfp << ")\n";
            } else if (op->type() == "Tensor.repeat") {
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                     << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << "." << op->type().substr(7) << "(";
                if (op->GetInputOperands().size() == 2) {
                    pyfp << "*v_" << SanitizeIdentifier(op->GetInputOperands()[1]->name());
                } else {
                    const auto& shape = op->GetParameters().at("sizes")->toValue<std::vector<int>>();
                    size_t size = shape.size();
                    for (const auto& elem: shape) {
                        pyfp << elem << (--size ? ", " : "");
                    }
                }
                pyfp << ")\n";
            } else if (op->type() == "torch.cat" || op->type() == "torch.stack") {
                // cat
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name()) << " = " << op->type() << "(";
                if (op->GetInputOperands().size() == 1) {
                    pyfp << "v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name());
                } else {
                    pyfp << "(";
                    size_t size = op->GetInputOperands().size();
                    for (const auto& elem: op->GetInputOperands()) {
                        pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                    }
                    pyfp << ")";
                }
                pyfp << ", dim=" << op->GetParameters().at("dim")->toValue<int>();
                pyfp << ")\n";
            } else if (op->type() == "torch.einsum") {
                // einsum
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name()) << " = " << op->type() << "(";
                pyfp << "\'" << op->GetParameters().at("equation")->toValue<std::string>() << "\'";
                for (const auto& elem: op->GetInputOperands()) {
                    pyfp << ", v_" << SanitizeIdentifier(elem->name());
                }
                pyfp << ")\n";
            } else if (op->type() == "prim::TupleUnpack") {
                size_t size = op->GetOutputOperands().size();
                for (const auto& elem: op->GetOutputOperands()) {
                    pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                }
                pyfp << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << std::endl;
            } else if (op->type() == "prim::TupleConstruct") {
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name()) << " = (";
                for (const auto& elem: op->GetInputOperands()) {
                    pyfp << "v_" << SanitizeIdentifier(elem->name()) << ", ";
                }
                pyfp << ")\n";
            } else if (op->type() == "prim::ListUnpack") {
                size_t size = op->GetOutputOperands().size();
                for (const auto& elem: op->GetOutputOperands()) {
                    pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                }
                pyfp << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << std::endl;
            } else if (op->type() == "prim::ListConstruct") {
                pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name()) << " = [";
                size_t size = op->GetInputOperands().size();
                for (const auto& elem: op->GetInputOperands()) {
                    pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                }
                pyfp << "]\n";
            } else if (op->type() == "nn.GRU" || op->type() == "nn.RNN") {
                if (op->GetOutputOperands().size() == 1) {
                    pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name()) << ", _";
                } else {
                    pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                         << ", v_" << SanitizeIdentifier(op->GetOutputOperands()[1]->name());
                }
                pyfp << " = self." << SanitizeIdentifier(op->name()) << "(";
                pyfp << "v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name());

                if (op->GetInputOperands().size() == 2) {
                    pyfp << ", v_" << SanitizeIdentifier(op->GetInputOperands()[1]->name());
                }
                pyfp << ")\n";
            } else if (op->type() == "nn.LSTM") {
                if (op->GetOutputOperands().size() == 1) {
                    pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name()) << ", _";
                } else {
                    pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name())
                         << ", (v_" << SanitizeIdentifier(op->GetOutputOperands()[1]->name())
                         << ", v_" << SanitizeIdentifier(op->GetOutputOperands()[2]->name()) << ")";
                }
                pyfp << " = self." << SanitizeIdentifier(op->name()) << "(";
                pyfp << "v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name());

                if (op->GetInputOperands().size() == 3) {
                    pyfp << ", (v_" << SanitizeIdentifier(op->GetInputOperands()[1]->name())
                         << ", v_" << SanitizeIdentifier(op->GetInputOperands()[2]->name()) << ")";
                }
                pyfp << ")\n";
            } else if (op->type() == "nn.MultiheadAttention") {
                bool need_weights = true;
                if (op->GetOutputOperands().size() == 1) {
                    pyfp << "v_" << SanitizeIdentifier(op->GetOutputOperands()[0]->name()) << ", _";
                    need_weights = false;
                } else {
                    size_t size = op->GetOutputOperands().size();
                    for (const auto& elem: op->GetOutputOperands()) {
                        pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                    }
                }
                pyfp << " = self." << SanitizeIdentifier(op->name()) << "(";

                if (op->GetInputOperands().size() == 1) {
                    std::string in0 = SanitizeIdentifier(op->GetInputOperands()[0]->name());
                    pyfp << "v_" << in0 << ", v_" << in0 << ", v_" << in0;
                } else if (op->GetInputOperands().size() == 2) {
                    std::string in0 = SanitizeIdentifier(op->GetInputOperands()[0]->name());
                    std::string in1 = SanitizeIdentifier(op->GetInputOperands()[1]->name());

                    if (op->GetInputNames().size() == 2 && op->GetInputNames()[1] == "attn_mask") {
                        pyfp << "v_" << in0 << ", v_" << in0 << ", v_" << in0 << ", attn_mask=v_" << in1;
                    } else {
                        pyfp << "v_" << in0 << ", v_" << in1 << ", v_" << in1;
                    }
                } else if (op->GetInputOperands().size() == 3) {
                    std::string in0 = SanitizeIdentifier(op->GetInputOperands()[0]->name());
                    std::string in1 = SanitizeIdentifier(op->GetInputOperands()[1]->name());
                    std::string in2 = SanitizeIdentifier(op->GetInputOperands()[2]->name());

                    if (op->GetInputNames().size() == 3 && op->GetInputNames()[2] == "attn_mask") {
                        pyfp << "v_" << in0 << ", v_" << in1 << ", v_" << in1 << ", attn_mask=v_" << in2;
                    } else {
                        pyfp << "v_" << in0 << ", v_" << in1 << ", v_" << in2;
                    }
                } else if (op->GetInputOperands().size() == 4) {
                    std::string in0 = SanitizeIdentifier(op->GetInputOperands()[0]->name());
                    std::string in1 = SanitizeIdentifier(op->GetInputOperands()[1]->name());
                    std::string in2 = SanitizeIdentifier(op->GetInputOperands()[2]->name());
                    std::string in3 = SanitizeIdentifier(op->GetInputOperands()[3]->name());

                    if (op->GetInputNames().size() == 4 && op->GetInputNames()[3] == "attn_mask") {
                        pyfp << "v_" << in0 << ", v_" << in1 << ", v_" << in2 << ", attn_mask=v_" << in3;
                    } else {
                        pyfp << "v_" << in0 << ", v_" << in1 << ", v_" << in2 << ", v_" << in3;
                    }
                } else {
                    size_t size = op->GetInputOperands().size();
                    for (const auto& elem: op->GetInputOperands()) {
                        pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                    }
                }

                if (need_weights) {
                    pyfp << ", need_weights=True";
                } else {
                    pyfp << ", need_weights=False";
                }
                pyfp << ")\n";
            } else if (op->type().substr(0, 3) == "nn." || op->type().substr(0, 16) == "torchvision.ops.") {
                // self.xxx
                size_t size = op->GetOutputOperands().size();
                for (const auto& elem: op->GetOutputOperands()) {
                    pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                }
                pyfp << " = self." << SanitizeIdentifier(op->name()) << "(";

                size = op->GetInputOperands().size();
                for (const auto& elem: op->GetInputOperands()) {
                    pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                }
                pyfp << ")\n";
            } else {
                if (op->type().find("::") == std::string::npos && op->type().find('.') == std::string::npos) {
                    pyfp << "todo " << op->type() << std::endl;
                }

                // direct
                size_t size = op->GetOutputOperands().size();
                for (const auto& elem: op->GetOutputOperands()) {
                    pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                }

                if (op->type().substr(0, 7) == "Tensor.") {
                    if (op->type() == "Tensor.fill") {
                        pyfp << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << ".fill_(";
                    } else {
                        pyfp << " = v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name()) << "." << op->type().substr(0, 7) << "(";
                    }

                    if (op->GetInputNames().size() == op->GetInputOperands().size()) {
                        for (size_t i = 1; i < op->GetInputOperands().size(); ++i) {
                            if (!op->GetInputNames()[i].empty()) {
                                continue;
                            }
                            pyfp << "v_" << SanitizeIdentifier(op->GetInputOperands()[i]->name()) << ", ";
                        }

                        for (size_t i = 1; i < op->GetInputOperands().size(); ++i) {
                            if (op->GetInputNames()[i].empty()) {
                                continue;
                            }
                            pyfp << op->GetInputNames()[i] << "=v_" << SanitizeIdentifier(op->GetInputOperands()[i]->name()) << ", ";
                        }
                    } else {
                        for (size_t i = 1; i < op->GetInputOperands().size(); ++i) {
                            pyfp << "v_" << SanitizeIdentifier(op->GetInputOperands()[i]->name()) << ", ";
                        }
                    }
                } else {
                    pyfp << " = " << op->type() << "(";
                    if (op->GetInputNames().size() == op->GetInputOperands().size()) {
                        for (size_t i = 0; i < op->GetInputOperands().size(); ++i) {
                            if (!op->GetInputNames()[i].empty()) {
                                continue;
                            }
                            pyfp << "v_" << SanitizeIdentifier(op->GetInputOperands()[i]->name());
                            if (i + 1 != op->GetInputOperands().size()) {
                                pyfp << ", ";
                            }
                        }

                        for (size_t i = 0; i < op->GetInputOperands().size(); ++i) {
                            if (op->GetInputNames()[i].empty()) {
                                continue;
                            }
                            pyfp << op->GetInputNames()[i] << "=v_" << SanitizeIdentifier(op->GetInputOperands()[i]->name());
                            if (i + 1 != op->GetInputOperands().size()) {
                                pyfp << ", ";
                            }
                        }
                    } else {
                        size = op->GetInputOperands().size();
                        for (const auto& elem: op->GetInputOperands()) {
                            pyfp << "v_" << SanitizeIdentifier(elem->name()) << (--size ? ", " : "");
                        }
                    }
                }

                int i = 0;
                for (const auto& it: op->GetParameters()) {
                    if (op->type().substr(0, 7) == "Tensor." && i == 0) {
                        pyfp << it.first << "=";
                    } else if (op->GetInputOperands().empty() && i == 0) {
                        pyfp << it.first << "=";
                    } else {
                        pyfp << ", " << it.first << "=";
                    }
                    ++i;

                    const auto& param = it.second;
                    if (param->type() == ParameterType::kParameterUnknown) {
                        if (op->type() == "Tensor.index_put" && it.first == "values") {
                            pyfp << "torch.tensor(False)";
                        } else {
                            pyfp << "None";
                        }
                    }

                    if (param->type() == ParameterType::kParameterBool) {
                        if (param->toValue<bool>()) {
                            pyfp << "True";
                        } else {
                            pyfp << "False";
                        }
                    }

                    if (param->type() == ParameterType::kParameterInt) {
                        if (op->type() == "Tensor.index_put" && it.first == "values") {
                            pyfp << "torch.tensor(" << param->toValue<int>() << ")";
                        } else {
                            pyfp << param->toValue<int>();
                        }
                    }

                    if (param->type() == ParameterType::kParameterFloat) {
                        if (op->type() == "Tensor.index_put" && it.first == "values") {
                            pyfp << "torch.tensor(" << param->toValue<float>() << ")";
                        } else {
                            pyfp << param->toValue<float>();
                        }
                    }

                    if (param->type() == ParameterType::kParameterString) {
                        std::string val = param->toValue<std::string>();
                        if (val.substr(0, 6) == "torch.") {
                            pyfp << val;
                        } else if (op->type() == "Tensor.index_put" && it.first == "values") {
                            if (val == "inf" || val == "-inf") {
                                pyfp << "torch.tensor(float(\'" << val << "\'))";
                            } else {
                                pyfp << "torch.tensor(\'" << val << "\')";
                            }
                        } else {
                            if (val == "inf" || val == "-inf") {
                                pyfp << "float(\'" << val << "\')";
                            } else {
                                pyfp << "\'" << val << "\'";
                            }
                        }
                    }

                    if (param->type() == ParameterType::kParameterArrayInt) {
                        std::vector<int> val = param->toValue<std::vector<int>>();
                        pyfp << "(";
                        for (size_t j = 0; j < val.size(); ++j) {
                            if ((op->type() == "F.adaptive_avg_pool2d" ||
                                 op->type() == "F.adaptive_avg_pool3d" ||
                                 op->type() == "F.adaptive_max_pool2d" ||
                                 op->type() == "F.adaptive_max_pool3d") &&
                                it.first == "output_size" && val[j] == 0) {
                                pyfp << "None";
                            } else {
                                pyfp << val[j];
                            }

                            if (j + 1 != val.size() || val.size() == 1) {
                                pyfp << ",";
                            }
                        }
                        pyfp << ")";
                    }

                    if (param->type() == ParameterType::kParameterArrayFloat) {
                        std::vector<float> val = param->toValue<std::vector<float>>();
                        pyfp << "(";
                        for (size_t j = 0; j < val.size(); ++j) {
                            pyfp << val[j];
                            if (j + 1 != val.size() || val.size() == 1) {
                                pyfp << ",";
                            }
                        }
                        pyfp << ")";
                    }

                    if (param->type() == ParameterType::kParameterArrayString) {
                        std::vector<std::string> val = param->toValue<std::vector<std::string>>();
                        pyfp << "(";
                        for (size_t j = 0; j < val.size(); ++j) {
                            if (val[j].substr(0, 6) == "torch.") {
                                pyfp << val[j];
                            } else {
                                pyfp << "\'" << val[j] << "\'";
                            }

                            if (j + 1 != val.size() || val.size() == 1) {
                                pyfp << ",";
                            }
                        }
                        pyfp << ")";
                    }

                    if (param->type() == ParameterType::kParameterComplex) {
                        pyfp << "(" << param->toString() << ")";
                    }

                    if (param->type() == ParameterType::kParameterArrayComplex) {
                        auto val = param->toValue<std::vector<std::complex<float>>>();
                        pyfp << "(";
                        for (size_t j = 0; j < val.size(); ++j) {
                            pyfp << "(" << val[j].real() << "+" << val[j].imag() << "i)";

                            if (j + 1 != val.size() || val.size() == 1) {
                                pyfp << ",";
                            }
                        }
                        pyfp << ")";
                    }
                }
                pyfp << ")\n";
            }
        }
    }

    // return
    {
        pyfp << "        return ";
        int outputCount = 0;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Output") {
                outputCount++;
            }
        }

        int outputIndex = 0;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Output") {
                pyfp << "v_" << SanitizeIdentifier(op->GetInputOperands()[0]->name());
                if (outputIndex + 1 != outputCount) {
                    pyfp << ", ";
                }
                outputIndex++;
            }
        }
        pyfp << std::endl;
    }

    pyfp << std::endl;

    // export torchscript
    {
        pyfp << "def export_torchscript():\n";
        pyfp << "    net = Model()\n";
        pyfp << "    net.eval()\n";
        pyfp << std::endl;
        pyfp << "    torch.manual_seed(0)\n";

        std::vector<std::string> input_names;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Input") {
                const auto& r = op->GetOutputOperands()[0];
                std::string inputName = "v_" + SanitizeIdentifier(r->name());
                if (IsInteger(r->type())) {
                    pyfp << "    " << inputName << " = torch.randint(10, (";
                    for (size_t i = 0; i < r->GetShape().size(); ++i) {
                        pyfp << r->GetShape()[i];
                        if (i + 1 != r->GetShape().size() || r->GetShape().size() == 1) {
                            pyfp << ", ";
                        }
                    }

                    pyfp << "), dtype=" << DataType2TorchString(r->type()) << ")\n";
                } else {
                    pyfp << "    " << inputName << " = torch.rand(";
                    for (const auto& elem: r->GetShape()) {
                        pyfp << elem << ", ";
                    }
                    pyfp << "dtype=" << DataType2TorchString(r->type()) << ")\n";
                }
                input_names.push_back(inputName);
            }
        }

        pyfp << std::endl;

        if (input_names.size() == 1) {
            pyfp << "    mod = torch.jit.trace(net, " << input_names[0] << ")\n";
        } else {
            pyfp << "    mod = torch.jit.trace(net, (";
            size_t size = input_names.size();
            for (const auto& elem: input_names) {
                pyfp << elem << (--size ? ", " : "");
            }
            pyfp << "))\n";
        }
        pyfp << "    mod.save(\"" << pyPath << ".pt\")\n";
    }

    pyfp << std::endl;

    // export onnx
    {
        pyfp << "def export_onnx():\n";
        pyfp << "    net = Model()\n";
        pyfp << "    net.eval()\n";
        pyfp << std::endl;
        pyfp << "    torch.manual_seed(0)\n";

        std::vector<std::string> input_names;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Input") {
                const auto& r = op->GetOutputOperands()[0];
                std::string inputName = "v_" + SanitizeIdentifier(r->name());
                if (IsInteger(r->type())) {
                    pyfp << "    " << inputName << " = torch.randint(10, (";
                    for (size_t i = 0; i < r->GetShape().size(); ++i) {
                        pyfp << r->GetShape()[i];
                        if (i + 1 != r->GetShape().size() || r->GetShape().size() == 1) {
                            pyfp << ", ";
                        }
                    }

                    pyfp << "), dtype=" << DataType2TorchString(r->type()) << ")\n";
                } else {
                    pyfp << "    " << inputName << " = torch.rand(";
                    for (const auto& elem: r->GetShape()) {
                        pyfp << elem << ", ";
                    }
                    pyfp << "dtype=" << DataType2TorchString(r->type()) << ")\n";
                }
                input_names.push_back(inputName);
            }
        }

        pyfp << std::endl;

        if (input_names.size() == 1) {
            pyfp << "    torch.onnx._export(net, " << input_names[0];
        } else {
            pyfp << "    torch.onnx._export(net, (";
            size_t size = input_names.size();
            for (const auto& elem: input_names) {
                pyfp << elem << (--size ? ", " : "");
            }
            pyfp << ")";
        }

        pyfp << ", \"" << pyPath << ".onnx\", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13";
        pyfp << ", input_names=[";

        int inputCount = 0;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Input") {
                inputCount++;
            }
        }

        int inputIndex = 0;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Input") {
                pyfp << "\'in" << inputIndex << "\'";
                if (inputIndex + 1 != inputCount) {
                    pyfp << ", ";
                }
                inputIndex++;
            }
        }

        pyfp << "]";

        pyfp << ", output_names=[";
        int outputCount = 0;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Output") {
                outputCount++;
            }
        }

        int outputIndex = 0;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Output") {
                pyfp << "\'out" << outputIndex << "\'";
                if (outputIndex + 1 != outputCount) {
                    pyfp << ", ";
                }
                outputIndex++;
            }
        }
        pyfp << "]";
        pyfp << ")\n";
    }

    pyfp << std::endl;

    // test inference
    {
        pyfp << "def test_inference():\n";
        pyfp << "    net = Model()\n";
        pyfp << "    net.eval()\n";
        pyfp << std::endl;
        pyfp << "    torch.manual_seed(0)\n";

        std::vector<std::string> input_names;
        for (const auto& op: GetOperators()) {
            if (op->type() == "pnnx.Input") {
                const auto& r = op->GetOutputOperands()[0];
                std::string inputName = "v_" + SanitizeIdentifier(r->name());
                if (IsInteger(r->type())) {
                    pyfp << "    " << inputName << " = torch.randint(10, (";
                    for (size_t i = 0; i < r->GetShape().size(); ++i) {
                        pyfp << r->GetShape()[i];
                        if (i + 1 != r->GetShape().size() || r->GetShape().size() == 1) {
                            pyfp << ", ";
                        }
                    }
                    pyfp << "), dtype=" << DataType2TorchString(r->type()) << ")\n";
                } else {
                    pyfp << "    " << inputName << " = torch.rand(";
                    for (const auto& elem: r->GetShape()) {
                        pyfp << elem << ", ";
                    }
                    pyfp << "dtype=" << DataType2TorchString(r->type()) << ")\n";
                }
                input_names.push_back(inputName);
            }
        }

        pyfp << std::endl;
        if (input_names.size() == 1) {
            pyfp << "    return net(" << input_names[0] << ")\n";
        } else {
            pyfp << "    return net(";
            size_t size = input_names.size();
            for (const auto& elem: input_names) {
                pyfp << elem << (--size ? ", " : "");
            }
            pyfp << ")\n";
        }
    }

    pyfp << std::endl;

    // main
    {
        pyfp << "if __name__ == \"__main__\":\n";
        pyfp << "    print(test_inference())\n";
    }

    pyfp.close();
    return 0;
}

std::shared_ptr<Operand> Graph::CreateOperator(const std::string& type, const std::string& name,
                                               const std::map<std::string, std::shared_ptr<Parameter>>& params,
                                               const std::map<std::string, std::shared_ptr<Attribute>>& attrs,
                                               const std::vector<std::shared_ptr<Operand>>& inputOperands,
                                               const std::vector<std::string>& inputOperandNames,
                                               const std::string& outputName,
                                               DataType outputType,
                                               const std::vector<int>& outputShape) {
    auto op = std::make_shared<Operator>(name, type, params, attrs, inputOperands, inputOperandNames);
    ops_.push_back(op);

    for (const auto& it: inputOperands) {
        it->AddConsumer(op);
    }

    if (!outputName.empty() && !outputShape.empty()) {
        auto outOperand = std::make_shared<Operand>(outputName, outputType, outputShape);
        op->AddOutputOperand(outOperand);
        outOperand->SetProducer(op);

        operands_.push_back(outOperand);
        return outOperand;
    }

    return {};
}

std::shared_ptr<Operator> Graph::CreateOperator(const std::string& type, const std::string& name) {
    auto op = std::make_shared<Operator>(name, type);
    ops_.push_back(op);
    return op;
}

std::shared_ptr<Operator> Graph::CreateOperatorBefore(const std::string& type, const std::string& name, const std::shared_ptr<Operator>& cur) {
    auto op = std::make_shared<Operator>(name, type);
    ops_.insert(std::find(ops_.begin(), ops_.end(), cur), op);
    return op;
}

std::shared_ptr<Operator> Graph::CreateOperatorAfter(const std::string& type, const std::string& name, const std::shared_ptr<Operator>& cur) {
    auto op = std::make_shared<Operator>(name, type);
    ops_.insert(std::find(ops_.begin(), ops_.end(), cur) + 1, op);
    return op;
}

std::shared_ptr<Operand> Graph::CreateOperand(const std::string& name) {
    auto r = std::make_shared<Operand>(name);
    operands_.push_back(r);
    return r;
}

std::shared_ptr<Operand> Graph::GetOperand(const std::string& name) {
    for (const auto& r: operands_) {
        if (r->name() == name) {
            return r;
        }
    }
    return {};
}

}// namespace pnnx