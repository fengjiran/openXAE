//
// Created by fengj on 2024/5/31.
//

#include "utils.h"

#include <algorithm>
#include <cstring>
#include <iostream>

namespace pnnx {

unsigned short float32_to_float16(float value) {
    // FP32
    // sign: 1
    // exponent: 8
    // significand(mantissa): 23
    union {
        unsigned int u;
        float f;
    } tmp;
    tmp.f = value;

    unsigned short sign = (tmp.u & 0x80000000) >> 31;
    unsigned short exponent = (tmp.u & 0x7F800000) >> 23;
    unsigned int significand = tmp.u & 0x7FFFFF;

    // FP16
    // sign: 1
    // exponent: 5
    // significand(mantissa): 10
    unsigned short fp16;
    if (exponent == 0) {
        // zero or denormal, always underflow
        fp16 = (sign << 15) | (0x00 << 10) | 0x00;
    } else if (exponent == 0xFF) {
        // infinity or NaN
        fp16 = (sign << 15) | (0x1F << 10) | (significand ? 0x200 : 0x00);
    } else {
        // normalized
        short newexp = exponent + (-127 + 15);
        if (newexp >= 31) {
            // overflow, return infinity
            fp16 = (sign << 15) | (0x1F << 10) | 0x00;
        } else if (newexp <= 0) {
            // Some normal fp32 cannot be expressed as normal fp16
            fp16 = (sign << 15) | (0x00 << 10) | 0x00;
        } else {
            // normal fp16
            fp16 = (sign << 15) | (newexp << 10) | (significand >> 13);
        }
    }

    return fp16;
}

float float16_to_float32(unsigned short value) {
    // FP16
    // sign: 1
    // exponent: 5
    // significand(mantissa): 10
    unsigned short sign = (value & 0x8000) >> 15;
    unsigned short exponent = (value & 0x7c00) >> 10;
    unsigned short significand = value & 0x03FF;

    // FP32
    // sign: 1
    // exponent: 8
    // significand(mantissa): 23
    union {
        unsigned int u;
        float f;
    } tmp;

    if (exponent == 0) {
        if (significand == 0) {
            // zero
            tmp.u = (sign << 31);
        } else {
            // denormal
            exponent = 0;
            // find non-zero bit
            while ((significand & 0x200) == 0) {
                significand <<= 1;
                exponent++;
            }
            significand <<= 1;
            significand &= 0x3FF;
            tmp.u = (sign << 31) | ((-exponent + (-15 + 127)) << 23) | (significand << 13);
        }
    } else if (exponent == 0x1F) {
        // infinity or NaN
        tmp.u = (sign << 31) | (0xFF << 23) | (significand << 13);
    } else {
        // normalized
        tmp.u = (sign << 31) | ((exponent + (-15 + 127)) << 23) | (significand << 13);
    }

    return tmp.f;
}

bool IsInteger(DataType type) {
    bool flag;
    switch (type) {
        case DataType::kDataTypeInt32:
        case DataType::kDataTypeInt64:
        case DataType::kDataTypeInt16:
        case DataType::kDataTypeInt8:
        case DataType::kDataTypeUInt8:
        case DataType::kDataTypeBool:
            flag = true;
            break;

        case DataType::kDataTypeUnknown:
        case DataType::kDataTypeFloat32:
        case DataType::kDataTypeFloat64:
        case DataType::kDataTypeFloat16:
        case DataType::kDataTypeComplex64:
        case DataType::kDataTypeComplex128:
        case DataType::kDataTypeComplex32:
        case DataType::kDataTypeBFloat16:
            flag = false;
            break;

        default:
            flag = false;
    }

    return flag;
}

std::string DataType2String(DataType type) {
    std::string str;
    switch (type) {
        case DataType::kDataTypeFloat32:
            str = "f32";
            break;
        case DataType::kDataTypeFloat64:
            str = "f64";
            break;
        case DataType::kDataTypeFloat16:
            str = "f16";
            break;
        case DataType::kDataTypeInt32:
            str = "i32";
            break;
        case DataType::kDataTypeInt64:
            str = "i64";
            break;
        case DataType::kDataTypeInt16:
            str = "i16";
            break;
        case DataType::kDataTypeInt8:
            str = "i8";
            break;
        case DataType::kDataTypeUInt8:
            str = "u8";
            break;
        case DataType::kDataTypeBool:
            str = "bool";
            break;
        case DataType::kDataTypeComplex64:
            str = "c64";
            break;
        case DataType::kDataTypeComplex128:
            str = "c128";
            break;
        case DataType::kDataTypeComplex32:
            str = "c32";
            break;
        case DataType::kDataTypeBFloat16:
            str = "bf16";
            break;
        case DataType::kDataTypeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

std::string DataType2NumpyString(DataType type) {
    std::string str;
    switch (type) {
        case DataType::kDataTypeFloat32:
            str = "float32";
            break;
        case DataType::kDataTypeFloat64:
            str = "float64";
            break;
        case DataType::kDataTypeFloat16:
            str = "float16";
            break;
        case DataType::kDataTypeInt32:
            str = "int32";
            break;
        case DataType::kDataTypeInt64:
            str = "int64";
            break;
        case DataType::kDataTypeInt16:
            str = "int16";
            break;
        case DataType::kDataTypeInt8:
            str = "int8";
            break;
        case DataType::kDataTypeUInt8:
            str = "uint8";
            break;
        case DataType::kDataTypeBool:
            str = "bool8";
            break;
        case DataType::kDataTypeComplex64:
            str = "csingle";
            break;
        case DataType::kDataTypeComplex128:
            str = "cdouble";
            break;
        case DataType::kDataTypeComplex32:
            str = "chalf";
            break;
        case DataType::kDataTypeBFloat16:
            str = "bfloat16";
            break;
        case DataType::kDataTypeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

std::string DataType2TorchString(DataType type) {
    std::string str;
    switch (type) {
        case DataType::kDataTypeFloat32:
            str = "torch.float";
            break;
        case DataType::kDataTypeFloat64:
            str = "torch.double";
            break;
        case DataType::kDataTypeFloat16:
            str = "torch.half";
            break;
        case DataType::kDataTypeInt32:
            str = "torch.int";
            break;
        case DataType::kDataTypeInt64:
            str = "torch.long";
            break;
        case DataType::kDataTypeInt16:
            str = "torch.short";
            break;
        case DataType::kDataTypeInt8:
            str = "torch.int8";
            break;
        case DataType::kDataTypeUInt8:
            str = "torch.uint8";
            break;
        case DataType::kDataTypeBool:
            str = "torch.bool";
            break;
        case DataType::kDataTypeComplex64:
            str = "torch.complex64";
            break;
        case DataType::kDataTypeComplex128:
            str = "torch.complex128";
            break;
        case DataType::kDataTypeComplex32:
            str = "torch.complex32";
            break;
        case DataType::kDataTypeBFloat16:
            str = "torch.bfloat16";
            break;
        case DataType::kDataTypeUnknown:
            str = "unknown";
            break;
    }
    return str;
}

size_t SizeOf(DataType type) {
    size_t size;
    switch (type) {
        case DataType::kDataTypeUnknown:
            size = 0;
            break;

        case DataType::kDataTypeInt8:
        case DataType::kDataTypeUInt8:
        case DataType::kDataTypeBool:
            size = 1;
            break;

        case DataType::kDataTypeInt16:
        case DataType::kDataTypeFloat16:
        case DataType::kDataTypeBFloat16:
            size = 2;
            break;

        case DataType::kDataTypeInt32:
        case DataType::kDataTypeFloat32:
        case DataType::kDataTypeComplex32:
            size = 4;
            break;

        case DataType::kDataTypeInt64:
        case DataType::kDataTypeFloat64:
        case DataType::kDataTypeComplex64:
            size = 8;
            break;

        case DataType::kDataTypeComplex128:
            size = 16;
            break;
    }
    return size;
}

DataType String2Type(const std::string& s) {
    if (s == "f32") return DataType::kDataTypeFloat32;
    if (s == "f64") return DataType::kDataTypeFloat64;
    if (s == "f16") return DataType::kDataTypeFloat16;
    if (s == "i32") return DataType::kDataTypeInt32;
    if (s == "i64") return DataType::kDataTypeInt64;
    if (s == "i16") return DataType::kDataTypeInt16;
    if (s == "i8") return DataType::kDataTypeInt8;
    if (s == "u8") return DataType::kDataTypeUInt8;
    if (s == "bool") return DataType::kDataTypeBool;
    if (s == "c64") return DataType::kDataTypeComplex64;
    if (s == "c128") return DataType::kDataTypeComplex128;
    if (s == "c32") return DataType::kDataTypeComplex32;
    if (s == "bf16") return DataType::kDataTypeBFloat16;
    return DataType::kDataTypeUnknown;
}

std::string GetBasename(const std::string& path) {
    std::string dirpath;
    std::string filename;

    size_t dirpos = path.find_last_of("/\\");
    if (dirpos != std::string::npos) {
        dirpath = path.substr(0, dirpos + 1);
        filename = path.substr(dirpos + 1);
    } else {
        filename = path;
    }

    std::string base = filename.substr(0, filename.find_last_of('.'));
    // sanitize -
    std::replace(base.begin(), base.end(), '-', '_');
    return dirpath + base;
}

void ParseStringList(char* s, std::vector<std::string>& list) {
    list.clear();
    char* pch = strtok(s, ",");
    while (pch) {
        list.emplace_back(pch);
        pch = strtok(nullptr, ",");
    }
}

void PrintStringList(const std::vector<std::string>& list) {
    auto size = list.size();
    for (const auto& item: list) {
        std::cerr << item + std::string(--size ? "," : "");
    }

    std::cerr << std::endl;
}

void ParseShapeList(char* s, std::vector<std::vector<int64_t>>& shapes, std::vector<std::string>& types) {
    shapes.clear();
    types.clear();

    char* pch = strtok(s, "[]");
    while (pch != nullptr) {
        // assign user data type
        if (!types.empty() && (pch[0] == 'b' ||
                               pch[0] == 'f' ||
                               pch[0] == 'i' ||
                               pch[0] == 'u' ||
                               pch[0] == 'c')) {
            char type[32];
            int nscan = sscanf(pch, "%31[^,]", type);
            if (nscan == 1) {
                types[types.size() - 1] = std::string(type);
            }
        }

        // parse a,b,c
        int v;
        int nconsumed = 0;
        int nscan = sscanf(pch, "%d%n", &v, &nconsumed);
        if (nscan == 1) {
            // ok we get shape
            pch += nconsumed;

            std::vector<int64_t> ss;
            ss.push_back(v);

            nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            while (nscan == 1) {
                pch += nconsumed;

                ss.push_back(v);

                nscan = sscanf(pch, ",%d%n", &v, &nconsumed);
            }

            // shape end
            shapes.push_back(ss);
            types.emplace_back("f32");
        }

        pch = strtok(nullptr, "[]");
    }
}

void PrintShapeList(const std::vector<std::vector<int64_t>>& shapes, const std::vector<std::string>& types) {
    for (size_t i = 0; i < shapes.size(); ++i) {
        const std::vector<int64_t>& s = shapes[i];
        const std::string& t = types[i];
        std::cerr << "[";
        auto size = s.size();
        for (const auto& item: s) {
            std::cerr << std::to_string(item) + (--size ? "," : "");
        }
        std::cerr << "]" << t;
        if (i != shapes.size() - 1) {
            std::cerr << ",";
        }
    }
}

bool ModelFileMaybeTorchscript(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        std::cerr << "open failed: " << path << std::endl;
        return false;
    }

    uint32_t signature = 0;
    fread((char*) &signature, sizeof(signature), 1, fp);

    fclose(fp);

    // torchscript is a zip
    return signature == 0x04034b50;
}

}// namespace pnnx
