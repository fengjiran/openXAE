//
// Created by fengj on 2024/5/30.
//
#include "utils.h"

#if BUILD_TORCH2PNNX
#include "torch/torch2pnnx.h"
#endif

static void ShowUsage() {
    std::cerr << "Usage: pnnx [model.pt] [(key=value)...]\n";
    std::cerr << "  pnnxparam=model.pnnx.param\n";
    std::cerr << "  pnnxbin=model.pnnx.bin\n";
    std::cerr << "  pnnxpy=model_pnnx.py\n";
    std::cerr << "  pnnxonnx=model.pnnx.onnx\n";
    std::cerr << "  ncnnparam=model.ncnn.param\n";
    std::cerr << "  ncnnbin=model.ncnn.bin\n";
    std::cerr << "  ncnnpy=model_ncnn.py\n";
    std::cerr << "  fp16=1\n";
    std::cerr << "  optlevel=2\n";
    std::cerr << "  device=cpu/gpu\n";
    std::cerr << "  inputshape=[1,3,224,224],...\n";
    std::cerr << "  inputshape2=[1,3,320,320],...\n";
    std::cerr << "  customop=/home/nihui/.cache/torch_extensions/fused/fused.so,...\n";
    std::cerr << "  moduleop=models.common.Focus,models.yolo.Detect,...\n";
    std::cerr << "Sample usage: pnnx mobilenet_v2.pt inputshape=[1,3,224,224]\n";
    std::cerr << "              pnnx yolov5s.pt inputshape=[1,3,640,640]f32 inputshape2=[1,3,320,320]f32 device=gpu moduleop=models.common.Focus,models.yolo.Detect\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        ShowUsage();
        return -1;
    }

    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') {
            ShowUsage();
            return -1;
        }
    }

    std::string ptPath(argv[1]);
    std::string ptBase = pnnx::GetBasename(ptPath);
    std::string pnnxParamPath = ptBase + ".pnnx.param";
    std::string pnnxBinPath = ptBase + ".pnnx.bin";
    std::string pnnxPyPath = ptBase + "_pnnx.py";
    std::string pnnxOnnxPath = ptBase + ".pnnx.onnx";
    std::string ncnnParamPath = ptBase + ".ncnn.param";
    std::string ncnnBinPath = ptBase + ".ncnn.bin";
    std::string ncnnPyPath = ptBase + "_ncnn.py";
    int fp16 = 1;
    int optlevel = 2;
    std::string device = "cpu";
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<std::string> input_types;
    std::vector<std::vector<int64_t>> input_shapes2;
    std::vector<std::string> input_types2;
    std::vector<std::string> customop_modules;
    std::vector<std::string> module_operators;

    for (int i = 2; i < argc; ++i) {
        // key = value
        char* kv = argv[i];
        char* eqs = strchr(kv, '=');
        if (!eqs) {
            std::cerr << "unrecognized arg: " << kv << std::endl;
            continue;
        }

        // split kv
        eqs[0] = '\0';
        const char* key = kv;
        char* value = eqs + 1;

        if (strcmp(key, "pnnxparam") == 0)
            pnnxParamPath = std::string(value);
        if (strcmp(key, "pnnxbin") == 0)
            pnnxBinPath = std::string(value);
        if (strcmp(key, "pnnxpy") == 0)
            pnnxPyPath = std::string(value);
        if (strcmp(key, "pnnxonnx") == 0)
            pnnxOnnxPath = std::string(value);
        if (strcmp(key, "ncnnparam") == 0)
            ncnnParamPath = std::string(value);
        if (strcmp(key, "ncnnbin") == 0)
            ncnnBinPath = std::string(value);
        if (strcmp(key, "ncnnpy") == 0)
            ncnnPyPath = std::string(value);
        if (strcmp(key, "fp16") == 0)
            fp16 = std::atoi(value);
        if (strcmp(key, "optlevel") == 0)
            optlevel = std::atoi(value);
        if (strcmp(key, "device") == 0)
            device = value;
        if (strcmp(key, "inputshape") == 0)
            pnnx::ParseShapeList(value, input_shapes, input_types);
        if (strcmp(key, "inputshape2") == 0)
            pnnx::ParseShapeList(value, input_shapes2, input_types2);
        if (strcmp(key, "customop") == 0)
            pnnx::ParseStringList(value, customop_modules);
        if (strcmp(key, "moduleop") == 0)
            pnnx::ParseStringList(value, module_operators);
    }

    // print options
    {
        std::cerr << "pnnxparam = " << pnnxParamPath << std::endl;
        std::cerr << "pnnxbin = " << pnnxBinPath << std::endl;
        std::cerr << "pnnxpy = "
                  << pnnxPyPath << std::endl;
        std::cerr << "pnnxonnx = "
                  << pnnxOnnxPath << std::endl;
        std::cerr << "ncnnparam = "
                  << ncnnParamPath << std::endl;
        std::cerr << "ncnnbin = "
                  << ncnnBinPath << std::endl;
        std::cerr << "ncnnpy = "
                  << ncnnPyPath << std::endl;
        std::cerr << "fp16 = "
                  << fp16 << std::endl;
        std::cerr << "optlevel = "
                  << optlevel << std::endl;
        std::cerr << "device = "
                  << device << std::endl;
        std::cerr << "inputshape = ";
        pnnx::PrintShapeList(input_shapes, input_types);
        std::cerr << std::endl;
        std::cerr << "inputshape2 = ";
        pnnx::PrintShapeList(input_shapes2, input_types2);
        std::cerr << std::endl;
        std::cerr << "customop = ";
        pnnx::PrintStringList(customop_modules);
        std::cerr << std::endl;
        std::cerr << "moduleop = ";
        pnnx::PrintStringList(module_operators);
        std::cerr << std::endl;
    }
}