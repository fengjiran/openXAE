//
// Created by 赵丹 on 24-5-30.
//
#include "ir.h"
#include "storezip.h"
#include "utils.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stack>
#include <string>

namespace pnnx {

static bool type_is_integer(int type) {
    if (type == 1) return false;
    if (type == 2) return false;
    if (type == 3) return false;
    if (type == 4) return true;
    if (type == 5) return true;
    if (type == 6) return true;
    if (type == 7) return true;
    if (type == 8) return true;
    if (type == 9) return true;
    if (type == 10) return false;
    if (type == 11) return false;
    if (type == 12) return false;
    if (type == 13) return false;
    return false;
}

}// namespace pnnx