//
// Created by fengj on 2024/5/30.
//

#ifndef OPENXAE_STOREZIP_H
#define OPENXAE_STOREZIP_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pnnx {

class StoreZipReader {
public:
    StoreZipReader() : fp(nullptr) {}
    ~StoreZipReader() { close(); }

    int open(const std::string& path);

    std::vector<std::string> get_names() const;

    uint64_t get_file_size(const std::string& name) const;

    int read_file(const std::string& name, char* data);

    int close();

private:
    FILE* fp;
    struct StoreZipMeta {
        uint64_t offset;
        uint64_t size;
    };
    std::map<std::string, StoreZipMeta> filemetas;
};

class StoreZipWriter {
public:
    StoreZipWriter();
    ~StoreZipWriter();

    int open(const std::string& path);

    int write_file(const std::string& name, const char* data, uint64_t size);

    int close();

private:
    FILE* fp;
    struct StoreZipMeta {
        std::string name;
        uint64_t lfh_offset;
        uint32_t crc32;
        uint64_t size;
    };

    std::vector<StoreZipMeta> filemetas;
};

}// namespace pnnx

#endif//OPENXAE_STOREZIP_H
