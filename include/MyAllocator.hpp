//
// Created by 赵丹 on 24-5-10.
//

#ifndef OPENXAE_MYALLOCATOR_HPP
#define OPENXAE_MYALLOCATOR_HPP

#include "config.hpp"
#include <cstddef>
#include <memory>

namespace XAcceleratorEngine {

template<typename T>
class MyAllocator {
public:
    using value_type = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using void_pointer = void*;

public:
    MyAllocator() noexcept = default;

    template<typename U>
    explicit MyAllocator(const MyAllocator<U>&) noexcept {}

    pointer allocate(size_type n) {
        if (n > std::allocator_traits<MyAllocator>::max_size(*this)) {
            throw std::bad_array_new_length();
        }
        return static_cast<pointer>(::operator new(n * sizeof(value_type)));
    }

    void deallocate(pointer p, size_type n) noexcept {
        ::operator delete(p);
    }

    static MyAllocator select_on_container_copy_construction() {
        return {};
    }
};

template<typename T, typename U>
inline bool operator==(const MyAllocator<T>&, const MyAllocator<U>&) noexcept {
    return true;
}

template<typename T, typename U>
inline bool operator!=(const MyAllocator<T>&, const MyAllocator<U>&) noexcept {
    return false;
}

}// namespace XAcceleratorEngine

#endif//OPENXAE_MYALLOCATOR_HPP
