//
// Created by richard on 5/4/24.
//
#ifndef OPENXAE_MYVECTOR_HPP
#define OPENXAE_MYVECTOR_HPP

#include "config.hpp"
#include "glog/logging.h"

namespace XAcceleratorEngine {

template<typename T, typename Allocator = std::allocator<T>>
class vec {
public:
    using allocTraits = std::allocator_traits<Allocator>;
    using size_type = typename allocTraits::size_type;

public:
    /**
     * @brief Default constructor
     */
    vec() : start(nullptr), cap(nullptr), firstFree(nullptr) {}

    /**
     * @brief Constructor with size and default initialization
     *
     * @param n Size
     */
    explicit vec(size_type n);

    /**
     * @brief Constructor with size and initial value
     *
     * @param n Size
     * @param t Initial value
     */
    vec(size_type n, const T& t);

    vec(const T* first, const T* last);

    //    vec(const char* const* first, const char* const* last);

    /**
     * @brief Constructor with initializer list
     *
     * @param il Initializer list
     */
    vec(std::initializer_list<T> il);

    vec& operator=(std::initializer_list<T> il);

    /**
     * @brief Copy constructor
     *
     * @param rhs Right Hand Side
     */
    vec(const vec& rhs);

    /**
     * @brief Copy assignment operator
     *
     * @param rhs Right Hand Side
     * @return The self reference
     */
    vec& operator=(const vec& rhs);

    /**
     * @brief Move constructor
     *
     * @param rhs Right Hand Side
     */
    vec(vec&& rhs) noexcept;

    /**
     * @brief Move assignment
     *
     * @param rhs Right Hand Side
     * @return The self reference
     */
    vec& operator=(vec&& rhs) noexcept;

    /**
     * @brief Get the number of elements in the container
     *
     * @return The number of elements in the container.
     */
    size_type size() const { return firstFree - start; }

    /**
     * @brief Get the number of elements that the container has currently allocated space for
     *
     * @return Capacity of the currently allocated storage.
     */
    size_type capacity() const { return cap - start; }

    /**
     * @brief Get the first element pointer of the vector
     *
     * @return The pointer of the first element
     */
    T* begin() const { return start; }

    /**
     * @brief Get the ptr to the element following the last element of the vector
     *
     * @return The ptr to the element following the last element of the vector
     */
    T* end() const { return firstFree; }

    CPP_NODISCARD bool empty() const { return firstFree == start; }

    void reserve(size_type n);
    void resize(size_type n);
    void resize(size_type n, const T& t);
    T* data() { return start; }

    void push_back(const T& t);
    void push_back(T&& t);

    template<typename... Args>
    void emplace_back(Args&&... args);

    T& operator[](size_type pos) noexcept;
    const T& operator[](size_type pos) const noexcept;

    T& at(size_type pos);
    const T& at(size_type pos) const;

    T& front() noexcept;
    const T& front() const noexcept;

    /**
     * @brief Get the reference to the last element in the container.
     *
     * @return Reference to the last element.
     */
    T& back() noexcept;

    /**
     * @brief Get the reference to the last element in the container.
     *
     * @return Reference to the last element.
     */
    const T& back() const noexcept;

    ~vec();

private:
    std::pair<T*, T*> Allocate(const T* b, const T* e);
    void free();
    void reallocate();
    void reallocate(size_type newCap);
    void CheckAndAlloc() {
        if (firstFree == cap) {
            reallocate();
        }
    }

    Allocator alloc;
    T* start;
    T* cap;
    T* firstFree;
};

template<typename T, typename Allocator>
const T& vec<T, Allocator>::back() const noexcept {
    CHECK(!empty()) << "back() called on an empty vector";
    return *(firstFree - 1);
}

template<typename T, typename Allocator>
T& vec<T, Allocator>::back() noexcept {
    CHECK(!empty()) << "back() called on an empty vector";
    return *(firstFree - 1);
}

template<typename T, typename Allocator>
const T& vec<T, Allocator>::front() const noexcept {
    CHECK(!empty()) << "front() called on an empty vector";
    return *start;
}

template<typename T, typename Allocator>
T& vec<T, Allocator>::front() noexcept {
    CHECK(!empty()) << "front() called on an empty vector";
    return *start;
}

template<typename T, typename Allocator>
const T& vec<T, Allocator>::operator[](size_type pos) const noexcept {
    CHECK(pos < size()) << "vector[] index out of bounds";
    return start[pos];
}

template<typename T, typename Allocator>
T& vec<T, Allocator>::operator[](size_type pos) noexcept {
    CHECK(pos < size()) << "vector[] index out of bounds";
    return start[pos];
}

template<typename T, typename Allocator>
T& vec<T, Allocator>::at(size_type pos) {
    if (pos >= size()) {
        throw std::out_of_range("index out of bounds");
    }
    return (*this)[pos];
}

template<typename T, typename Allocator>
const T& vec<T, Allocator>::at(size_type pos) const {
    if (pos >= size()) {
        throw std::out_of_range("index out of bounds");
    }
    return (*this)[pos];
}

template<typename T, typename Allocator>
template<typename... Args>
void vec<T, Allocator>::emplace_back(Args&&... args) {
    CheckAndAlloc();
    allocTraits::construct(alloc, firstFree++, std::forward<Args>(args)...);
}

template<typename T, typename Allocator>
void vec<T, Allocator>::push_back(T&& t) {
    CheckAndAlloc();
    allocTraits::construct(alloc, firstFree++, std::move(t));
}

template<typename T, typename Allocator>
void vec<T, Allocator>::push_back(const T& t) {
    CheckAndAlloc();
    allocTraits::construct(alloc, firstFree++, t);
}

template<typename T, typename Allocator>
void vec<T, Allocator>::resize(size_type n) {
    if (n > size()) {
        while (size() < n) {
            push_back(T());
        }
    } else {
        while (size() > n) {
            allocTraits::destroy(alloc, --firstFree);
        }
    }
}

template<typename T, typename Allocator>
void vec<T, Allocator>::resize(size_type n, const T& t) {
    if (n > size()) {
        while (size() < n) {
            push_back(t);
        }
    }
}

template<typename T, typename Allocator>
void vec<T, Allocator>::reserve(size_type n) {
    if (n > capacity()) {
        reallocate(n);
    }
}

template<typename T, typename Allocator>
vec<T, Allocator>::~vec() {
    free();
}

template<typename T, typename Allocator>
void vec<T, Allocator>::reallocate(size_type newCap) {
    auto data = allocTraits::allocate(alloc, newCap);
    auto src = start;
    auto dst = data;
    for (size_type i = 0; i < size(); ++i) {
        allocTraits::construct(alloc, dst, std::move(*src));
        ++src;
        ++dst;
    }
    free();
    start = data;
    firstFree = dst;
    cap = start + newCap;
}

template<typename T, typename Allocator>
void vec<T, Allocator>::reallocate() {
    size_type newCap = size() != 0 ? 2 * size() : 1;
    auto data = allocTraits::allocate(alloc, newCap);
    auto src = start;
    auto dst = data;
    for (size_type i = 0; i < size(); ++i) {
        allocTraits::construct(alloc, dst, std::move(*src));
        ++src;
        ++dst;
    }
    free();
    start = data;
    firstFree = dst;
    cap = start + newCap;
}

template<typename T, typename Allocator>
vec<T, Allocator>& vec<T, Allocator>::operator=(vec&& rhs) noexcept {
    if (this != &rhs) {
        free();
        start = rhs.start;
        firstFree = rhs.firstFree;
        cap = rhs.cap;

        rhs.start = nullptr;
        rhs.firstFree = nullptr;
        rhs.cap = nullptr;
    }
    return *this;
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(vec&& rhs) noexcept : start(rhs.start), firstFree(rhs.firstFree), cap(rhs.cap) {
    LOG(INFO) << " use move constructor";
    rhs.start = nullptr;
    rhs.firstFree = nullptr;
    rhs.cap = nullptr;
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(size_type n) {
    auto data = allocTraits::allocate(alloc, n);
    start = data;
    firstFree = data;
    cap = start + n;
    for (size_type i = 0; i < n; ++i) {
        allocTraits::construct(alloc, firstFree++, T());
    }
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(size_type n, const T& t) {
    auto data = allocTraits::allocate(alloc, n);
    start = data;
    firstFree = data;
    cap = start + n;
    for (size_type i = 0; i < n; ++i) {
        allocTraits::construct(alloc, firstFree++, t);
    }
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(const T* first, const T* last) {
    auto data = Allocate(first, last);
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(std::initializer_list<T> il) {
    auto data = Allocate(il.begin(), il.end());
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
vec<T, Allocator>& vec<T, Allocator>::operator=(std::initializer_list<T> il) {
    auto data = Allocate(il.begin(), il.end());
    free();
    start = data.first;
    firstFree = data.second;
    cap = data.second;
    return *this;
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(const vec<T, Allocator>& rhs) {
    auto data = Allocate(rhs.begin(), rhs.end());
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
vec<T, Allocator>& vec<T, Allocator>::operator=(const vec<T, Allocator>& rhs) {
    if (this != &rhs) {
        auto data = Allocate(rhs.begin(), rhs.end());
        free();
        start = data.first;
        firstFree = data.second;
        cap = data.second;
    }
    return *this;
}

template<typename T, typename Allocator>
std::pair<T*, T*> vec<T, Allocator>::Allocate(const T* b, const T* e) {
    auto dst = allocTraits::allocate(alloc, e - b);
    return {dst, std::uninitialized_copy(b, e, dst)};
}

template<typename T, typename Allocator>
void vec<T, Allocator>::free() {
    if (start) {
        auto p = firstFree;
        while (p != start) {
            allocTraits::destroy(alloc, --p);
        }
        allocTraits::deallocate(alloc, start, cap - start);
    }
}

template<typename T, typename Allocator>
bool operator==(const vec<T, Allocator>& x, const vec<T, Allocator>& y) {
    return x.size() == y.size() && std::equal(x.begin(), x.end(), y.begin());
}

template<typename T, typename Allocator>
bool operator!=(const vec<T, Allocator>& x, const vec<T, Allocator>& y) {
    return !(x == y);
}

template<typename T, typename Allocator>
std::ostream& operator<<(std::ostream& s, const vec<T, Allocator>& v) {
    s.put('{');

    //    // Range-based for loop initialization statements(c++ 20)
    //    for (char comma[]{'\0', ' ', '\0'}; const auto& e : v) {
    //        s << comma << e;
    //        comma[0] = ',';
    //    }

    char comma[]{'\0', ' ', '\0'};
    for (const auto& e: v) {
        s << comma << e;
        comma[0] = ',';
    }
    return s << "}\n";
}

}// namespace XAcceleratorEngine
#endif
