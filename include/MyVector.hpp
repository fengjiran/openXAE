//
// Created by richard on 5/4/24.
//
#ifndef OPENXAE_MYVECTOR_HPP
#define OPENXAE_MYVECTOR_HPP

#include "MyAllocator.hpp"
#include "MyIterator.hpp"
#include "config.hpp"
#include "glog/logging.h"

namespace XAcceleratorEngine {

template<typename T>
struct type_identity {
    using type = T;
};

template<typename T>
using type_identity_t = typename type_identity<T>::type;

template<typename T, typename Allocator = MyAllocator<T> /* = std::allocator<T>*/>
class vec {
public:
    using value_type = T;
    using allocator_type = Allocator;
    using alloc_traits = std::allocator_traits<allocator_type>;
    using size_type = typename alloc_traits::size_type;
    using difference_type = typename alloc_traits::difference_type;
    using pointer = typename alloc_traits::pointer;
    using const_pointer = typename alloc_traits::const_pointer;
    using reference = value_type&;
    using const_reference = const value_type&;

    using iterator = MyIterator<pointer>;
    using const_iterator = MyIterator<const_pointer>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    static_assert((std::is_same<typename allocator_type::value_type, value_type>::value),
                  "Allocator::value_type must be same type as value_type");

public:
    /**
     * @brief Default constructor
     */
    vec() noexcept(noexcept(allocator_type()))
        : start(nullptr), cap(nullptr), firstFree(nullptr), alloc(allocator_type()) {}

    explicit vec(const allocator_type& alloc_) noexcept
        : start(nullptr), cap(nullptr), firstFree(nullptr), alloc(alloc_) {}

    /**
     * @brief Constructor with size and default initialization
     *
     * @param n Size
     */
    explicit vec(size_type n, const allocator_type& alloc_ = allocator_type());

    /**
     * @brief Constructor with size and initial value
     *
     * @param n Size
     * @param t Initial value
     */
    vec(size_type n, const_reference value, const allocator_type& alloc_ = allocator_type());

    /**
     * @brief Constructor with range [first, last)
     *
     * @param first first ptr
     * @param last last ptr
     */
    vec(const_pointer first, const_pointer last);

    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, iterator>::type = 0>
    vec(InputIterator first, InputIterator last, const allocator_type& alloc_ = allocator_type());


    /**
     * @brief Constructor with initializer list
     *
     * @param il Initializer list
     */
    vec(std::initializer_list<T> il, const allocator_type& alloc_ = allocator_type());

    vec& operator=(std::initializer_list<T> il);

    /**
     * @brief Copy constructor
     *
     * @param rhs Right Hand Side
     */
    vec(const vec& rhs);

    vec(const vec& rhs, const type_identity_t<allocator_type>& alloc_);

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

    vec(vec&& rhs, const type_identity_t<allocator_type>& alloc_);

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
     * @brief Get the first element iterator of the vector
     *
     * @return The pointer of the first element
     */
    iterator begin() noexcept { return MakeIter(start); }

    const_iterator begin() const noexcept { return MakeIter(start); }

    /**
     * @brief Get the ptr to the element following the last element of the vector
     *
     * @return The ptr to the element following the last element of the vector
     */
    //    pointer end() const { return firstFree; }

    iterator end() noexcept { return MakeIter(firstFree); }

    const_iterator end() const noexcept { return MakeIter(firstFree); }

    CPP_NODISCARD bool empty() const { return firstFree == start; }

    void reserve(size_type n);
    void resize(size_type n);
    void resize(size_type n, const_reference t);
    pointer data() { return start; }

    void push_back(const_reference t);
    void push_back(value_type&& t);

    template<typename... Args>
    void emplace_back(Args&&... args);

    reference operator[](size_type pos) noexcept;
    const_reference operator[](size_type pos) const noexcept;

    reference at(size_type pos);
    const_reference at(size_type pos) const;

    reference front() noexcept;
    const_reference front() const noexcept;

    /**
     * @brief Get the reference to the last element in the container.
     *
     * @return Reference to the last element.
     */
    reference back() noexcept;

    /**
     * @brief Get the reference to the last element in the container.
     *
     * @return Reference to the last element.
     */
    const_reference back() const noexcept;

    ~vec();

private:
    std::pair<pointer, pointer> Allocate(const_pointer b, const_pointer e);

    template<typename InputIterator,
             typename has_input_iterator_category<InputIterator, iterator>::type = 0>
    std::pair<pointer, pointer> Allocate(InputIterator first, InputIterator last);

    void free();
    void reallocate();
    void reallocate(size_type newCap);
    void CheckAndAlloc() {
        if (firstFree == cap) {
            reallocate();
        }
    }

    iterator MakeIter(pointer p) noexcept {
        return iterator(p);
    }

    const_iterator MakeIter(const_pointer p) const noexcept {
        return const_iterator(p);
    }

private:
    allocator_type alloc;
    pointer start;
    pointer cap;
    pointer firstFree;
};

template<typename T, typename Allocator>
typename vec<T, Allocator>::const_reference
vec<T, Allocator>::back() const noexcept {
    CHECK(!empty()) << "back() called on an empty vector";
    return *(firstFree - 1);
}

template<typename T, typename Allocator>
typename vec<T, Allocator>::reference
vec<T, Allocator>::back() noexcept {
    CHECK(!empty()) << "back() called on an empty vector";
    return *(firstFree - 1);
}

template<typename T, typename Allocator>
typename vec<T, Allocator>::const_reference
vec<T, Allocator>::front() const noexcept {
    CHECK(!empty()) << "front() called on an empty vector";
    return *start;
}

template<typename T, typename Allocator>
typename vec<T, Allocator>::reference
vec<T, Allocator>::front() noexcept {
    CHECK(!empty()) << "front() called on an empty vector";
    return *start;
}

template<typename T, typename Allocator>
typename vec<T, Allocator>::const_reference
vec<T, Allocator>::operator[](size_type pos) const noexcept {
    CHECK(pos < size()) << "vector[] index out of bounds";
    return start[pos];
}

template<typename T, typename Allocator>
typename vec<T, Allocator>::reference
vec<T, Allocator>::operator[](size_type pos) noexcept {
    CHECK(pos < size()) << "vector[] index out of bounds";
    return start[pos];
}

template<typename T, typename Allocator>
typename vec<T, Allocator>::reference
vec<T, Allocator>::at(size_type pos) {
    if (pos >= size()) {
        throw std::out_of_range("index out of bounds");
    }
    return (*this)[pos];
}

template<typename T, typename Allocator>
typename vec<T, Allocator>::const_reference
vec<T, Allocator>::at(size_type pos) const {
    if (pos >= size()) {
        throw std::out_of_range("index out of bounds");
    }
    return (*this)[pos];
}

template<typename T, typename Allocator>
template<typename... Args>
void vec<T, Allocator>::emplace_back(Args&&... args) {
    CheckAndAlloc();
    alloc_traits::construct(alloc, firstFree++, std::forward<Args>(args)...);
}

template<typename T, typename Allocator>
void vec<T, Allocator>::push_back(value_type&& t) {
    CheckAndAlloc();
    alloc_traits::construct(alloc, firstFree++, std::move(t));
}

template<typename T, typename Allocator>
void vec<T, Allocator>::push_back(const_reference t) {
    CheckAndAlloc();
    alloc_traits::construct(alloc, firstFree++, t);
}

template<typename T, typename Allocator>
void vec<T, Allocator>::resize(size_type n) {
    if (n > size()) {
        while (size() < n) {
            push_back(T());
        }
    } else {
        while (size() > n) {
            alloc_traits::destroy(alloc, --firstFree);
        }
    }
}

template<typename T, typename Allocator>
void vec<T, Allocator>::resize(size_type n, const_reference t) {
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
    auto data = alloc_traits::allocate(alloc, newCap);
    auto src = start;
    auto dst = data;
    for (size_type i = 0; i < size(); ++i) {
        alloc_traits::construct(alloc, dst, std::move(*src));
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
    auto data = alloc_traits::allocate(alloc, newCap);
    auto src = start;
    auto dst = data;
    for (size_type i = 0; i < size(); ++i) {
        alloc_traits::construct(alloc, dst, std::move(*src));
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
vec<T, Allocator>::vec(vec&& rhs) noexcept
    : start(rhs.start), firstFree(rhs.firstFree), cap(rhs.cap), alloc(std::move(rhs.alloc)) {
    rhs.start = nullptr;
    rhs.firstFree = nullptr;
    rhs.cap = nullptr;
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(vec&& rhs, const type_identity_t<allocator_type>& alloc_)
    : start(rhs.start), firstFree(rhs.firstFree), cap(rhs.cap), alloc(alloc_) {
    rhs.start = nullptr;
    rhs.firstFree = nullptr;
    rhs.cap = nullptr;
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(size_type n, const allocator_type& alloc_)
    : alloc(alloc_) {
    auto data = alloc_traits::allocate(alloc, n);
    start = data;
    firstFree = data;
    cap = start + n;
    for (size_type i = 0; i < n; ++i) {
        alloc_traits::construct(alloc, firstFree++, T());
    }
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(size_type n, const_reference value, const allocator_type& alloc_)
    : alloc(alloc_) {
    auto data = alloc_traits::allocate(alloc, n);
    start = data;
    firstFree = data;
    cap = start + n;
    for (size_type i = 0; i < n; ++i) {
        alloc_traits::construct(alloc, firstFree++, value);
    }
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(const_pointer first, const_pointer last) {
    auto data = Allocate(first, last);
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, typename vec<T, Allocator>::iterator>::type>
vec<T, Allocator>::vec(InputIterator first, InputIterator last, const allocator_type& alloc_)
    : alloc(alloc_) {
    auto data = Allocate(first, last);
    start = data.first;
    firstFree = data.second;
    cap = data.second;
}

template<typename T, typename Allocator>
vec<T, Allocator>::vec(std::initializer_list<T> il, const allocator_type& alloc_)
    : alloc(alloc_) {
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
vec<T, Allocator>::vec(const vec<T, Allocator>& rhs, const type_identity_t<allocator_type>& alloc_)
    : alloc(alloc_) {
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
std::pair<typename vec<T, Allocator>::pointer, typename vec<T, Allocator>::pointer>
vec<T, Allocator>::Allocate(const_pointer b, const_pointer e) {
    auto dst = alloc_traits::allocate(alloc, e - b);
    return {dst, std::uninitialized_copy(b, e, dst)};
}

template<typename T, typename Allocator>
template<typename InputIterator,
         typename has_input_iterator_category<InputIterator, typename vec<T, Allocator>::iterator>::type>
std::pair<typename vec<T, Allocator>::pointer, typename vec<T, Allocator>::pointer>
vec<T, Allocator>::Allocate(InputIterator first, InputIterator last) {
    auto dst = alloc_traits::allocate(alloc, last - first);
    return {dst, std::uninitialized_copy(first, last, dst)};
}

template<typename T, typename Allocator>
void vec<T, Allocator>::free() {
    if (start) {
        auto p = firstFree;
        while (p != start) {
            alloc_traits::destroy(alloc, --p);
        }
        alloc_traits::deallocate(alloc, start, cap - start);
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
