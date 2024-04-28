//
// Created by richard on 4/28/24.
//

#ifndef OPENXAE_TENSOR_HPP
#define OPENXAE_TENSOR_HPP

#include <armadillo>
#include <vector>

namespace XAcceleratorEngine {

template<typename T>
class Tensor {
public:
    /**
     * @brief Construct a new empty Tensor.
     */
    Tensor() = default;

    /**
     * @brief Construct a 1D tensor
     *
     * @param size vector size
     */
    explicit Tensor(uint32_t size);

    /**
     * @brief Construct a 2D tensor
     *
     * @param rows number of rows
     * @param cols number of cols
     */
    Tensor(uint32_t rows, uint32_t cols);

    /**
     * @brief Construct a 3D tensor
     *
     * @param channels number of channels
     * @param rows number of rows
     * @param cols number of cols
     */
    Tensor(uint32_t channels, uint32_t rows, uint32_t cols);

    /**
     * @brief Construct a tensor with shape
     *
     * @param shape tensor dimension
     */
    explicit Tensor(const std::vector<uint32_t>& shape);

    /**
     * @brief Construct a 1D tensor with raw ptr
     *
     * @param rawPtr raw pointer of data
     * @param size vector size
     */
    Tensor(T* rawPtr, uint32_t size);

    /**
     * @brief Construct a 2D tensor with raw ptr
     *
     * @param rawPtr raw pointer of data
     * @param rows number of rows
     * @param cols number of cols
     */
    Tensor(T* rawPtr, uint32_t rows, uint32_t cols);

    /**
     * @brief Construct a 3D tensor with raw ptr
     *
     * @param rawPtr raw pointer of data
     * @param channels number of channels
     * @param rows number of rows
     * @param cols number of cols
     */
    Tensor(T* rawPtr, uint32_t channels, uint32_t rows, uint32_t cols);

    /**
     * @brief Construct a tensor with raw ptr and shape
     *
     * @param rawPtr raw pointer of data
     * @param shape tensor dimension
     */
    Tensor(T* rawPtr, const std::vector<uint32_t>& shape);

    /**
     * @brief Copy constructor
     */
    Tensor(const Tensor& rhs);

    /**
     * @brief Copy assignment
     */
    Tensor& operator=(const Tensor& rhs);

    /**
     * @brief Move constructor
     */
    Tensor(Tensor&& rhs) noexcept;

    /**
     * @brief Move assignment
     */
    Tensor& operator=(Tensor&& rhs) noexcept;

    /**
     * @brief Fills tensor with value
     *
     * @param value Fill value
     */
    void Fill(T value);

    /**
     * @brief Fills tensor with vector data
     *
     * @param values Fill value
     * @param rowMajor Fill by row major order
     */
    void Fill(const std::vector<T>& values, bool rowMajor = true);

    /**
     * @brief Get raw tensor shape
     *
     *@return Raw tensor dimensions
     */
    const std::vector<uint32_t>& GetRawShape() const;

    /**
     * @brief Get raw tensor shape
     *
     *@return Raw tensor dimensions
     */
    std::vector<uint32_t> GetShape() const;

    /**
     *@brief Print tensor
     */
    void Show();

    /**
     * @brief Get number of tensor channels
     *
     * @return Number of tensor channels
     */
    uint32_t GetChannels() const;

    /**
     * @brief Get number of rows
     *
     * @return Number of tensor rows
     */
    uint32_t GetRows() const;

    /**
     * @brief Get number of cols
     *
     * @return Number of tensor cols
     */
    uint32_t GetCols() const;

    /**
     * @brief Gets total number of elements
     *
     * @return Total number of elements
     */
    size_t GetSize() const;

    /**
     * @brief Gets actually total number of elements
     *
     * @return Total actually number of elements
     */
    size_t GetPlaneSize() const;

    /**
     * @brief Checks if tensor is empty
     *
     * @return True if empty, false otherwise
     */
    bool empty() const;

    /**
     * @brief Set the tensor data
     *
     * @param data Data to set
     */
    void SetData(const arma::Cube<T>& data);

    /**
     * @brief Get the element reference at offset
     *
     * @param offset Element offset
     * @return Element reference
     */
    T& index(uint32_t offset);

    /**
     * @brief Get the element reference at offset
     *
     * @param offset Element offset
     * @return Element value
     */
    const T& index(uint32_t offset) const;

    /**
     * @brief Get the tensor data
     *
     * @return Tensor data reference
     */
    arma::Cube<T>& data();

    /**
     * @brief Get the tensor data
     *
     * @return Tensor data const reference
     */
    const arma::Cube<T>& data() const;

    /**
     * @brief Get the channel matrix
     *
     * @param channel Channel index
     * @return Channel matrix reference
     */
    arma::Mat<T>& slice(uint32_t channel);

    /**
     * @brief Get the channel matrix const reference
     *
     * @param channel Channel index
     * @return Channel matrix const reference
     */
    const arma::Mat<T>& slice(uint32_t channel) const;

    /**
     * @brief Get the element at location
     *
     * @param channel Channel index
     * @param row Row index
     * @param col Col index
     * @return Element at location
     */
    T& at(uint32_t channel, uint32_t row, uint32_t col);

    /**
     * @brief Get the element ref at location
     *
     * @param channel Channel index
     * @param row Row index
     * @param col Col index
     * @return Element ref at location
     */
    const T& at(uint32_t channel, uint32_t row, uint32_t col) const;

    /**
     * @brief Padding the tensor
     *
     * @param pads Padding amount for dimensions, the length must be 4
     * @param value Padding value
     */
    void Padding(const std::vector<uint32_t>& pads, T value);

    /**
     * @brief Fill with ones
     */
    void Ones();

    /**
     * @brief Initialize with normal distribution
     *
     * @param mean Mean
     * @param var Variance
     */
    void RandN(T mean = 0, T var = 1);

    /**
     * @brief Initialize with uniform distribution
     *
     * @param min Minimum value
     * @param max Maximum value
     */
    void RandU(T min = 0, T max = 1);

    /**
     * @brief Get the raw data pointer
     *
     * @return Raw data pointer
     */
    T* RawPtr();

    /**
     * @brief Get the const raw data pointer
     *
     * @return Const raw data pointer
     */
    const T* RawPtr() const;

    /**
     * @brief Get the raw data pointer with offset
     *
     * @return Raw data pointer + offset
     */
    T* RawPtr(uint32_t offset);

    /**
     * @brief Get the const raw data pointer with offset
     *
     * @return Raw data pointer + offset
     */
    const T* RawPtr(uint32_t offset) const;

private:
    /// Raw tensor dimensions
    std::vector<uint32_t> rawDims_;
    /// Tensor data
    arma::Cube<T> data_;
};
}// namespace XAcceleratorEngine
#endif//OPENXAE_TENSOR_HPP
