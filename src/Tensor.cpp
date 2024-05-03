//
// Created by richard on 4/28/24.
//
#include "Tensor.hpp"
#include "glog/logging.h"

namespace XAcceleratorEngine {

template<typename T>
Tensor<T>::Tensor(uint32_t size) {
    data_ = arma::Cube<T>(1, size, 1);
    rawDims_ = std::vector<uint32_t>{size};
}

template<typename T>
Tensor<T>::Tensor(uint32_t rows, uint32_t cols) {
    data_ = arma::Cube<T>(rows, cols, 1);
    if (rows == 1) {
        rawDims_ = std::vector<uint32_t>{cols};
    } else {
        rawDims_ = std::vector<uint32_t>{rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    data_ = arma::Cube<T>(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        rawDims_ = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        rawDims_ = std::vector<uint32_t>{rows, cols};
    } else {
        rawDims_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(const std::vector<uint32_t>& shape) {
    CHECK(!shape.empty() && shape.size() <= 3);

    uint32_t remain = 3 - shape.size();
    std::vector<uint32_t> shape_(3, 1);
    std::copy(shape.begin(), shape.end(), shape_.begin() + remain);

    uint32_t channels = shape_[0];
    uint32_t rows = shape_[1];
    uint32_t cols = shape_[2];

    data_ = arma::Cube<T>(rows, cols, channels);
    if (channels == 1 && rows == 1) {
        rawDims_ = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        rawDims_ = std::vector<uint32_t>{rows, cols};
    } else {
        rawDims_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(T* rawPtr, uint32_t size) {
    CHECK_NE(rawPtr, nullptr);
    rawDims_ = std::vector<uint32_t>{size};
    data_ = arma::Cube<T>(rawPtr, 1, size, 1, false, true);
}

template<typename T>
Tensor<T>::Tensor(T* rawPtr, uint32_t rows, uint32_t cols) {
    CHECK_NE(rawPtr, nullptr);
    data_ = arma::Cube<T>(rawPtr, rows, cols, 1, false, true);
    if (rows == 1) {
        rawDims_ = std::vector<uint32_t>{cols};
    } else {
        rawDims_ = std::vector<uint32_t>{rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(T* rawPtr, uint32_t channels, uint32_t rows, uint32_t cols) {
    CHECK_NE(rawPtr, nullptr);
    data_ = arma::Cube<T>(rawPtr, rows, cols, channels, false, true);
    if (channels == 1 && rows == 1) {
        rawDims_ = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        rawDims_ = std::vector<uint32_t>{rows, cols};
    } else {
        rawDims_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(T* rawPtr, const std::vector<uint32_t>& shape) {
    CHECK_NE(rawPtr, nullptr);
    CHECK(!shape.empty() && shape.size() <= 3);

    uint32_t remain = 3 - shape.size();
    std::vector<uint32_t> shape_(3, 1);
    std::copy(shape.begin(), shape.end(), shape_.begin() + remain);

    uint32_t channels = shape_[0];
    uint32_t rows = shape_[1];
    uint32_t cols = shape_[2];

    data_ = arma::Cube<T>(rawPtr, rows, cols, channels, false, true);

    if (channels == 1 && rows == 1) {
        rawDims_ = std::vector<uint32_t>{cols};
    } else if (channels == 1) {
        rawDims_ = std::vector<uint32_t>{rows, cols};
    } else {
        rawDims_ = std::vector<uint32_t>{channels, rows, cols};
    }
}

template<typename T>
Tensor<T>::Tensor(const Tensor<T>& rhs) : data_(rhs.data_), rawDims_(rhs.rawDims_) {}

template<typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& rhs) {
    if (this != &rhs) {
        data_ = rhs.data_;
        rawDims_ = rhs.rawDims_;
    }
    return *this;
}

template<typename T>
Tensor<T>::Tensor(Tensor<T>&& rhs) noexcept : rawDims_(std::move(rhs.rawDims_)), data_(std::move(rhs.data_)) {}

template<typename T>
Tensor<T>& Tensor<T>::operator=(Tensor<T>&& rhs) noexcept {
    if (this != &rhs) {
        data_ = std::move(rhs.data_);
        rawDims_ = std::move(rhs.rawDims_);
    }
    return *this;
}

template<typename T>
void Tensor<T>::Fill(T value) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    data_.fill(value);
}

template<typename T>
void Tensor<T>::Fill(const std::vector<T>& values, bool rowMajor) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    CHECK_EQ(values.size(), data_.size());
    if (rowMajor) {
        uint32_t planes = GetPlaneSize();
        for (uint32_t i = 0; i < GetChannels(); ++i) {
            arma::Mat<T> slice(const_cast<T*>(values.data()) + i * planes, GetCols(), GetRows(), false, true);
            data_.slice(i) = slice.t();
        }
    } else {
        std::copy(values.begin(), values.end(), data_.memptr());
    }
}

template<typename T>
const std::vector<uint32_t>& Tensor<T>::GetRawShape() const {
    CHECK(!rawDims_.empty());
    CHECK_LE(rawDims_.size(), 3);
    CHECK_GE(rawDims_.size(), 1);
    return rawDims_;
}

template<typename T>
std::vector<uint32_t> Tensor<T>::GetShape() const {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    return {GetChannels(), GetRows(), GetCols()};
}

template<typename T>
uint32_t Tensor<T>::GetChannels() const {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    return data_.n_slices;
}

template<typename T>
uint32_t Tensor<T>::GetRows() const {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    return data_.n_rows;
}

template<typename T>
uint32_t Tensor<T>::GetCols() const {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    return data_.n_cols;
}

template<typename T>
size_t Tensor<T>::GetSize() const {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    return data_.size();
}

template<typename T>
size_t Tensor<T>::GetPlaneSize() const {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    return GetCols() * GetCols();
}

template<typename T>
bool Tensor<T>::empty() const {
    return data_.empty();
}

template<typename T>
void Tensor<T>::SetData(const arma::Cube<T>& data) {
    CHECK(data.n_rows == data_.n_rows) << data.n_rows << " != " << data_.n_rows;
    CHECK(data.n_cols == data_.n_cols) << data.n_cols << " != " << data_.n_cols;
    CHECK(data.n_slices == data_.n_slices) << data.n_slices << " != " << data_.n_slices;
    data_ = data;
}

template<typename T>
T& Tensor<T>::index(uint32_t offset) {
    CHECK(offset < data_.size()) << "Tensor index out of bound";
    return data_.at(offset);
}

template<typename T>
const T& Tensor<T>::index(uint32_t offset) const {
    CHECK(offset < data_.size()) << "Tensor index out of bound";
    return data_.at(offset);
}

template<typename T>
arma::Cube<T>& Tensor<T>::data() {
    return data_;
}

template<typename T>
const arma::Cube<T>& Tensor<T>::data() const {
    return data_;
}

template<typename T>
arma::Mat<T>& Tensor<T>::slice(uint32_t channel) {
    CHECK_LT(channel, GetChannels());
    return data_.slice(channel);
}

template<typename T>
const arma::Mat<T>& Tensor<T>::slice(uint32_t channel) const {
    CHECK_LT(channel, GetChannels());
    return data_.slice(channel);
}

template<typename T>
T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) {
    CHECK_LT(channel, GetChannels());
    CHECK_LT(row, GetRows());
    CHECK_LT(col, GetCols());
    return data_.at(row, col, channel);
}

template<typename T>
const T& Tensor<T>::at(uint32_t channel, uint32_t row, uint32_t col) const {
    CHECK_LT(channel, GetChannels());
    CHECK_LT(row, GetRows());
    CHECK_LT(col, GetCols());
    return data_.at(row, col, channel);
}

template<typename T>
void Tensor<T>::Padding(const std::vector<uint32_t>& pads, T value) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    CHECK_EQ(pads.size(), 4);
    uint32_t up = pads[0];
    uint32_t bottom = pads[1];
    uint32_t left = pads[2];
    uint32_t right = pads[3];

    arma::Cube<T> newData(data_.n_rows + up + bottom,
                          data_.n_cols + left + right,
                          data_.n_slices);
    newData.fill(value);
    newData.subcube(up,
                    left,
                    0,
                    newData.n_rows - bottom - 1,
                    newData.n_cols - right - 1,
                    newData.n_slices - 1) = data_;
    data_ = std::move(newData);
    rawDims_ = std::vector<uint32_t>{GetChannels(), GetRows(), GetCols()};
}

template<typename T>
void Tensor<T>::Ones() {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    Fill(static_cast<T>(1));
}

template<>
void Tensor<float>::RandN(float mean, float var) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    std::random_device rd;
    std::mt19937 mt(rd());
    std::normal_distribution<float> dist(mean, var);
    for (size_t i = 0; i < GetSize(); ++i) {
        index(i) = dist(mt);
    }
}

template<>
void Tensor<int32_t>::RandU(int32_t min, int32_t max) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int32_t> dist(min, max);
    for (size_t i = 0; i < GetSize(); ++i) {
        index(i) = dist(mt);
    }
}

template<>
void Tensor<uint8_t>::RandU(uint8_t min, uint8_t max) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    std::random_device rd;
    std::mt19937 mt(rd());
#ifdef _MSC_VER
    std::uniform_int_distribution<int32_t> dist(min, max);
    for (uin32_t i = 0; i < GetSize(); ++i) {
        index(i) = dist(mt) % std::numeric_limits<uint8_t>::max();
    }
#else
    std::uniform_int_distribution<uint8_t> dist(min, max);
    for (size_t i = 0; i < GetSize(); ++i) {
        index(i) = dist(mt);
    }
#endif
}

template<>
void Tensor<float>::RandU(float min, float max) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    CHECK(min <= max);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for (size_t i = 0; i < GetSize(); ++i) {
        index(i) = dist(mt);
    }
}

template<typename T>
T* Tensor<T>::RawPtr() {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    return data_.memptr();
}

template<typename T>
const T* Tensor<T>::RawPtr() const {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    return data_.memptr();
}

template<typename T>
T* Tensor<T>::RawPtr(uint32_t offset) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    CHECK_LT(offset, GetSize());
    return data_.memptr() + offset;
}

template<typename T>
const T* Tensor<T>::RawPtr(uint32_t offset) const {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    CHECK_LT(offset, GetSize());
    return data_.memptr() + offset;
}

template<typename T>
const T* Tensor<T>::MatrixRawPtr(uint32_t index) const {
    CHECK_LT(index, GetChannels());
    size_t offset = index * GetPlaneSize();
    CHECK_LE(offset, GetSize());
    return RawPtr(offset);
}

template<typename T>
T* Tensor<T>::MatrixRawPtr(uint32_t index) {
    CHECK_LT(index, GetChannels());
    size_t offset = index * GetPlaneSize();
    CHECK_LE(offset, GetSize());
    return RawPtr(offset);
}

template<typename T>
void Tensor<T>::Reshape(const std::vector<uint32_t>& shape, bool rowMajor) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    CHECK(!shape.empty());
    size_t currentSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    CHECK_LE(shape.size(), 3);
    CHECK_EQ(GetSize(), currentSize);
    if (!rowMajor) {
        if (shape.size() == 3) {
            data_.reshape(shape[1], shape[2], shape[0]);
            rawDims_ = {shape[0], shape[1], shape[2]};
        } else if (shape.size() == 2) {
            data_.reshape(shape[0], shape[1], 1);
            rawDims_ = {shape[0], shape[1]};
        } else {
            data_.reshape(1, shape[0], 1);
            rawDims_ = {shape[0]};
        }
    } else {
        if (shape.size() == 3) {
            Review(shape);
            rawDims_ = shape;
        } else if (shape.size() == 2) {
            Review({1, shape[0], shape[1]});
            rawDims_ = shape;
        } else {
            Review({1, 1, shape[0]});
            rawDims_ = shape;
        }
    }
}

template<typename T>
void Tensor<T>::Flatten(bool rowMajor) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    uint32_t size = GetSize();
    Reshape({size}, rowMajor);
}

template<typename T>
void Tensor<T>::Transform(const std::function<T(T)>& filter) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    data_.transform(filter);
}

template<typename T>
void Tensor<T>::Review(const std::vector<uint32_t>& shape) {
    CHECK(!data_.empty()) << "The data area of the tensor is empty.";
    CHECK_EQ(shape.size(), 3);
    uint32_t targetCh = shape[0];
    uint32_t targetRows = shape[1];
    uint32_t targetCols = shape[2];

    CHECK_EQ(GetSize(), targetCh * targetRows * targetCols);
    arma::Cube<T> newData(targetRows, targetCols, targetCh);
    uint32_t targetPlaneSize = targetRows * targetCols;
#pragma omp parallel for
    for (uint32_t ch = 0; ch < GetChannels(); ++ch) {
        uint32_t planeStart = ch * GetRows() * GetCols();
        for (uint32_t srcCol = 0; srcCol < GetCols(); ++srcCol) {
            const T* colPtr = data_.slice_colptr(ch, srcCol);
            for (uint32_t srcRow = 0; srcRow < GetRows(); ++srcRow) {
                uint32_t idx = planeStart + srcRow * GetCols() + srcCol;
                uint32_t dstCh = idx / targetPlaneSize;
                uint32_t dstChOffset = idx % targetPlaneSize;
                uint32_t dstRow = dstChOffset / targetCols;
                uint32_t dstCol = dstChOffset % targetCols;
                newData.at(dstRow, dstCol, dstCh) = *(colPtr + srcRow);
            }
        }
    }
    data_ = std::move(newData);
}

template<typename T>
void Tensor<T>::Show() {
    for (uint32_t i = 0; i < GetChannels(); ++i) {
        LOG(INFO) << "Channel: " << i
                  << "\n"
                  << slice(i);
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> CreateTensor(uint32_t channels, uint32_t rows, uint32_t cols) {
    return std::make_shared<Tensor<T>>(channels, rows, cols);
}

template<typename T>
std::shared_ptr<Tensor<T>> CreateTensor(uint32_t rows, uint32_t cols) {
    return std::make_shared<Tensor<T>>(1, rows, cols);
}

template<typename T>
std::shared_ptr<Tensor<T>> CreateTensor(uint32_t size) {
    return std::make_shared<Tensor<T>>(1, 1, size);
}

template<typename T>
std::shared_ptr<Tensor<T>> CreateTensor(const std::vector<uint32_t>& shape) {
    return std::make_shared<Tensor<T>>(shape);
}

template<typename T>
bool TensorIsSame(const std::shared_ptr<Tensor<T>>& a, const std::shared_ptr<Tensor<T>>& b, T threshold) {
    CHECK(a != nullptr);
    CHECK(b != nullptr);
    if (a->GetShape() != b->GetShape()) {
        return false;
    }
    return arma::approx_equal(a->data(), b->data(), "absdiff", threshold);
}

template<typename T>
std::shared_ptr<Tensor<T>> CloneTensor(const std::shared_ptr<Tensor<T>>& tensor) {
    return std::make_shared<Tensor<T>>(*tensor);
}

template<typename T>
std::tuple<std::shared_ptr<Tensor<T>>, std::shared_ptr<Tensor<T>>> BroadcastTensor(
        const std::shared_ptr<Tensor<T>>& tensor1, const std::shared_ptr<Tensor<T>>& tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->GetShape() == tensor2->GetShape()) {
        return {tensor1, tensor2};
    } else {
        CHECK(tensor1->GetChannels() == tensor2->GetChannels());
        if (tensor2->GetRows() == 1 && tensor2->GetCols() == 1) {
            std::shared_ptr<Tensor<T>> newTensor = CreateTensor<T>(tensor2->GetChannels(),
                                                                   tensor1->GetRows(),
                                                                   tensor1->GetCols());
            for (uint32_t i = 0; i < tensor2->GetChannels(); ++i) {
                T* ptr = newTensor->MatrixRawPtr(i);
                std::fill(ptr, ptr + newTensor->GetPlaneSize(), tensor2->index(i));
            }
            return {tensor1, newTensor};
        } else if (tensor1->GetRows() == 1 && tensor1->GetCols() == 1) {
            std::shared_ptr<Tensor<T>> newTensor = CreateTensor<T>(tensor1->GetChannels(),
                                                                   tensor2->GetRows(),
                                                                   tensor2->GetCols());
            for (uint32_t i = 0; i < tensor1->GetChannels(); ++i) {
                T* ptr = newTensor->MatrixRawPtr(i);
                std::fill(ptr, ptr + newTensor->GetPlaneSize(), tensor1->index(i));
            }
            return {newTensor, tensor2};
        } else {
            LOG(FATAL) << "Broadcast shapes are not adapting.";
            return {tensor1, tensor2};
        }
    }
}

template<typename T>
std::shared_ptr<Tensor<T>> TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                                            const std::shared_ptr<Tensor<T>>& tensor2) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr);
    if (tensor1->GetShape() == tensor2->GetShape()) {
        std::shared_ptr<Tensor<T>> output = CreateTensor<Tensor<T>>(tensor1->GetShape());
        output->SetData(tensor1->data() + tensor2->data());
        return output;
    } else {
        // broadcast
        CHECK(tensor1->GetChannels() == tensor2->GetChannels()) << "Tensor shapes are not adapting.";
        const auto& [inputTensor1, inputTensor2] = BroadcastTensor(tensor1, tensor2);
        std::shared_ptr<Tensor<T>> output = CreateTensor<Tensor<T>>(inputTensor1->GetShape());
        output->SetData(inputTensor1->data() + inputTensor2->data());
        return output;
    }
}

template<typename T>
void TensorElementAdd(const std::shared_ptr<Tensor<T>>& tensor1,
                      const std::shared_ptr<Tensor<T>>& tensor2,
                      const std::shared_ptr<Tensor<T>>& output) {
    CHECK(tensor1 != nullptr && tensor2 != nullptr && output != nullptr);
    if (tensor1->GetShape() == tensor2->GetShape()) {
        CHECK(tensor1->GetShape() == output->GetShape());
        output->SetData(tensor1->data() + tensor2->data());
    } else {
        // broadcast
        CHECK(tensor1->GetChannels() == tensor2->GetChannels()) << "Tensor shapes are not adapting.";
        const auto& [inputTensor1, inputTensor2] = BroadcastTensor(tensor1, tensor2);
        CHECK(inputTensor1->GetShape() == output->GetShape());
        output->SetData(inputTensor1->data() + inputTensor2->data());
    }
}

template class Tensor<float>;
template class Tensor<uint32_t>;
template class Tensor<uint8_t>;
}// namespace XAcceleratorEngine