#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace milk {

/// @brief Matrix transpose selector for row-major matrix kernels.
enum class Transpose {
    No,
    Yes
};

/// @brief Scalar types supported by milk vector and matrix containers.
template <typename T>
concept Scalar = std::is_same_v<T, int> || std::is_same_v<T, float> || std::is_same_v<T, double>;

template <Scalar T = double>
class Vector;

template <Scalar T = double>
class Matrix;

template <std::floating_point T = double>
class Solver;

/// @brief Byte alignment used for milk-owned contiguous numeric storage.
inline constexpr std::size_t kAlignment = 32;

namespace detail {

template <typename Expr>
using bare_t = std::remove_cvref_t<Expr>;

template <typename T>
concept SolverScalar = std::floating_point<T> || std::same_as<T, std::complex<double>>;

template <typename T>
struct is_vector_leaf : std::false_type {};

template <Scalar T>
struct is_vector_leaf<Vector<T>> : std::true_type {};

template <typename T>
struct is_matrix_leaf : std::false_type {};

template <Scalar T>
struct is_matrix_leaf<Matrix<T>> : std::true_type {};

template <typename Expr>
inline constexpr bool is_vector_leaf_v = is_vector_leaf<bare_t<Expr>>::value;

template <typename Expr>
inline constexpr bool is_matrix_leaf_v = is_matrix_leaf<bare_t<Expr>>::value;

template <typename Expr>
using vector_expr_storage_t = std::conditional_t<is_vector_leaf_v<Expr>, const bare_t<Expr>&, bare_t<Expr>>;

template <typename Expr>
using matrix_expr_storage_t = std::conditional_t<is_matrix_leaf_v<Expr>, const bare_t<Expr>&, bare_t<Expr>>;

template <typename Expr>
concept VectorExpression = requires(const bare_t<Expr>& expr, std::size_t index) {
    typename bare_t<Expr>::value_type;
    { expr.size() } -> std::convertible_to<std::size_t>;
    { expr.eval(index) } -> std::convertible_to<typename bare_t<Expr>::value_type>;
};

template <typename Expr>
concept MatrixExpression = requires(const bare_t<Expr>& expr, std::size_t row, std::size_t col) {
    typename bare_t<Expr>::value_type;
    { expr.rows() } -> std::convertible_to<std::size_t>;
    { expr.cols() } -> std::convertible_to<std::size_t>;
    { expr.eval(row, col) } -> std::convertible_to<typename bare_t<Expr>::value_type>;
};

template <Scalar T>
inline T* allocate(std::size_t count) {
    if (count == 0) {
        return nullptr;
    }
    return static_cast<T*>(::operator new[](count * sizeof(T), std::align_val_t{kAlignment}));
}

template <Scalar T>
inline void deallocate(T* ptr) noexcept {
    if (ptr != nullptr) {
        ::operator delete[](ptr, std::align_val_t{kAlignment});
    }
}

template <typename T>
inline void fill(T* ptr, std::size_t count, const T& value) {
    if (count != 0) {
        std::fill_n(ptr, count, value);
    }
}

template <typename T>
inline void copy(const T* src, T* dst, std::size_t count) {
    if (count != 0) {
        std::copy_n(src, count, dst);
    }
}

template <typename T>
inline void axpy(std::size_t count, T alpha, const T* x, int incx, T* y, int incy) {
    for (std::size_t index = 0; index < count; ++index) {
        y[index * static_cast<std::size_t>(incy)] += alpha * x[index * static_cast<std::size_t>(incx)];
    }
}

template <typename T>
inline void copy_strided(std::size_t count, const T* src, int incx, T* dst, int incy) {
    for (std::size_t index = 0; index < count; ++index) {
        dst[index * static_cast<std::size_t>(incy)] = src[index * static_cast<std::size_t>(incx)];
    }
}

template <typename T>
inline T dot(const T* lhs, int incx, const T* rhs, int incy, std::size_t size) {
    T sum = T{};
    for (std::size_t index = 0; index < size; ++index) {
        sum += lhs[index * static_cast<std::size_t>(incx)] * rhs[index * static_cast<std::size_t>(incy)];
    }
    return sum;
}

template <typename T>
inline double magnitude(const T& value) {
    using std::abs;
    return static_cast<double>(abs(value));
}

inline double magnitude(const std::complex<double>& value) {
    const double real = value.real();
    const double imag = value.imag();
    const double ar   = std::abs(real);
    const double ai   = std::abs(imag);
    const double mx   = std::max(ar, ai);
    return (mx < 1.0e150 && mx > 1.0e-150)
               ? std::sqrt(real * real + imag * imag)
               : std::abs(value);
}

template <SolverScalar T>
inline double solver_tolerance() {
    if constexpr (std::same_as<T, float>) {
        return 1.0e-6;
    } else {
        return 1.0e-12;
    }
}

template <SolverScalar T>
inline void swap_rows(T* matrix, std::size_t cols, std::size_t lhs, std::size_t rhs) {
    if (lhs == rhs) {
        return;
    }
    for (std::size_t col = 0; col < cols; ++col) {
        std::swap(matrix[lhs * cols + col], matrix[rhs * cols + col]);
    }
}

template <SolverScalar T>
inline void lu_factorize_inplace(T* a, std::size_t n, std::size_t lda, int* ipiv) {
    const double eps = solver_tolerance<T>();

    for (std::size_t pivot_col = 0; pivot_col < n; ++pivot_col) {
        std::size_t pivot_row = pivot_col;
        double      pivot_abs = magnitude(a[pivot_col * lda + pivot_col]);
        for (std::size_t row = pivot_col + 1; row < n; ++row) {
            const double candidate = magnitude(a[row * lda + pivot_col]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }

        if (pivot_abs <= eps) {
            throw std::runtime_error("milk: singular matrix");
        }

        swap_rows(a, lda, pivot_col, pivot_row);
        if (ipiv != nullptr) {
            ipiv[pivot_col] = static_cast<int>(pivot_row + 1);
        }

        const T pivot = a[pivot_col * lda + pivot_col];
        for (std::size_t row = pivot_col + 1; row < n; ++row) {
            a[row * lda + pivot_col] /= pivot;
            const T factor = a[row * lda + pivot_col];
            for (std::size_t col = pivot_col + 1; col < n; ++col) {
                a[row * lda + col] -= factor * a[pivot_col * lda + col];
            }
        }
    }
}

template <SolverScalar T>
inline void lu_solve_inplace(const T* lu, std::size_t n, std::size_t lda,
                             const int* ipiv, T* b, std::size_t nrhs, std::size_t ldb) {
    const double eps = solver_tolerance<T>();

    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        const std::size_t pivot_row = static_cast<std::size_t>(ipiv[pivot] - 1);
        if (pivot_row != pivot) {
            swap_rows(b, ldb, pivot, pivot_row);
        }
    }

    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t rhs = 0; rhs < nrhs; ++rhs) {
            T value = b[row * ldb + rhs];
            for (std::size_t col = 0; col < row; ++col) {
                value -= lu[row * lda + col] * b[col * ldb + rhs];
            }
            b[row * ldb + rhs] = value;
        }
    }

    for (std::size_t offset = 0; offset < n; ++offset) {
        const std::size_t row  = n - 1 - offset;
        const T           diag = lu[row * lda + row];
        if (magnitude(diag) <= eps) {
            throw std::runtime_error("milk: singular matrix");
        }
        for (std::size_t rhs = 0; rhs < nrhs; ++rhs) {
            T value = b[row * ldb + rhs];
            for (std::size_t col = row + 1; col < n; ++col) {
                value -= lu[row * lda + col] * b[col * ldb + rhs];
            }
            b[row * ldb + rhs] = value / diag;
        }
    }
}

template <SolverScalar T>
inline void solve_linear_system_inplace(T* a, T* b, std::size_t n, std::size_t nrhs) {
    std::vector<int> ipiv(n);
    lu_factorize_inplace(a, n, n, ipiv.data());
    lu_solve_inplace(a, n, n, ipiv.data(), b, nrhs, nrhs);
}

template <SolverScalar T>
inline void solve_linear_system_inplace(T* a, T* b, std::size_t n, std::size_t nrhs, int* ipiv) {
    lu_factorize_inplace(a, n, n, ipiv);
    lu_solve_inplace(a, n, n, ipiv, b, nrhs, nrhs);
}

template <VectorExpression LHS, VectorExpression RHS>
    requires std::same_as<typename bare_t<LHS>::value_type, typename bare_t<RHS>::value_type>
class VectorSum {
public:
    using value_type = typename bare_t<LHS>::value_type;

    VectorSum(const LHS& lhs, const RHS& rhs)
        : _lhs(lhs), _rhs(rhs) {
        assert(_lhs.size() == _rhs.size());
    }

    [[nodiscard]] std::size_t size() const noexcept { return _lhs.size(); }
    [[nodiscard]] value_type  eval(std::size_t index) const { return _lhs.eval(index) + _rhs.eval(index); }

private:
    vector_expr_storage_t<LHS> _lhs;
    vector_expr_storage_t<RHS> _rhs;
};

template <VectorExpression LHS, VectorExpression RHS>
    requires std::same_as<typename bare_t<LHS>::value_type, typename bare_t<RHS>::value_type>
class VectorDifference {
public:
    using value_type = typename bare_t<LHS>::value_type;

    VectorDifference(const LHS& lhs, const RHS& rhs)
        : _lhs(lhs), _rhs(rhs) {
        assert(_lhs.size() == _rhs.size());
    }

    [[nodiscard]] std::size_t size() const noexcept { return _lhs.size(); }
    [[nodiscard]] value_type  eval(std::size_t index) const { return _lhs.eval(index) - _rhs.eval(index); }

private:
    vector_expr_storage_t<LHS> _lhs;
    vector_expr_storage_t<RHS> _rhs;
};

template <VectorExpression Expr>
class VectorScale {
public:
    using value_type = typename bare_t<Expr>::value_type;

    VectorScale(const Expr& expr, value_type scalar)
        : _expr(expr), _scalar(scalar) {}

    [[nodiscard]] std::size_t size() const noexcept { return _expr.size(); }
    [[nodiscard]] value_type  eval(std::size_t index) const { return _expr.eval(index) * _scalar; }

private:
    vector_expr_storage_t<Expr> _expr;
    value_type                  _scalar;
};

template <VectorExpression Expr>
class VectorDivide {
public:
    using value_type = typename bare_t<Expr>::value_type;

    VectorDivide(const Expr& expr, value_type scalar)
        : _expr(expr), _scalar(scalar) {
        if (scalar == value_type{}) {
            throw std::domain_error("milk: division by zero");
        }
    }

    [[nodiscard]] std::size_t size() const noexcept { return _expr.size(); }
    [[nodiscard]] value_type  eval(std::size_t index) const { return _expr.eval(index) / _scalar; }

private:
    vector_expr_storage_t<Expr> _expr;
    value_type                  _scalar;
};

template <MatrixExpression LHS, MatrixExpression RHS>
    requires std::same_as<typename bare_t<LHS>::value_type, typename bare_t<RHS>::value_type>
class MatrixSum {
public:
    using value_type = typename bare_t<LHS>::value_type;

    MatrixSum(const LHS& lhs, const RHS& rhs)
        : _lhs(lhs), _rhs(rhs) {
        assert(_lhs.rows() == _rhs.rows() && _lhs.cols() == _rhs.cols());
    }

    [[nodiscard]] std::size_t rows() const noexcept { return _lhs.rows(); }
    [[nodiscard]] std::size_t cols() const noexcept { return _lhs.cols(); }
    [[nodiscard]] value_type  eval(std::size_t row, std::size_t col) const { return _lhs.eval(row, col) + _rhs.eval(row, col); }

private:
    matrix_expr_storage_t<LHS> _lhs;
    matrix_expr_storage_t<RHS> _rhs;
};

template <MatrixExpression LHS, MatrixExpression RHS>
    requires std::same_as<typename bare_t<LHS>::value_type, typename bare_t<RHS>::value_type>
class MatrixDifference {
public:
    using value_type = typename bare_t<LHS>::value_type;

    MatrixDifference(const LHS& lhs, const RHS& rhs)
        : _lhs(lhs), _rhs(rhs) {
        assert(_lhs.rows() == _rhs.rows() && _lhs.cols() == _rhs.cols());
    }

    [[nodiscard]] std::size_t rows() const noexcept { return _lhs.rows(); }
    [[nodiscard]] std::size_t cols() const noexcept { return _lhs.cols(); }
    [[nodiscard]] value_type  eval(std::size_t row, std::size_t col) const { return _lhs.eval(row, col) - _rhs.eval(row, col); }

private:
    matrix_expr_storage_t<LHS> _lhs;
    matrix_expr_storage_t<RHS> _rhs;
};

template <MatrixExpression Expr>
class MatrixScale {
public:
    using value_type = typename bare_t<Expr>::value_type;

    MatrixScale(const Expr& expr, value_type scalar)
        : _expr(expr), _scalar(scalar) {}

    [[nodiscard]] std::size_t rows() const noexcept { return _expr.rows(); }
    [[nodiscard]] std::size_t cols() const noexcept { return _expr.cols(); }
    [[nodiscard]] value_type  eval(std::size_t row, std::size_t col) const { return _expr.eval(row, col) * _scalar; }

private:
    matrix_expr_storage_t<Expr> _expr;
    value_type                  _scalar;
};

template <MatrixExpression Expr>
class MatrixDivide {
public:
    using value_type = typename bare_t<Expr>::value_type;

    MatrixDivide(const Expr& expr, value_type scalar)
        : _expr(expr), _scalar(scalar) {
        if (scalar == value_type{}) {
            throw std::domain_error("milk: division by zero");
        }
    }

    [[nodiscard]] std::size_t rows() const noexcept { return _expr.rows(); }
    [[nodiscard]] std::size_t cols() const noexcept { return _expr.cols(); }
    [[nodiscard]] value_type  eval(std::size_t row, std::size_t col) const { return _expr.eval(row, col) / _scalar; }

private:
    matrix_expr_storage_t<Expr> _expr;
    value_type                  _scalar;
};

} // namespace detail

/// @brief Contiguous aligned one-dimensional numeric container.
/// @tparam T Scalar value type.
template <Scalar T>
class Vector {
public:
    using value_type = T;

    /// @brief Construct an empty vector.
    Vector() = default;

    /// @brief Construct a zero-filled vector.
    /// @param size Number of entries.
    explicit Vector(std::size_t size)
        : _data(detail::allocate<T>(size)), _size(size) {
        detail::fill(_data, _size, T{});
    }

    /// @brief Construct a vector filled with a value.
    /// @param size Number of entries.
    /// @param value Initial value for each entry.
    Vector(std::size_t size, const T& value)
        : Vector(size) {
        fill(value);
    }

    /// @brief Construct a vector from initializer values.
    /// @param values Entries copied into the vector.
    Vector(std::initializer_list<T> values)
        : Vector(values.size()) {
        std::copy(values.begin(), values.end(), _data);
    }

    template <detail::VectorExpression Expr>
        requires std::same_as<typename detail::bare_t<Expr>::value_type, T> &&
                 (!std::same_as<typename detail::bare_t<Expr>, Vector>)
    /// @brief Construct a vector by evaluating an expression.
    /// @param expr Vector expression to evaluate.
    Vector(const Expr& expr)
        : Vector(expr.size()) {
        assignExpression(expr);
    }

    ~Vector() {
        detail::deallocate(_data);
    }

    Vector(const Vector& other)
        : Vector(other._size) {
        detail::copy(other._data, _data, _size);
    }

    Vector(Vector&& other) noexcept
        : _data(std::exchange(other._data, nullptr)),
          _size(std::exchange(other._size, 0)) {}

    Vector& operator=(const Vector& other) {
        if (this == &other) {
            return *this;
        }
        resizeForOverwrite(other._size);
        detail::copy(other._data, _data, _size);
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        detail::deallocate(_data);
        _data = std::exchange(other._data, nullptr);
        _size = std::exchange(other._size, 0);
        return *this;
    }

    template <detail::VectorExpression Expr>
        requires std::same_as<typename detail::bare_t<Expr>::value_type, T> &&
                 (!std::same_as<typename detail::bare_t<Expr>, Vector>)
    Vector& operator=(const Expr& expr) {
        resizeForOverwrite(expr.size());
        assignExpression(expr);
        return *this;
    }

    /// @brief Release storage and reset size to zero.
    void clear() {
        detail::deallocate(_data);
        _data = nullptr;
        _size = 0;
    }

    /// @brief Resize while preserving the overlapping prefix.
    /// @param size New number of entries.
    void resize(std::size_t size) {
        if (size == _size) {
            return;
        }
        Vector            replacement(size);
        const std::size_t copied = std::min(_size, size);
        detail::copy(_data, replacement._data, copied);
        swap(replacement);
    }

    /// @brief Resize and fill all entries.
    /// @param size New number of entries.
    /// @param value Value assigned to each entry.
    void assign(std::size_t size, const T& value) {
        resizeForOverwrite(size);
        fill(value);
    }

    /// @brief Fill all entries with a value.
    /// @param value Value assigned to each entry.
    void fill(const T& value) {
        detail::fill(_data, _size, value);
    }

    /// @brief Set all entries to zero.
    void setZero() {
        fill(T{});
    }

    /// @brief Return the number of entries.
    [[nodiscard]] std::size_t size() const noexcept { return _size; }
    /// @brief Return the vector length as a row count.
    [[nodiscard]] std::size_t rows() const noexcept { return _size; }
    /// @brief Return one column for vector-as-column semantics.
    [[nodiscard]] std::size_t cols() const noexcept { return 1; }
    /// @brief Return true when the vector has no entries.
    [[nodiscard]] bool empty() const noexcept { return _size == 0; }

    /// @brief Return a value for expression evaluation.
    /// @param index Entry index.
    [[nodiscard]] T eval(std::size_t index) const {
        assert(index < _size);
        return _data[index];
    }

    /// @brief Return mutable contiguous storage.
    T* data() noexcept { return _data; }
    /// @brief Return const contiguous storage.
    const T* data() const noexcept { return _data; }

    T& operator[](std::size_t index) noexcept {
        assert(index < _size);
        return _data[index];
    }

    const T& operator[](std::size_t index) const noexcept {
        assert(index < _size);
        return _data[index];
    }

    T& operator()(std::size_t index) noexcept {
        assert(index < _size);
        return _data[index];
    }

    const T& operator()(std::size_t index) const noexcept {
        assert(index < _size);
        return _data[index];
    }

    T*       begin() noexcept { return _data; }
    T*       end() noexcept { return _data + _size; }
    const T* begin() const noexcept { return _data; }
    const T* end() const noexcept { return _data + _size; }

    /// @brief Copy entries into a std::vector.
    std::vector<T> toVector() const {
        return std::vector<T>(_data, _data + _size);
    }

    /// @brief Replace entries from a std::vector.
    /// @param vec Source values.
    void fromVector(const std::vector<T>& vec) {
        resizeForOverwrite(vec.size());
        detail::copy(vec.data(), _data, vec.size());
    }

    /// @brief Return the sum of all entries.
    T sum() const {
        T result = T{};
        for (std::size_t index = 0; index < _size; ++index) {
            result += _data[index];
        }
        return result;
    }

    /// @brief Return the dot product with another vector.
    /// @param rhs Right-hand vector.
    T dot(const Vector& rhs) const {
        assert(_size == rhs._size);
        return detail::dot(_data, 1, rhs._data, 1, _size);
    }

    template <detail::VectorExpression Expr>
        requires std::same_as<typename detail::bare_t<Expr>::value_type, T>
    /// @brief Add a vector expression in place.
    /// @param rhs Expression added entrywise.
    Vector& operator+=(const Expr& rhs) {
        assert(_size == rhs.size());
        for (std::size_t index = 0; index < _size; ++index) {
            _data[index] += rhs.eval(index);
        }
        return *this;
    }

    template <detail::VectorExpression Expr>
        requires std::same_as<typename detail::bare_t<Expr>::value_type, T>
    /// @brief Subtract a vector expression in place.
    /// @param rhs Expression subtracted entrywise.
    Vector& operator-=(const Expr& rhs) {
        assert(_size == rhs.size());
        for (std::size_t index = 0; index < _size; ++index) {
            _data[index] -= rhs.eval(index);
        }
        return *this;
    }

    /// @brief Scale all entries in place.
    /// @param scalar Multiplicative factor.
    Vector& operator*=(const T& scalar) {
        for (std::size_t index = 0; index < _size; ++index) {
            _data[index] *= scalar;
        }
        return *this;
    }

    /// @brief Divide all entries in place.
    /// @param scalar Divisor.
    Vector& operator/=(const T& scalar) {
        if (scalar == T{}) {
            throw std::domain_error("milk: division by zero");
        }
        for (std::size_t index = 0; index < _size; ++index) {
            _data[index] /= scalar;
        }
        return *this;
    }

    /// @brief Swap storage with another vector.
    /// @param other Vector to swap with.
    void swap(Vector& other) noexcept {
        std::swap(_data, other._data);
        std::swap(_size, other._size);
    }

private:
    void resizeForOverwrite(std::size_t size) {
        if (size == _size) {
            return;
        }

        T* replacement = detail::allocate<T>(size);
        detail::deallocate(_data);
        _data = replacement;
        _size = size;
    }

    template <detail::VectorExpression Expr>
    void assignExpression(const Expr& expr) {
        for (std::size_t index = 0; index < _size; ++index) {
            _data[index] = expr.eval(index);
        }
    }

    T*          _data = nullptr;
    std::size_t _size = 0;
};

/// @brief Row-major contiguous aligned two-dimensional numeric container.
/// @tparam T Scalar value type.
template <Scalar T>
class Matrix {
public:
    using value_type = T;

    /// @brief Construct an empty matrix.
    Matrix() = default;

    /// @brief Construct a zero-filled row-major matrix.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    Matrix(std::size_t rows, std::size_t cols)
        : _data(detail::allocate<T>(rows * cols)), _rows(rows), _cols(cols) {
        detail::fill(_data, size(), T{});
    }

    /// @brief Construct a matrix filled with a value.
    /// @param rows Number of rows.
    /// @param cols Number of columns.
    /// @param value Initial value for each entry.
    Matrix(std::size_t rows, std::size_t cols, const T& value)
        : Matrix(rows, cols) {
        fill(value);
    }

    /// @brief Construct a matrix from nested initializer rows.
    /// @param values Row lists with equal length.
    Matrix(std::initializer_list<std::initializer_list<T>> values)
        : Matrix(values.size(), values.size() == 0 ? 0 : values.begin()->size()) {
        std::size_t row = 0;
        for (const auto& row_values : values) {
            if (row_values.size() != _cols) {
                throw std::invalid_argument("milk: jagged matrix initializer");
            }
            std::copy(row_values.begin(), row_values.end(), _data + row * _cols);
            ++row;
        }
    }

    template <detail::MatrixExpression Expr>
        requires std::same_as<typename detail::bare_t<Expr>::value_type, T> &&
                 (!std::same_as<typename detail::bare_t<Expr>, Matrix>)
    /// @brief Construct a matrix by evaluating an expression.
    /// @param expr Matrix expression to evaluate.
    Matrix(const Expr& expr)
        : Matrix(expr.rows(), expr.cols()) {
        assignExpression(expr);
    }

    ~Matrix() {
        detail::deallocate(_data);
    }

    Matrix(const Matrix& other)
        : Matrix(other._rows, other._cols) {
        detail::copy(other._data, _data, size());
    }

    Matrix(Matrix&& other) noexcept
        : _data(std::exchange(other._data, nullptr)),
          _rows(std::exchange(other._rows, 0)),
          _cols(std::exchange(other._cols, 0)) {}

    Matrix& operator=(const Matrix& other) {
        if (this == &other) {
            return *this;
        }
        resizeForOverwrite(other._rows, other._cols);
        detail::copy(other._data, _data, size());
        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        detail::deallocate(_data);
        _data = std::exchange(other._data, nullptr);
        _rows = std::exchange(other._rows, 0);
        _cols = std::exchange(other._cols, 0);
        return *this;
    }

    template <detail::MatrixExpression Expr>
        requires std::same_as<typename detail::bare_t<Expr>::value_type, T> &&
                 (!std::same_as<typename detail::bare_t<Expr>, Matrix>)
    Matrix& operator=(const Expr& expr) {
        resizeForOverwrite(expr.rows(), expr.cols());
        assignExpression(expr);
        return *this;
    }

    /// @brief Release storage and reset shape to zero by zero.
    void clear() {
        detail::deallocate(_data);
        _data = nullptr;
        _rows = 0;
        _cols = 0;
    }

    /// @brief Resize while preserving the overlapping upper-left block.
    /// @param rows New row count.
    /// @param cols New column count.
    void resize(std::size_t rows, std::size_t cols) {
        if (rows == _rows && cols == _cols) {
            return;
        }
        Matrix            replacement(rows, cols);
        const std::size_t copied_rows = std::min(_rows, rows);
        const std::size_t copied_cols = std::min(_cols, cols);
        for (std::size_t row = 0; row < copied_rows; ++row) {
            detail::copy(_data + row * _cols, replacement._data + row * cols, copied_cols);
        }
        swap(replacement);
    }

    /// @brief Resize and fill all entries.
    /// @param rows New row count.
    /// @param cols New column count.
    /// @param value Value assigned to each entry.
    void assign(std::size_t rows, std::size_t cols, const T& value) {
        resizeForOverwrite(rows, cols);
        fill(value);
    }

    /// @brief Fill all matrix entries with a value.
    /// @param value Value assigned to each entry.
    void fill(const T& value) {
        detail::fill(_data, size(), value);
    }

    /// @brief Set all matrix entries to zero.
    void setZero() {
        fill(T{});
    }

    /// @brief Return the row count.
    [[nodiscard]] std::size_t rows() const noexcept { return _rows; }
    /// @brief Return the column count.
    [[nodiscard]] std::size_t cols() const noexcept { return _cols; }
    /// @brief Return rows multiplied by columns.
    [[nodiscard]] std::size_t size() const noexcept { return _rows * _cols; }
    /// @brief Return true when the matrix has no entries.
    [[nodiscard]] bool empty() const noexcept { return size() == 0; }

    /// @brief Return a value for expression evaluation.
    /// @param row Row index.
    /// @param col Column index.
    [[nodiscard]] T eval(std::size_t row, std::size_t col) const {
        assert(row < _rows && col < _cols);
        return _data[row * _cols + col];
    }

    /// @brief Return mutable contiguous row-major storage.
    T* data() noexcept { return _data; }
    /// @brief Return const contiguous row-major storage.
    const T* data() const noexcept { return _data; }

    T& operator[](std::size_t index) noexcept {
        assert(index < size());
        return _data[index];
    }

    const T& operator[](std::size_t index) const noexcept {
        assert(index < size());
        return _data[index];
    }

    T& operator()(std::size_t index) noexcept {
        assert(index < size());
        return _data[index];
    }

    const T& operator()(std::size_t index) const noexcept {
        assert(index < size());
        return _data[index];
    }

    T& operator()(std::size_t row, std::size_t col) noexcept {
        assert(row < _rows && col < _cols);
        return _data[row * _cols + col];
    }

    const T& operator()(std::size_t row, std::size_t col) const noexcept {
        assert(row < _rows && col < _cols);
        return _data[row * _cols + col];
    }

    /// @brief Return a pointer to the first entry in a row.
    /// @param row Row index.
    T* rowData(std::size_t row) noexcept {
        assert(row < _rows);
        return _data + row * _cols;
    }

    /// @brief Return a const pointer to the first entry in a row.
    /// @param row Row index.
    const T* rowData(std::size_t row) const noexcept {
        assert(row < _rows);
        return _data + row * _cols;
    }

    /// @brief Return the transpose of this matrix.
    Matrix transpose() const {
        Matrix result(_cols, _rows);
        for (std::size_t row = 0; row < _rows; ++row) {
            for (std::size_t col = 0; col < _cols; ++col) {
                result(col, row) = (*this)(row, col);
            }
        }
        return result;
    }

    /// @brief Return an identity matrix.
    /// @param size Number of rows and columns.
    static Matrix identity(std::size_t size) {
        Matrix result(size, size);
        for (std::size_t index = 0; index < size; ++index) {
            result(index, index) = T{1};
        }
        return result;
    }

    /// @brief Return the infinity norm.
    double infiniteNorm() const {
        double max_sum = 0.0;
        for (std::size_t row = 0; row < _rows; ++row) {
            double row_sum = 0.0;
            for (std::size_t col = 0; col < _cols; ++col) {
                row_sum += std::abs(static_cast<double>((*this)(row, col)));
            }
            max_sum = std::max(max_sum, row_sum);
        }
        return max_sum;
    }

    /// @brief Moore-Penrose pseudo-inverse via one-sided Jacobi SVD.
    /// Robust to rank-deficient / collinear / badly-scaled columns: one-sided Jacobi computes the
    /// SVD to high relative accuracy even when column norms span many orders (so no per-column
    /// pre-scaling is needed), and singular values below `rcond * sigma_max` are treated as zero,
    /// giving the minimum-norm least-squares solution. For an (m x n) matrix A returns A^+ (n x m);
    /// the least-squares solution of A x = b is x = A^+ b.
    /// @param rcond Relative singular-value cutoff for numerical rank (default 1e-12).
    /// @param maxRank when >= 0, keep only the `maxRank` largest singular
    /// values regardless of `rcond`. A fixed rank holds the model complexity
    /// constant across fits; a relative threshold lets it vary with the
    /// spectrum, which shows up as terms switching on and off between keys.
    [[nodiscard]] Matrix pseudoInverse(double rcond = 1.0e-12, int maxRank = -1) const
        requires std::floating_point<T> {
        // Orthogonalize columns, so work on the tall orientation (m >= n) and transpose the
        // result for wide inputs: A^+ = (A^T)^{+T}.
        const bool wide = _rows < _cols;
        Matrix      a   = wide ? transpose() : *this; // m x n, m >= n
        const std::size_t m = a.rows(), n = a.cols();
        Matrix v = identity(n);
        if (m == 0 || n == 0)
            return Matrix(_cols, _rows);

        // One-sided Jacobi sweeps: right-rotate column pairs (p, q) to mutual orthogonality.
        for (int sweep = 0; sweep < 60; ++sweep) {
            double off = 0.0;
            for (std::size_t p = 0; p < n; ++p) {
                for (std::size_t q = p + 1; q < n; ++q) {
                    double app = 0.0, aqq = 0.0, apq = 0.0;
                    for (std::size_t i = 0; i < m; ++i) {
                        const double aip = a(i, p), aiq = a(i, q);
                        app += aip * aip;
                        aqq += aiq * aiq;
                        apq += aip * aiq;
                    }
                    if (app * aqq > 0.0)
                        off = std::max(off, std::abs(apq) / std::sqrt(app * aqq));
                    if (std::abs(apq) <= 1.0e-300)
                        continue;
                    const double tau = (aqq - app) / (2.0 * apq);
                    const double t   = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                    const double cs  = 1.0 / std::sqrt(1.0 + t * t);
                    const double sn  = cs * t;
                    for (std::size_t i = 0; i < m; ++i) {
                        const double aip = a(i, p), aiq = a(i, q);
                        a(i, p) = static_cast<T>(cs * aip - sn * aiq);
                        a(i, q) = static_cast<T>(sn * aip + cs * aiq);
                    }
                    for (std::size_t i = 0; i < n; ++i) {
                        const double vip = v(i, p), viq = v(i, q);
                        v(i, p) = static_cast<T>(cs * vip - sn * viq);
                        v(i, q) = static_cast<T>(sn * vip + cs * viq);
                    }
                }
            }
            if (off < 1.0e-15)
                break;
        }

        // Column norms of the rotated matrix are the singular values; its columns are sigma_j * U_j.
        std::vector<double> sigma(n, 0.0);
        double              smax = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            double s2 = 0.0;
            for (std::size_t i = 0; i < m; ++i)
                s2 += static_cast<double>(a(i, j)) * static_cast<double>(a(i, j));
            sigma[j] = std::sqrt(s2);
            smax     = std::max(smax, sigma[j]);
        }
        double cutoff = rcond * smax;
        if (maxRank >= 0) {
            std::vector<double> sorted(sigma);
            std::sort(sorted.begin(), sorted.end(), std::greater<double>());
            const std::size_t keep = static_cast<std::size_t>(maxRank);
            cutoff = keep == 0                 ? std::numeric_limits<double>::infinity()
                     : keep >= sorted.size()   ? -1.0
                                               : sorted[keep];
        }

        // A^+ = V Sigma^+ U^T, with U_j = a_col_j / sigma_j, dropping sigma_j <= cutoff:
        //   A^+(k,i) = sum_j  V(k,j) * a(i,j) / sigma_j^2   over kept j.
        Matrix pinv(n, m);
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t i = 0; i < m; ++i) {
                double acc = 0.0;
                for (std::size_t j = 0; j < n; ++j)
                    if (sigma[j] > cutoff && std::isfinite(cutoff))
                        acc += static_cast<double>(v(k, j)) * static_cast<double>(a(i, j)) / (sigma[j] * sigma[j]);
                pinv(k, i) = static_cast<T>(acc);
            }
        }
        return wide ? pinv.transpose() : pinv;
    }

    /// @brief Return the inverse of a square matrix.
    [[nodiscard]] Matrix inverse() const
        requires std::floating_point<T>;
    /// @brief Solve this square matrix against a vector right-hand side.
    /// @param rhs Right-hand side vector.
    [[nodiscard]] Vector<T> solve(const Vector<T>& rhs) const
        requires std::floating_point<T>;

    template <detail::MatrixExpression Expr>
        requires std::same_as<typename detail::bare_t<Expr>::value_type, T>
    /// @brief Add a matrix expression in place.
    /// @param rhs Expression added entrywise.
    Matrix& operator+=(const Expr& rhs) {
        assert(_rows == rhs.rows() && _cols == rhs.cols());
        for (std::size_t row = 0; row < _rows; ++row) {
            for (std::size_t col = 0; col < _cols; ++col) {
                (*this)(row, col) += rhs.eval(row, col);
            }
        }
        return *this;
    }

    template <detail::MatrixExpression Expr>
        requires std::same_as<typename detail::bare_t<Expr>::value_type, T>
    /// @brief Subtract a matrix expression in place.
    /// @param rhs Expression subtracted entrywise.
    Matrix& operator-=(const Expr& rhs) {
        assert(_rows == rhs.rows() && _cols == rhs.cols());
        for (std::size_t row = 0; row < _rows; ++row) {
            for (std::size_t col = 0; col < _cols; ++col) {
                (*this)(row, col) -= rhs.eval(row, col);
            }
        }
        return *this;
    }

    /// @brief Scale all entries in place.
    /// @param scalar Multiplicative factor.
    Matrix& operator*=(const T& scalar) {
        for (std::size_t index = 0; index < size(); ++index) {
            _data[index] *= scalar;
        }
        return *this;
    }

    /// @brief Divide all entries in place.
    /// @param scalar Divisor.
    Matrix& operator/=(const T& scalar) {
        if (scalar == T{}) {
            throw std::domain_error("milk: division by zero");
        }
        for (std::size_t index = 0; index < size(); ++index) {
            _data[index] /= scalar;
        }
        return *this;
    }

    /// @brief Swap storage and shape with another matrix.
    /// @param other Matrix to swap with.
    void swap(Matrix& other) noexcept {
        std::swap(_data, other._data);
        std::swap(_rows, other._rows);
        std::swap(_cols, other._cols);
    }

private:
    void resizeForOverwrite(std::size_t rows, std::size_t cols) {
        if (rows == _rows && cols == _cols) {
            return;
        }

        T* replacement = detail::allocate<T>(rows * cols);
        detail::deallocate(_data);
        _data = replacement;
        _rows = rows;
        _cols = cols;
    }

    template <detail::MatrixExpression Expr>
    void assignExpression(const Expr& expr) {
        for (std::size_t row = 0; row < _rows; ++row) {
            for (std::size_t col = 0; col < _cols; ++col) {
                (*this)(row, col) = expr.eval(row, col);
            }
        }
    }

    T*          _data = nullptr;
    std::size_t _rows = 0;
    std::size_t _cols = 0;
};

namespace detail {

inline std::string trim_csv_cell(const std::string& value) {
    const std::size_t begin = value.find_first_not_of(" \t\n\r\f\v");
    if (begin == std::string::npos) return "";

    const std::size_t end = value.find_last_not_of(" \t\n\r\f\v");
    return value.substr(begin, end - begin + 1);
}

inline std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::size_t              begin = 0;

    while (begin <= line.size()) {
        const std::size_t comma = line.find(',', begin);
        if (comma == std::string::npos) {
            fields.push_back(trim_csv_cell(line.substr(begin)));
            break;
        }

        fields.push_back(trim_csv_cell(line.substr(begin, comma - begin)));
        begin = comma + 1;
    }

    return fields;
}

} // namespace detail

/// @brief Load a labeled CSV matrix after checking row and column labels.
/// @param path CSV path with a blank corner cell, column labels, and row labels.
/// @param row_labels Expected row labels in file order.
/// @param col_labels Expected column labels in file order.
template <Scalar T = double>
Matrix<T> LabeledMatrixFromCSV(const std::filesystem::path&    path,
                               const std::vector<std::string>& row_labels,
                               const std::vector<std::string>& col_labels) {
    const std::size_t nrow = row_labels.size();
    const std::size_t ncol = col_labels.size();
    Matrix<T>         matrix(nrow, ncol);
    std::ifstream     file(path);
    if (!file.is_open())
        throw std::runtime_error("milk: failed to open CSV: " + path.string());

    std::string line;
    if (!std::getline(file, line))
        throw std::runtime_error("milk: empty labeled CSV matrix: " + path.string());

    const auto header = detail::split_csv_line(line);
    if (header.size() != ncol + 1)
        throw std::runtime_error("milk: invalid labeled CSV header size: " + path.string());

    for (std::size_t col = 0; col < ncol; ++col) {
        if (header[col + 1] != col_labels[col])
            throw std::runtime_error("milk: unexpected labeled CSV column label: " + path.string());
    }

    for (std::size_t row = 0; row < nrow; ++row) {
        if (!std::getline(file, line))
            throw std::runtime_error("milk: insufficient labeled CSV rows: " + path.string());

        const auto cells = detail::split_csv_line(line);
        if (cells.size() != ncol + 1)
            throw std::runtime_error("milk: invalid labeled CSV row size: " + path.string());
        if (cells[0] != row_labels[row])
            throw std::runtime_error("milk: unexpected labeled CSV row label: " + path.string());

        for (std::size_t col = 0; col < ncol; ++col) {
            std::size_t parsed = 0;
            matrix(row, col)   = static_cast<T>(std::stod(cells[col + 1], &parsed));
            if (parsed != cells[col + 1].size())
                throw std::runtime_error("milk: invalid numeric value in labeled CSV: " + path.string());
        }
    }

    return matrix;
}

/// @brief Two-dimensional table with bilinear interpolation.
class Table {
public:
    /// @brief X-axis grid values.
    Vector<double> x_axis;
    /// @brief Y-axis grid values.
    Vector<double> y_axis;
    /// @brief Table values stored as rows over y and columns over x.
    Matrix<double> values;

    /// @brief Parse a table CSV whose first row is x and first column is y.
    /// @param path CSV file path.
    static Table ParseFromCSV(const std::filesystem::path& path) {
        Table         table;
        std::ifstream file(path);
        if (!file.is_open())
            throw std::runtime_error("milk: failed to open table CSV: " + path.string());

        std::string line;
        if (!std::getline(file, line))
            throw std::runtime_error("milk: empty table CSV: " + path.string());

        const auto header = detail::split_csv_line(line);
        if (header.size() < 2)
            throw std::runtime_error("milk: invalid table CSV header: " + path.string());

        table.x_axis.assign(header.size() - 1, 0.0);
        for (std::size_t col = 1; col < header.size(); ++col)
            table.x_axis[col - 1] = std::stod(header[col]);

        std::vector<double> y_values;
        std::vector<double> flat;
        while (std::getline(file, line)) {
            if (line.empty()) continue;

            const auto row = detail::split_csv_line(line);
            if (row.size() != header.size())
                throw std::runtime_error("milk: invalid table CSV row size: " + path.string());

            y_values.push_back(std::stod(row[0]));
            for (std::size_t col = 1; col < row.size(); ++col)
                flat.push_back(std::stod(row[col]));
        }

        table.y_axis.assign(y_values.size(), 0.0);
        table.values.assign(y_values.size(), table.x_axis.size(), 0.0);
        for (std::size_t row = 0; row < y_values.size(); ++row) {
            table.y_axis[row] = y_values[row];
            for (std::size_t col = 0; col < table.x_axis.size(); ++col)
                table.values(row, col) = flat[row * table.x_axis.size() + col];
        }
        return table;
    }

    /// @brief Bilinearly interpolate a value, clamping outside the tabulated range.
    /// @param x X-axis query value.
    /// @param y Y-axis query value.
    [[nodiscard]] double Get(double x, double y) const {
        const std::size_t nx = x_axis.size();
        const std::size_t ny = y_axis.size();
        if (nx < 2 || ny < 2)
            throw std::runtime_error("milk: table interpolation needs at least 2x2 points");

        x = std::clamp(x, x_axis[0], x_axis[nx - 1]);
        y = std::clamp(y, y_axis[0], y_axis[ny - 1]);

        std::size_t ix = FindLowerIndex(x_axis, x);
        std::size_t iy = FindLowerIndex(y_axis, y);
        if (ix >= nx - 1) ix = nx - 2;
        if (iy >= ny - 1) iy = ny - 2;

        const double x0 = x_axis[ix], x1 = x_axis[ix + 1];
        const double y0 = y_axis[iy], y1 = y_axis[iy + 1];
        const double fx = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0;
        const double fy = (y1 > y0) ? (y - y0) / (y1 - y0) : 0.0;

        const double z00 = values(iy, ix);
        const double z01 = values(iy, ix + 1);
        const double z10 = values(iy + 1, ix);
        const double z11 = values(iy + 1, ix + 1);

        const double z0 = z00 + fx * (z01 - z00);
        const double z1 = z10 + fx * (z11 - z10);
        return z0 + fy * (z1 - z0);
    }

private:
    static std::size_t FindLowerIndex(const Vector<double>& axis, double value) {
        std::size_t lo = 0;
        std::size_t hi = axis.size() - 1;
        while (lo + 1 < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (axis[mid] <= value)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }
};

template <detail::VectorExpression LHS, detail::VectorExpression RHS>
    requires std::same_as<typename detail::bare_t<LHS>::value_type, typename detail::bare_t<RHS>::value_type>
/// @brief Return a lazy vector sum expression.
/// @param lhs Left vector expression.
/// @param rhs Right vector expression.
inline auto operator+(const LHS& lhs, const RHS& rhs) {
    return detail::VectorSum<LHS, RHS>(lhs, rhs);
}

template <detail::VectorExpression LHS, detail::VectorExpression RHS>
    requires std::same_as<typename detail::bare_t<LHS>::value_type, typename detail::bare_t<RHS>::value_type>
/// @brief Return a lazy vector difference expression.
/// @param lhs Left vector expression.
/// @param rhs Right vector expression.
inline auto operator-(const LHS& lhs, const RHS& rhs) {
    return detail::VectorDifference<LHS, RHS>(lhs, rhs);
}

template <detail::VectorExpression Expr>
/// @brief Return a lazy vector scale expression.
/// @param expr Vector expression.
/// @param scalar Multiplicative factor.
inline auto operator*(const Expr& expr, const typename detail::bare_t<Expr>::value_type& scalar) {
    return detail::VectorScale<Expr>(expr, scalar);
}

template <detail::VectorExpression Expr>
/// @brief Return a lazy vector scale expression.
/// @param scalar Multiplicative factor.
/// @param expr Vector expression.
inline auto operator*(const typename detail::bare_t<Expr>::value_type& scalar, const Expr& expr) {
    return detail::VectorScale<Expr>(expr, scalar);
}

template <detail::VectorExpression Expr>
/// @brief Return a lazy vector division expression.
/// @param expr Vector expression.
/// @param scalar Divisor.
inline auto operator/(const Expr& expr, const typename detail::bare_t<Expr>::value_type& scalar) {
    return detail::VectorDivide<Expr>(expr, scalar);
}

template <detail::MatrixExpression LHS, detail::MatrixExpression RHS>
    requires std::same_as<typename detail::bare_t<LHS>::value_type, typename detail::bare_t<RHS>::value_type>
/// @brief Return a lazy matrix sum expression.
/// @param lhs Left matrix expression.
/// @param rhs Right matrix expression.
inline auto operator+(const LHS& lhs, const RHS& rhs) {
    return detail::MatrixSum<LHS, RHS>(lhs, rhs);
}

template <detail::MatrixExpression LHS, detail::MatrixExpression RHS>
    requires std::same_as<typename detail::bare_t<LHS>::value_type, typename detail::bare_t<RHS>::value_type>
/// @brief Return a lazy matrix difference expression.
/// @param lhs Left matrix expression.
/// @param rhs Right matrix expression.
inline auto operator-(const LHS& lhs, const RHS& rhs) {
    return detail::MatrixDifference<LHS, RHS>(lhs, rhs);
}

template <detail::MatrixExpression Expr>
/// @brief Return a lazy matrix scale expression.
/// @param expr Matrix expression.
/// @param scalar Multiplicative factor.
inline auto operator*(const Expr& expr, const typename detail::bare_t<Expr>::value_type& scalar) {
    return detail::MatrixScale<Expr>(expr, scalar);
}

template <detail::MatrixExpression Expr>
/// @brief Return a lazy matrix scale expression.
/// @param scalar Multiplicative factor.
/// @param expr Matrix expression.
inline auto operator*(const typename detail::bare_t<Expr>::value_type& scalar, const Expr& expr) {
    return detail::MatrixScale<Expr>(expr, scalar);
}

template <detail::MatrixExpression Expr>
/// @brief Return a lazy matrix division expression.
/// @param expr Matrix expression.
/// @param scalar Divisor.
inline auto operator/(const Expr& expr, const typename detail::bare_t<Expr>::value_type& scalar) {
    return detail::MatrixDivide<Expr>(expr, scalar);
}

template <detail::VectorExpression LHS, detail::VectorExpression RHS>
    requires std::same_as<typename detail::bare_t<LHS>::value_type, typename detail::bare_t<RHS>::value_type>
/// @brief Compute a vector dot product.
/// @param lhs Left vector expression.
/// @param rhs Right vector expression.
inline auto dot(const LHS& lhs, const RHS& rhs) -> typename detail::bare_t<LHS>::value_type {
    assert(lhs.size() == rhs.size());
    using T = typename detail::bare_t<LHS>::value_type;

    if constexpr (detail::is_vector_leaf_v<LHS> && detail::is_vector_leaf_v<RHS>) {
        return detail::dot(lhs.data(), 1, rhs.data(), 1, lhs.size());
    } else {
        T sum = T{};
        for (std::size_t index = 0; index < lhs.size(); ++index) {
            sum += lhs.eval(index) * rhs.eval(index);
        }
        return sum;
    }
}

template <detail::VectorExpression LHS, detail::VectorExpression RHS>
    requires std::same_as<typename detail::bare_t<LHS>::value_type, typename detail::bare_t<RHS>::value_type>
/// @brief Compute a vector dot product with operator syntax.
/// @param lhs Left vector expression.
/// @param rhs Right vector expression.
inline auto operator*(const LHS& lhs, const RHS& rhs) -> typename detail::bare_t<LHS>::value_type {
    return dot(lhs, rhs);
}

template <detail::MatrixExpression M, detail::VectorExpression V>
    requires std::same_as<typename detail::bare_t<M>::value_type, typename detail::bare_t<V>::value_type>
/// @brief Multiply a matrix expression by a vector expression.
/// @param mat_expr Matrix expression.
/// @param vec_expr Vector expression.
inline auto operator*(const M& mat_expr, const V& vec_expr) -> Vector<typename detail::bare_t<M>::value_type> {
    using T = typename detail::bare_t<M>::value_type;
    Matrix<T> mat(mat_expr);
    Vector<T> vec(vec_expr);
    assert(mat.cols() == vec.size());

    Vector<T> result(mat.rows());
    for (std::size_t row = 0; row < mat.rows(); ++row) {
        result[row]      = T{};
        const T* row_ptr = mat.rowData(row);
        for (std::size_t col = 0; col < mat.cols(); ++col) {
            result[row] += row_ptr[col] * vec[col];
        }
    }
    return result;
}

template <detail::VectorExpression V, detail::MatrixExpression M>
    requires std::same_as<typename detail::bare_t<M>::value_type, typename detail::bare_t<V>::value_type>
/// @brief Multiply a vector expression by a matrix expression.
/// @param vec_expr Vector expression.
/// @param mat_expr Matrix expression.
inline auto operator*(const V& vec_expr, const M& mat_expr) -> Vector<typename detail::bare_t<M>::value_type> {
    using T = typename detail::bare_t<M>::value_type;
    Vector<T> vec(vec_expr);
    Matrix<T> mat(mat_expr);
    assert(vec.size() == mat.rows());

    Vector<T> result(mat.cols());
    result.fill(T{});
    for (std::size_t row = 0; row < mat.rows(); ++row) {
        const T  factor  = vec[row];
        const T* row_ptr = mat.rowData(row);
        for (std::size_t col = 0; col < mat.cols(); ++col) {
            result[col] += factor * row_ptr[col];
        }
    }
    return result;
}

template <detail::MatrixExpression LHS, detail::MatrixExpression RHS>
    requires std::same_as<typename detail::bare_t<LHS>::value_type, typename detail::bare_t<RHS>::value_type>
/// @brief Multiply two matrix expressions and return a materialized matrix.
/// @param lhs_expr Left matrix expression.
/// @param rhs_expr Right matrix expression.
inline auto operator*(const LHS& lhs_expr, const RHS& rhs_expr) -> Matrix<typename detail::bare_t<LHS>::value_type> {
    using T = typename detail::bare_t<LHS>::value_type;
    Matrix<T> lhs(lhs_expr);
    Matrix<T> rhs(rhs_expr);
    assert(lhs.cols() == rhs.rows());

    Matrix<T> result(lhs.rows(), rhs.cols());
    result.fill(T{});

    constexpr std::size_t tile = 32;
    for (std::size_t row_block = 0; row_block < lhs.rows(); row_block += tile) {
        const std::size_t row_end = std::min(row_block + tile, lhs.rows());
        for (std::size_t inner_block = 0; inner_block < lhs.cols(); inner_block += tile) {
            const std::size_t inner_end = std::min(inner_block + tile, lhs.cols());
            for (std::size_t col_block = 0; col_block < rhs.cols(); col_block += tile) {
                const std::size_t col_end = std::min(col_block + tile, rhs.cols());
                for (std::size_t row = row_block; row < row_end; ++row) {
                    T*       out_row = result.rowData(row);
                    const T* lhs_row = lhs.rowData(row);
                    for (std::size_t inner = inner_block; inner < inner_end; ++inner) {
                        const T  lhs_value = lhs_row[inner];
                        const T* rhs_row   = rhs.rowData(inner);
                        for (std::size_t col = col_block; col < col_end; ++col) {
                            out_row[col] += lhs_value * rhs_row[col];
                        }
                    }
                }
            }
        }
    }
    return result;
}

/// @brief Reusable scratch buffers for Gauss-Seidel CRAM Bateman solves.
struct CramWorkspace {
    using complex_t = std::complex<double>;

    /// @brief Resize all buffers for a depletion system.
    /// @param size Number of isotopes in the system.
    void resize(std::size_t size) {
        base_cols.resize(size);
        base_vals.resize(size);
        base_diag.resize(size);
        pole_diag.resize(size);
        rhs.resize(size);
        x.resize(size);
        accum.resize(size);
    }

    std::vector<std::vector<std::size_t>> base_cols; ///< off-diagonal column indices per row
    std::vector<std::vector<complex_t>>   base_vals; ///< off-diagonal values per row
    std::vector<complex_t>                base_diag; ///< diagonal entry per row (pole-independent)
    std::vector<complex_t>                pole_diag; ///< base_diag - theta[pole] per row, rebuilt per pole
    std::vector<complex_t>                rhs;
    std::vector<complex_t>                x;
    std::vector<complex_t>                accum;
};

/// @brief Static dense linear-system and Bateman CRAM routines.
/// @tparam T Floating-point value type.
template <std::floating_point T>
class Solver {
public:
    Solver()                         = delete;
    ~Solver()                        = delete;
    Solver(const Solver&)            = delete;
    Solver(Solver&&)                 = delete;
    Solver& operator=(const Solver&) = delete;
    Solver& operator=(Solver&&)      = delete;

    /// @brief Solve Ax=b by Gaussian elimination.
    /// @param A Square system matrix, copied and factorized.
    /// @param b Right-hand side vector, copied and overwritten.
    static Vector<T> solve(Matrix<T> A, Vector<T> b) {
        return solveGauss(std::move(A), std::move(b));
    }

    /// @brief Solve Ax=b by Gaussian elimination.
    /// @param A Square system matrix, copied and factorized.
    /// @param b Right-hand side vector, copied and overwritten.
    static Vector<T> solveGauss(Matrix<T> A, Vector<T> b) {
        if (A.rows() != A.cols() || A.rows() != b.size()) {
            throw std::invalid_argument("milk: Ax=b dimension mismatch");
        }
        solveGaussInPlace(A, b);
        return b;
    }

    /// @brief Solve Ax=b in place.
    /// @param A Square system matrix overwritten by LU factors.
    /// @param b Right-hand side vector overwritten by the solution.
    static void solveInPlace(Matrix<T>& A, Vector<T>& b) {
        solveGaussInPlace(A, b);
    }

    /// @brief Solve Ax=b in place by Gaussian elimination.
    /// @param A Square system matrix overwritten by LU factors.
    /// @param b Right-hand side vector overwritten by the solution.
    static void solveGaussInPlace(Matrix<T>& A, Vector<T>& b) {
        if (A.rows() != A.cols() || A.rows() != b.size()) {
            throw std::invalid_argument("milk: Ax=b dimension mismatch");
        }
        detail::solve_linear_system_inplace(A.data(), b.data(), A.rows(), 1);
    }

    /// @brief Solve AX=B in place by Gaussian elimination.
    /// @param A Square system matrix overwritten by LU factors.
    /// @param B Right-hand side matrix overwritten by the solution.
    static void solveGaussInPlace(Matrix<T>& A, Matrix<T>& B) {
        if (A.rows() != A.cols() || A.rows() != B.rows()) {
            throw std::invalid_argument("milk: AX=B dimension mismatch");
        }
        detail::solve_linear_system_inplace(A.data(), B.data(), A.rows(), B.cols());
    }

    /// @brief Return the inverse of a square matrix.
    /// @param A Square matrix copied and factorized.
    static Matrix<T> inverse(Matrix<T> A) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("milk: inverse requires a square matrix");
        }
        Matrix<T> inv = Matrix<T>::identity(A.rows());
        solveGaussInPlace(A, inv);
        return inv;
    }

    /// @brief Solve a Bateman depletion step with CRAM8 or CRAM16 using Gauss-Seidel pole solves.
    /// @param A Bateman transition matrix.
    /// @param N Initial number densities.
    /// @param dt Time step multiplier applied to A.
    /// @param out Output number densities.
    /// @param workspace Reusable complex work buffers.
    /// @param order CRAM order, either 8 or 16.
    /// @param first First isotope index to evolve; lower indices are copied from N.
    static void solveBatemanCRAM(const Matrix<T>& A, const Vector<T>& N, double dt, Vector<T>& out,
                                 CramWorkspace& workspace, int order = 16, std::size_t first = 0) {
        using complex_t = std::complex<double>;

        if (A.rows() != A.cols() || A.rows() != N.size()) {
            throw std::invalid_argument("milk: CRAM dimension mismatch");
        }

        const std::size_t n = A.rows();
        if (first > n) {
            throw std::invalid_argument("milk: CRAM invalid first isotope");
        }

        out.resize(n);
        if (n == 0) return;
        for (std::size_t row = 0; row < first; ++row)
            out[row] = N[row];
        if (first == n) return;

        double                   alpha0     = 0.0;
        int                      max_iter   = 0;
        double                   rel_tol    = 0.0;
        double                   matrix_sgn = 1.0;
        std::size_t              pole_count = 0;
        std::array<complex_t, 8> alpha{};
        std::array<complex_t, 8> theta{};

        if (order == 8) {
            alpha0     = 1.1722341374385704e-08;
            max_iter   = 64;
            rel_tol    = 1.0e-13;
            matrix_sgn = -1.0;
            pole_count = 4;
            alpha      = {
                complex_t{+1.83174069610856716e+00, -9.52542527224556679e+00},
                complex_t{-2.43619809577363400e+00, +3.71667983752542863e+00},
                complex_t{+6.32575834187860564e-01, -4.43912790240850230e-01},
                complex_t{-2.81291599910903876e-02, +1.15770931709880849e-02},
                complex_t{},
                complex_t{},
                complex_t{},
                complex_t{}
            };
            theta = {
                complex_t{-3.22092672186933981e+00, +1.19361884200025181e+00},
                complex_t{-2.29222964471934798e+00, +3.60076959131180274e+00},
                complex_t{-2.69470045809068803e-01, +6.08203046216700294e+00},
                complex_t{+3.40856168532368731e+00, +8.77303318542488775e+00},
                complex_t{},
                complex_t{},
                complex_t{},
                complex_t{}
            };
        } else if (order == 16) {
            alpha0     = 2.1248537104952237488e-16;
            max_iter   = 128;
            rel_tol    = 1.0e-14;
            matrix_sgn = 1.0;
            pole_count = 8;
            alpha      = {
                complex_t{-5.0901521865224915650e-7, -2.4220017652852287970e-5},
                complex_t{+2.1151742182466030907e-4, +4.3892969647380673918e-3},
                complex_t{ +1.1339775178483930527e2,  +1.0194721704215856450e2},
                complex_t{ +1.5059585270023467528e1,  -5.7514052776421819979e0},
                complex_t{ -6.4500878025539646595e1,  -2.2459440762652096056e2},
                complex_t{ -1.4793007113557999718e0,  +1.7686588323782937906e0},
                complex_t{ -6.2518392463207918892e1,  -1.1190391094283228480e1},
                complex_t{+4.1023136835410021273e-2, -1.5743466173455468191e-1}
            };
            theta = {
                complex_t{-1.0843917078696988026e1, +1.9277446167181652284e1},
                complex_t{-5.2649713434426468895e0, +1.6220221473167927305e1},
                complex_t{+5.9481522689511774808e0, +3.5874573620183222829e0},
                complex_t{+3.5091036084149180974e0, +8.4361989858843750826e0},
                complex_t{+6.4161776990994341923e0, +1.1941223933701386874e0},
                complex_t{+1.4193758971856659786e0, +1.0925363484496722585e1},
                complex_t{+4.9931747377179963991e0, +5.9968817136039422260e0},
                complex_t{-1.4139284624888862114e0, +1.3497725698892745389e1}
            };
        } else {
            throw std::invalid_argument("milk: CRAM order must be 8 or 16");
        }

        constexpr double abs_tol  = 1.0e-28;
        constexpr double diag_tol = 1.0e-30;

        workspace.resize(n);
        std::fill(workspace.accum.begin(), workspace.accum.end(), complex_t{0.0, 0.0});

        // Split the diagonal from the off-diagonal sparse row data once: the diagonal is
        // Gauss-Seidel-invariant per pole, so each sweep multiplies by a precomputed
        // reciprocal instead of rescanning the row and dividing (complex division is the
        // single hottest depletion libcall).
        for (std::size_t row = first; row < n; ++row) {
            auto& row_cols = workspace.base_cols[row];
            auto& row_vals = workspace.base_vals[row];
            row_cols.clear();
            row_vals.clear();
            workspace.base_diag[row] = complex_t{0.0, 0.0};
            for (std::size_t col = first; col < n; ++col) {
                const double value = static_cast<double>(A(row, col));
                if (value == 0.0) continue;
                if (col == row) {
                    workspace.base_diag[row] = complex_t{matrix_sgn * value * dt, 0.0};
                    continue;
                }
                row_cols.push_back(col);
                row_vals.emplace_back(matrix_sgn * value * dt, 0.0);
            }
        }

        for (std::size_t pole = 0; pole < pole_count; ++pole) {
            double rhs_norm = 0.0;
            for (std::size_t row = first; row < n; ++row) {
                const complex_t diag = workspace.base_diag[row] - theta[pole];
                if (detail::magnitude(diag) <= diag_tol) {
                    throw std::runtime_error("milk: CRAM Gauss-Seidel zero diagonal");
                }
                workspace.pole_diag[row] = diag;

                workspace.rhs[row] = complex_t(static_cast<double>(N[row]), 0.0) * alpha[pole];
                workspace.x[row]   = workspace.rhs[row] / diag;
                rhs_norm           = std::max(rhs_norm, detail::magnitude(workspace.rhs[row]));
            }
            rhs_norm = std::max(rhs_norm, 1.0e-30);

            bool converged = false;
            for (int iter = 0; iter < max_iter; ++iter) {
                for (std::size_t row = first; row < n; ++row) {
                    complex_t sum = workspace.rhs[row];

                    const auto& cols = workspace.base_cols[row];
                    const auto& vals = workspace.base_vals[row];
                    for (std::size_t i = 0; i < cols.size(); ++i)
                        sum -= vals[i] * workspace.x[cols[i]];

                    workspace.x[row] = sum / workspace.pole_diag[row];
                }

                double max_residual = 0.0;
                for (std::size_t row = first; row < n; ++row) {
                    complex_t ax = (workspace.base_diag[row] - theta[pole]) * workspace.x[row];

                    const auto& cols = workspace.base_cols[row];
                    const auto& vals = workspace.base_vals[row];
                    for (std::size_t i = 0; i < cols.size(); ++i)
                        ax += vals[i] * workspace.x[cols[i]];

                    max_residual = std::max(max_residual, detail::magnitude(workspace.rhs[row] - ax));
                }

                if (max_residual <= abs_tol + rel_tol * rhs_norm) {
                    converged = true;
                    break;
                }
            }

            if (!converged) {
                throw std::runtime_error("milk: CRAM Gauss-Seidel did not converge");
            }

            for (std::size_t row = first; row < n; ++row)
                workspace.accum[row] += workspace.x[row];
        }

        for (std::size_t row = first; row < n; ++row) {
            double value = alpha0 * static_cast<double>(N[row]) + 2.0 * workspace.accum[row].real();
            if (value < 0.0 && std::abs(value) < 1.0e-12) value = 0.0;
            out[row] = static_cast<T>(value);
        }
    }

    /// @brief Solve a Bateman depletion step with a temporary CRAM workspace.
    /// @param A Bateman transition matrix.
    /// @param N Initial number densities.
    /// @param dt Time step multiplier applied to A.
    /// @param out Output number densities.
    /// @param order CRAM order, either 8 or 16.
    /// @param first First isotope index to evolve; lower indices are copied from N.
    static void solveBatemanCRAM(const Matrix<T>& A, const Vector<T>& N, double dt, Vector<T>& out,
                                 int order = 16, std::size_t first = 0) {
        CramWorkspace workspace;
        solveBatemanCRAM(A, N, dt, out, workspace, order, first);
    }

    /// @brief Return a Bateman depletion result using CRAM8 or CRAM16.
    /// @param A Bateman transition matrix.
    /// @param N Initial number densities.
    /// @param dt Time step multiplier applied to A.
    /// @param order CRAM order, either 8 or 16.
    /// @param first First isotope index to evolve; lower indices are copied from N.
    static Vector<T> solveBatemanCRAM(const Matrix<T>& A, const Vector<T>& N, double dt,
                                      int order = 16, std::size_t first = 0) {
        Vector<T> out(A.rows());
        solveBatemanCRAM(A, N, dt, out, order, first);
        return out;
    }
};

template <Scalar T>
inline Matrix<T> Matrix<T>::inverse() const
    requires std::floating_point<T>
{
    return Solver<T>::inverse(*this);
}

template <Scalar T>
inline Vector<T> Matrix<T>::solve(const Vector<T>& rhs) const
    requires std::floating_point<T>
{
    return Solver<T>::solveGauss(*this, rhs);
}

template <std::floating_point T>
inline Matrix<T> inverse(const Matrix<T>& matrix) {
    return Solver<T>::inverse(matrix);
}

template <std::floating_point T>
inline Vector<T> solve(const Matrix<T>& matrix, const Vector<T>& rhs) {
    return Solver<T>::solveGauss(matrix, rhs);
}

/// @brief Return a Bateman depletion result using CRAM8 or CRAM16.
/// @param matrix Bateman transition matrix.
/// @param initial Initial number densities.
/// @param dt Time step multiplier applied to matrix.
/// @param order CRAM order, either 8 or 16.
/// @param first First isotope index to evolve; lower indices are copied from initial.
template <std::floating_point T = double>
inline Vector<T> solveBatemanCRAM(const Matrix<T>& matrix, const Vector<T>& initial, double dt,
                                  int order = 16, std::size_t first = 0) {
    return Solver<T>::solveBatemanCRAM(matrix, initial, dt, order, first);
}

/// @brief Compute a dot product for positive strides.
/// @param count Number of entries.
/// @param x First input vector.
/// @param incx Stride for x.
/// @param y Second input vector.
/// @param incy Stride for y.
inline double dot(std::size_t count, const double* x, int incx, const double* y, int incy) {
    return detail::dot(x, incx, y, incy, count);
}

/// @brief Copy one strided vector to another.
/// @param count Number of entries.
/// @param src Source vector.
/// @param inc_src Stride for src.
/// @param dst Destination vector.
/// @param inc_dst Stride for dst.
inline void copy(std::size_t count, const double* src, int inc_src, double* dst, int inc_dst) {
    detail::copy_strided(count, src, inc_src, dst, inc_dst);
}

/// @brief Add a scaled vector into another vector.
/// @param count Number of entries.
/// @param scale Scale factor for x.
/// @param x Input vector.
/// @param incx Stride for x.
/// @param y In-out vector.
/// @param incy Stride for y.
inline void addScaled(std::size_t count, double scale, const double* x, int incx, double* y, int incy) {
    detail::axpy(count, scale, x, incx, y, incy);
}

/// @brief Compute row-major matrix multiplication C = alpha*op(A)*op(B) + beta*C.
/// @param a_transpose Whether to transpose A.
/// @param b_transpose Whether to transpose B.
/// @param rows Rows of C.
/// @param cols Columns of C.
/// @param inner Inner dimension.
/// @param alpha Scale factor for op(A)*op(B).
/// @param a Matrix A data.
/// @param lda Leading dimension of A.
/// @param b Matrix B data.
/// @param ldb Leading dimension of B.
/// @param beta Existing C scale factor.
/// @param c In-out matrix C data.
/// @param ldc Leading dimension of C.
inline void multiply(Transpose     a_transpose,
                     Transpose     b_transpose,
                     std::size_t   rows,
                     std::size_t   cols,
                     std::size_t   inner,
                     double        alpha,
                     const double* a,
                     std::size_t   lda,
                     const double* b,
                     std::size_t   ldb,
                     double        beta,
                     double*       c,
                     std::size_t   ldc) {
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            double sum = 0.0;
            for (std::size_t k = 0; k < inner; ++k) {
                const double av = (a_transpose == Transpose::No) ? a[row * lda + k] : a[k * lda + row];
                const double bv = (b_transpose == Transpose::No) ? b[k * ldb + col] : b[col * ldb + k];
                sum += av * bv;
            }
            c[row * ldc + col] = alpha * sum + beta * c[row * ldc + col];
        }
    }
}

/// @brief Factor a row-major dense square matrix with LU pivoting.
/// @param n Matrix order.
/// @param a In-out matrix data.
/// @param lda Leading dimension of a.
/// @param pivots Pivot indices.
inline void factorizeLU(std::size_t n, double* a, std::size_t lda, int* pivots) {
    if (lda < n) {
        throw std::invalid_argument("milk: LU leading dimension is too small");
    }
    detail::lu_factorize_inplace(a, n, lda, pivots);
}

/// @brief Solve AX=B in place with a row-major dense matrix.
/// @param n Matrix order.
/// @param nrhs Number of right-hand sides.
/// @param a In-out coefficient matrix.
/// @param lda Leading dimension of a.
/// @param pivots Pivot workspace of length n.
/// @param b In-out right-hand sides and solution.
/// @param ldb Leading dimension of b.
inline void solveLinearSystem(std::size_t n, std::size_t nrhs, double* a, std::size_t lda,
                              int* pivots, double* b, std::size_t ldb) {
    if (lda < n || ldb < nrhs) {
        throw std::invalid_argument("milk: linear solve leading dimension is too small");
    }
    detail::lu_factorize_inplace(a, n, lda, pivots);
    detail::lu_solve_inplace(a, n, lda, pivots, b, nrhs, ldb);
}

/// @brief Invert a row-major LU-factorized dense square matrix.
/// @param n Matrix order.
/// @param lu In-out LU factors and inverse.
/// @param lda Leading dimension of lu.
/// @param pivots Pivot indices from factorizeLU.
inline void invertLU(std::size_t n, double* lu, std::size_t lda, const int* pivots) {
    if (lda < n) {
        throw std::invalid_argument("milk: inverse leading dimension is too small");
    }

    std::vector<double> inverse(n * lda, 0.0);
    std::vector<double> rhs(n);

    for (std::size_t col = 0; col < n; ++col) {
        std::fill(rhs.begin(), rhs.end(), 0.0);
        rhs[col] = 1.0;
        detail::lu_solve_inplace(lu, n, lda, pivots, rhs.data(), 1, 1);
        for (std::size_t row = 0; row < n; ++row) {
            inverse[row * lda + col] = rhs[row];
        }
    }

    std::copy(inverse.begin(), inverse.end(), lu);
}

} // namespace milk
