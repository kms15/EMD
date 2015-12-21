//    This project is an attempt to implement empirical mode decomposition
//    in C++11.
//
//    Copyright 2015 Kendrick Shaw
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <cassert>
#include <cmath>
#include <sstream>
#include <functional>
#include <type_traits>
#include <complex>
#include <algorithm>

using std::begin;
using std::end;
using std::abs;

//
// Define some convenient iostream operators for debugging
//

// make it easy to print a pair
template <typename T1, typename T2>
std::ostream& operator << (std::ostream& os, const std::pair<T1,T2>& p) {
    // print out the two elements of the pair as a tuple (e.g. "<2, 4>")
    return os << "<" << p.first << ", " << p.second << ">";
}

// make it easy to print a vector
template <typename T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) {
    // print out the elements of the vector between curly braces (e.g.
    // "{1, 3, 5}").
    os << "{";
    bool first = true;
    for (auto val : v) {
        if (first) {
            first = false;
        } else {
            os << ", ";
        }
        os << val;
    }
    os << "}";
    return os;
}



//
// Check if two values (or the values in two containers) are identical, at
// least within the given tolerances.  Two values are considered identical if
// their absolute difference abs(a - b) is less than abs_tolerance or if their
// relative difference (abs(a/b - 1) where abs(a) > abs(b)) is less than
// rel_tolerance.
//

// generic prototype
template <typename T1, typename T2, typename Tolerance>
bool within_tolerance(const T1& t1, const T2& t2,
        Tolerance abs_tolerance, Tolerance rel_tolerance);

// version for two sets of iterators
template <typename Iter1, typename Iter2, typename Tolerance>
bool within_tolerance(Iter1 begin1, Iter1 end1, Iter2 begin2, Iter2 end2,
    Tolerance abs_tolerance, Tolerance rel_tolerance) {

    auto i1 = begin1;
    auto i2 = begin2;
    for (; i1 != end1 && i2 != end2; ++i1, ++i2) {
        if (!within_tolerance(*i1, *i2, abs_tolerance, rel_tolerance))
            return false;
    }

    return i1 == end1 && i2 == end2;
}

// version for two containers
template <typename C1, typename C2, typename Tolerance>
bool within_tolerance(const C1& c1, const C2& c2,
        Tolerance abs_tolerance, Tolerance rel_tolerance) {
    return within_tolerance(begin(c1), end(c1), begin(c2), end(c2),
            abs_tolerance, rel_tolerance);
}

// version for two scalars (e.g. double, float, or complex)
template <typename Scalar1, typename Scalar2, typename Tolerance>
bool scalars_within_tolerance(const Scalar1& s1, const Scalar2& s2,
        Tolerance abs_tolerance, Tolerance rel_tolerance) {
    assert(abs_tolerance >= 0);
    assert(rel_tolerance >= 0);

    // passes if within absolute tolerance
    if (abs(s1 - s2) <= abs_tolerance) {
        return true;
    }

    // passes if within relative tolerance
    auto ratio = (abs(s1) > abs(s2) ? s1/s2 : s2/s1);
    return (abs(ratio - Scalar1{1}) <= rel_tolerance);
}

// explicit specialization for doubles
template <>
bool within_tolerance(const double& d1, const double& d2,
        double abs_tolerance, double rel_tolerance) {
    return scalars_within_tolerance(d1, d2, abs_tolerance, rel_tolerance);
}

// version for pairs
template <typename T1a, typename T1b, typename T2a, typename T2b,
         typename Tolerance>
bool within_tolerance(const std::pair<T1a,T1b>& p1,
        const std::pair<T2a,T2b>& p2,
        Tolerance abs_tolerance, Tolerance rel_tolerance) {
    // passes if both elements of the pair are within tolerance
    return within_tolerance(p1.first, p2.first,
            abs_tolerance, rel_tolerance) &&
        within_tolerance(p1.second, p2.second,
            abs_tolerance, rel_tolerance);
}

// version for complex numbers
template <>
bool within_tolerance(const std::complex<double>& c1, const std::complex<double>& c2,
        double abs_tolerance, double rel_tolerance) {
    return scalars_within_tolerance(c1, c2, abs_tolerance, rel_tolerance);
}


//
// Find local maxima in a series of points, sorted by x-value.  The return
// value is a pair, with the first element containing the x-values of the
// maxima and the second element containing the y values of the maxima. The
// beginning and end of the sequence are never considered to be local maxima.
// The compare function can be used to define a different definition of
// maxima, e.g. using std::less to find local minima or using a custom
// comparison to compare the absolute values of two complex numbers.
//

// iterator version of interface
template <typename IterX, typename IterY,
        typename IterXOut, typename IterYOut,
        typename Compare =
            std::greater<typename std::iterator_traits<IterY>::value_type>>
std::pair<IterXOut, IterYOut>
find_local_maxima(IterX xbegin, IterX xend, IterY ybegin, IterY yend,
    IterXOut xout, IterYOut yout, Compare compare = Compare{}) {

    // define names for the types of the X and Y values
    using XType = typename std::iterator_traits<IterX>::value_type;
    using YType = typename std::iterator_traits<IterY>::value_type;

    // We won't know if we've found a maxima until we see the value after the
    // the maxima, so we'll need to save the previous value as a potential
    // maxima.
    XType prev_x = std::numeric_limits<XType>::quiet_NaN();
    YType prev_y = std::numeric_limits<YType>::quiet_NaN();

    // We have a maxima if the values were rising but are now dropping.
    bool rising = false;

    // loop over the data points in order
    auto xiter = xbegin;
    auto yiter = ybegin;
    while (xiter != xend && yiter != yend) {
        if (compare(*yiter, prev_y)) {
            // if we're rising, the last point wasn't a maxima
            rising = true;
        } else {
            // if we're falling but were rising, the last point was a maxima
            if (rising) {
                *xout++ = prev_x;
                *yout++ = prev_y;
            }
            rising = false;
        }
        // store the current point and then advance the iterators
        prev_x = *xiter++;
        prev_y = *yiter++;
    }

    // the x and y lists should have been the same length
    assert(xiter == xend && yiter == yend);

    return std::make_pair(xout, yout);
}

// container version of interface
template <typename Xs, typename Ys,
        typename Compare = std::greater<typename Ys::value_type>>
std::pair<std::vector<typename Xs::value_type>,
        std::vector<typename Ys::value_type>>
find_local_maxima(const Xs& xs, const Ys& ys, Compare compare = Compare{}) {

    // confirm that the x and y lists are of the same length
    assert(xs.size() == ys.size());

    // create some temporary variables to store the results
    std::vector<typename Xs::value_type> extrema_xs;
    std::vector<typename Ys::value_type> extrema_ys;

    // use the iterator version to find the maxima
    find_local_maxima(begin(xs), end(xs), begin(ys), end(ys),
            back_inserter(extrema_xs), back_inserter(extrema_ys), compare);

    // return the results, using move to avoid an extra copy.
    return std::make_pair(std::move(extrema_xs), std::move(extrema_ys));
}


//
// Calculate a cubic spline through the given data points using the "natural"
// approach of forcing the second derivative to be zero at the end points.
// Returns the value of this spline at the given new x-values.
//

template <typename Old_Xs, typename Ys, typename New_Xs>
std::vector<typename Ys::value_type>
cubic_spline_interpolate(const Old_Xs& old_xs, const Ys& old_ys,
        const New_Xs& new_xs) {

    using Y_Val = typename Ys::value_type;
    std::vector<Y_Val> new_ys {};
    new_ys.reserve(new_xs.size());

    assert(old_xs.size() > 0);

    if (old_xs.size() == 1) {
        new_ys.assign(new_xs.size(), old_ys.front());
    } else {
        std::vector<Y_Val> d2ys {}; // second derivative of old_ys
        d2ys.reserve(new_xs.size());
        auto dx_begin = old_xs[1] - old_xs[0];
        auto dx_end = old_xs.back() - old_xs[old_xs.size() - 2];
        auto dy_begin = old_ys[1] - old_ys[0];
        auto dy_end = old_ys.back() - old_ys[old_ys.size() - 2];

        // assume the "natural" condition on the left side
        // (i.e. second derivative is 0)
        d2ys.push_back(0);

        if (old_xs.size() == 3) {
            // special case when the second data point is also the second to
            // last data point.
            d2ys.push_back(
                3*(dy_end/dx_end - dy_begin/dx_begin)/(dx_end + dx_begin)
            );
        } else if (old_xs.size() > 3) {
            // we need to solve a tridiagonal system to find the y'' values
            std::vector<Y_Val> diag {};
            std::vector<Y_Val> off_diag {};
            std::vector<Y_Val> rhs {};
            diag.reserve(old_xs.size() - 2);
            off_diag.reserve(old_xs.size() - 3);
            rhs.reserve(old_xs.size() - 2);

            // calculate the diagonal, off-diagonal, and rhs of the
            // linear system
            for (int i = 1; i < old_xs.size() - 1; ++i) {
                auto dx_m = old_xs[i] - old_xs[i-1];
                auto dx_p = old_xs[i+1] - old_xs[i];
                auto dy_m = old_ys[i] - old_ys[i-1];
                auto dy_p = old_ys[i+1] - old_ys[i];

                diag.push_back((dx_p + dx_m)/3);
                rhs.push_back(dy_p/dx_p - dy_m/dx_m);

                // off-diagonal is one shorter than the diagonal and rhs
                if (i < old_xs.size() - 2) {
                    off_diag.push_back(dx_p/6);
                }
            }

            // do the forward substitution pass
            for (int i = 1; i < rhs.size(); ++i) {
                auto q = off_diag[i-1]/diag[i-1];
                diag[i] -= q*off_diag[i-1];
                rhs[i] -= q*rhs[i-1];
            }

            // do the backwards substitution
            for (int i = rhs.size() - 2; i >= 0; --i) {
                auto q = off_diag[i]/diag[i+1];
                rhs[i] -= q*rhs[i+1];
            }

            // store the d2ys
            for (int i = 0; i < rhs.size(); ++i) {
                d2ys.push_back(rhs[i]/diag[i]);
            }
        }

        // assume the "natural" condition on the right side
        // (i.e. second derivative is 0)
        d2ys.push_back(0);

        // we'll walk several iterators through the lists
        auto i_new_x = new_xs.begin();
        auto i_old_x = old_xs.begin();
        auto i_old_y = old_ys.begin();
        auto i_d2y = d2ys.begin();

        // first handle the extrapolated points at the beginning
        while (i_new_x != new_xs.end() && *i_new_x <= *i_old_x) {
            auto slope_begin = dy_begin/dx_begin -
                dx_begin*(d2ys.front()/3 + d2ys[1]/6);
            new_ys.push_back(
                old_ys.front() + slope_begin * (*i_new_x - old_xs.front())
            );
            ++i_new_x;
        }
        // next handle the interpolated points in the middle
        while (i_new_x != new_xs.end()) {
            // find the smallest old x that is greater than the new x
            while (i_old_x != old_xs.end() && *i_old_x <= *i_new_x) {
                ++i_old_x;
                ++i_old_y;
                ++i_d2y;
            }
            if (i_old_x == old_xs.end()) {
                break; // we're done interpolating (and need to extrapolate)
            }

            auto dx = *i_old_x - *(i_old_x - 1);
            auto u = (*i_old_x - *i_new_x) / dx;
            auto v = 1 - u;

            new_ys.push_back(
                *(i_old_y-1)*u + *i_old_y*v +
                    *(i_d2y - 1) * (u*u*u - u)*dx*dx/6 +
                    *i_d2y * (v*v*v - v)*dx*dx/6
            );
            ++i_new_x;
        }
        // finally, handle the extrapolated points at the end
        while (i_new_x != new_xs.end()) {
            auto slope_end = dy_end/dx_end +
                dx_end*(d2ys[d2ys.size() - 2]/6 + d2ys.back()/3);
            new_ys.push_back(
                old_ys.back() + slope_end * (*i_new_x - old_xs.back())
            );
            ++i_new_x;
        }
    }

    return std::move(new_ys);
}


//
// Perform a single sifting pass of the empiric mode decomposition as
// described by Huang et al 1998.  This is performed by fitting a cubic
// spline through the local minima and maxima, then subtracting the mean of
// these two cubic splines from the original data.
//
// If the data contain no local maxima or they contain no local minima, the
// sifting process is aborted and an empty list is returned.
//
template <typename Xs, typename Ys>
std::vector<typename Ys::value_type>
sift(const Xs& xs, const Ys& ys) {
    std::vector<typename Ys::value_type> result;
    result.reserve(ys.size());

    auto maxima = find_local_maxima(xs, ys);
    auto minima = find_local_maxima(xs, ys, std::less<typename Ys::value_type>{});

    if (maxima.first.size() == 0 || minima.first.size() == 0) {
        // no extrema - can't sift.
        return result;
    }

    auto upper_envelope = cubic_spline_interpolate(maxima.first,
        maxima.second, xs);
    auto lower_envelope = cubic_spline_interpolate(minima.first,
        minima.second, xs);

    for (int i = 0; i < upper_envelope.size(); ++i) {
        result.push_back(ys[i] - (upper_envelope[i] + lower_envelope[i])/2);
    }

    return std::move(result);
}

//
// Calculate the difference between two sifting passes.  This is intended to
// match what is described in Huang et al 1998 equation 5.5.  It's not clear
// that equation 5.5 is really what the author's intended, however.  The
// value is described as a "standard deviation" but equation 5.5 lacks the
// normal scaling for length and square root of a typical standard deviation.
// Without the length scaling the expected value will grow without bound as
// the length becomes large, which doesn't really match with the author's
// suggestion of 0.2 to 0.3 as a threshold for a small difference
// (independent of length).  Thus I'm assuming that the right hand side of
// equation 5.5 should be divided by T and raised to the 1/2 power, as is
// typical for a "standard deviation".
//
template <typename T1, typename T2>
typename T1::value_type
sifting_difference(const T1& old_vals, const T2& new_vals) {
    typename T1::value_type sum = 0;

    assert(old_vals.size() == new_vals.size());
    auto i1 = old_vals.begin();
    auto i2 = new_vals.begin();

    for (; i1 != old_vals.end(); ++i1, ++i2) {
        sum += (*i1 - *i2) * (*i1 - *i2) / (*i1 * *i1);
    }

    return sqrt(sum / old_vals.size());
}


//
// Calculate the empirical mode decomposition of a time series.
//
template <typename Xs, typename Ys>
std::vector<std::vector<typename Ys::value_type>>
empirical_mode_decomposition(const Xs& xs, const Ys& ys) {
    using Y_Val = typename Ys::value_type;
    using Imf = std::vector<Y_Val>;

    Imf residual;
    std::vector<Imf> result;

    // until we start subtracting, the residual is the original data
    residual.reserve(ys.size());
    residual.assign(begin(ys), end(ys));

    while (true) {
        auto sifted = sift(xs, residual);

        // stop when we can't sift any more
        if (sifted.size() == 0) {
            break;
        }

        // iterate the sifting process until we reach a stopping condition
        Imf imf {residual};
        while (sifted.size() > 0 && sifting_difference(imf, sifted) > 0.2) {
            imf = std::move(sifted);
            sifted = sift(xs, imf);
        }

        // subtract out the imf from the residual
        for (size_t i = 0; i < residual.size(); ++i) {
            residual[i] -= imf[i];
        }

        result.push_back(imf);
    }

    result.push_back(residual);
    return std::move(result);
}

//
// reverses the bits in an n bit word
//
template <typename T>
T reverse_n_bits(T val, unsigned word_length) {
    using U = typename std::make_unsigned<T>::type;
    U u = static_cast<U>(val);

    // proposed word length should fit the type being used
    // (note: the size of the type must fit the next largest power of 2 bits)
    assert((sizeof(u) >= 8 && word_length <= 64) ||
            (sizeof(u) >= 4 && word_length <= 32) ||
            (sizeof(u) >= 2 && word_length <= 16) ||
            word_length <= 8);

    // we don'u support word lengths over 64 bits at the moment
    assert(word_length <= 64);

    // first swap adjacent bits...
    u = ((u & static_cast<U>(0xAAAAAAAAAAAAAAAAULL)) >> 1) +
        ((u & static_cast<U>(0x5555555555555555ULL)) << 1);
    // ...then adjacent pairs of bits...
    u = ((u & static_cast<U>(0xCCCCCCCCCCCCCCCCULL)) >> 2) +
        ((u & static_cast<U>(0x3333333333333333ULL)) << 2);
    // ...then adjacent nibbles...
    u = ((u & static_cast<U>(0xF0F0F0F0F0F0F0F0ULL)) >> 4) +
        ((u & static_cast<U>(0x0F0F0F0F0F0F0F0FULL)) << 4);

    if (sizeof(u) == 1) {
        return u >> (8 - word_length);
    }

    // ...then adjacent bytes...
    u = ((u & static_cast<U>(0xFF00FF00FF00FF00ULL)) >> 8) +
        ((u & static_cast<U>(0x00FF00FF00FF00FFULL)) << 8);

    if (sizeof(u) == 2) {
        return u >> (16 - word_length);
    }

    // ...then adjacent words...
    u = ((u & static_cast<U>(0xFFFF0000FFFF0000ULL)) >> 16) +
        ((u & static_cast<U>(0x0000FFFF0000FFFFULL)) << 16);

    if (sizeof(u) == 4) {
        return u >> (32 - word_length);
    }

    // ...then adjacent dwords...
    u = ((u & static_cast<U>(0xFFFFFFFF00000000ULL)) >> 32) +
        ((u & static_cast<U>(0x00000000FFFFFFFFULL)) << 32);

    return u >> (64 - word_length);
}


//
// creates a copy of the list where the elements have been shuffled to be
// at an index equal to their original index after it has been bit-reversed.
// The original data is zero-padded as needed to make its length a power
// of 2.
//
template <typename C, typename T=typename C::value_type>
std::vector<T>
bit_reverse_copy(const C& c) {
    unsigned num_address_bits = 0;
    while ((1 << num_address_bits) < c.size()) {
        ++num_address_bits;
    }

    std::vector<T> result(1 << num_address_bits);

    for (size_t i = 0; i < c.size(); ++i) {
        result[reverse_n_bits(i, num_address_bits)] = c[i];
    }

    return std::move(result);
}

//
// Utility function to make a type complex
//
template <typename T>
struct make_complex {
    using type = std::complex<T>;
};
template <typename S>
struct make_complex<std::complex<S>> {
    using type = std::complex<S>;
};

//
// helper function to perform the common core parts of the fft and ifft
// (do not call directly).
//
template <typename C, typename T, bool inverse>
std::vector<T>
fft_core(const C& c) {
    const auto pi = 2 * std::arg(T{0., 1.});

    std::vector<T> result = bit_reverse_copy<C,T>(c);

    size_t n {result.size()};
    size_t log_2_n;
    for (log_2_n = 0; (1 << log_2_n) < n; ++log_2_n) {};

    for (int s = 1; s <= log_2_n; ++s) {
        size_t m {size_t{1} << s};
        T omega_m = std::polar<typename T::value_type>(1.,
            (inverse ? 2 : -2)*pi/m);
        for (size_t k = 0; k < n; k += m) {
            T omega = 1;
            for (size_t j = 0; j < m/2; ++j) {
                T t = omega * result[k + j + m/2];
                T u = result[k + j];
                result[k + j] = u + t;
                result[k + j + m/2] = u - t;
                omega *= omega_m;
            }
        }
    }

    if (inverse) {
        for (auto& c : result) {
            c /= n;
        }
    }

    return std::move(result);
}


//
// Compute the fast Fourier transform of time series data, zero padding as
// needed to make the length of the data set a power of two.
//
template <typename C,
         typename T=typename make_complex<typename C::value_type>::type>
std::vector<T>
fft(const C& c) {
    return fft_core<C,T,false>(c);
}


//
// Compute the inverse fast Fourier transform.
//
template <typename C,
         typename T=typename make_complex<typename C::value_type>::type>
std::vector<T>
ifft(const C& c) {
    return fft_core<C,T,true>(c);
}


//
// Calculate the analytic representation of a time series (by adding i times
// the Hilbert transform of the time series).
//
template <typename C,
         typename T=typename make_complex<typename C::value_type>::type>
std::vector<T>
analytic_representation(const C& c) {
    auto ft = fft(c);

    // multiply the positive frequencies by two and zero out the negative
    // frequencies (leaving the boundary frequencies at 1).
    for (size_t i = 1; i < ft.size()/2; ++i) {
        ft[i] *= 2;
    }
    for (size_t i = ft.size()/2 + 1; i < ft.size(); ++i) {
        ft[i] = 0;
    }

    return ifft(ft);
}


//
// The main entry point.  For the moment, this just runs self-tests
//

int main() {
    // define some abbreviations we'll use below
    using V = std::vector<double>;
    using PV = std::pair<V,V>;
    using C = std::complex<double>;
    using VC = std::vector<std::complex<double>>;

    {
        std::cout << "testing iostream operators...\n";
        // should work for pairs and vectors
        std::stringstream s;
        s << PV{{1,3}, {4, 5, 6}};
        assert(s.str() == "<{1, 3}, {4, 5, 6}>");
    }
    {
        std::cout << "testing within_tolerance...\n";
        // zero should be an acceptable tolerance for an exact match...
        assert(within_tolerance(0., 0., 0., 0.));
        assert(within_tolerance(1., 1., 0., 0.));
        // ...but not if the values don't match.
        assert(!within_tolerance(1., 1.1, 0., 0.));
        assert(!within_tolerance(1.1, 1., 0., 0.));
        // absolute tolerance should work
        assert(within_tolerance(3., 3.1, 0.11, 0.));
        assert(!within_tolerance(3., 3.1, 0.09, 0.));
        assert(within_tolerance(-0.2, 0.1, 0.31, 0.));
        assert(!within_tolerance(0.2, -0.1, 0.29, 0.));
        assert(within_tolerance(0.2, -0.1, 0.31, 0.));
        assert(!within_tolerance(0.2, -0.1, 0.29, 0.));
        assert(!within_tolerance(-3., 3.1, 0.11, 0.));
        assert(!within_tolerance(3., -3.1, 0.11, 0.));
        // relative tolerance should work
        assert(within_tolerance(2., 2.2, 0., 0.11));
        assert(!within_tolerance(2., 2.2, 0., 0.09));
        assert(within_tolerance(2.2, 2., 0., 0.11));
        assert(!within_tolerance(2.2, 2., 0., 0.09));
        assert(!within_tolerance(2., -2.2, 0., 0.11));
        assert(!within_tolerance(-2., 2.2, 0., 0.11));
        assert(!within_tolerance(2.2, -2., 0., 0.11));
        assert(!within_tolerance(-2.2, 2., 0., 0.11));
        // weird zero crossing case for rel_tolerance > 2
        assert(within_tolerance(-2., 2.2, 0., 2.11));
        assert(!within_tolerance(-2., 2.2, 0., 2.09));
        // empty lists should be the same, but not list of differing size
        assert(within_tolerance(V{}, V{}, 0., 0.));
        assert(!within_tolerance(V{1.}, V{}, 0., 0.));
        assert(!within_tolerance(V{}, V{1.}, 0., 0.));
        // tolerance checks should check every element in a container
        assert(within_tolerance(V{2.}, V{2.2}, 0., 0.11));
        assert(!within_tolerance(V{2.}, V{2.2}, 0., 0.09));
        assert(within_tolerance(V{2., 0, 0}, V{2.2, 0, 0}, 0., 0.11));
        assert(!within_tolerance(V{2., 0, 0}, V{2.2, 0, 0}, 0., 0.09));
        assert(within_tolerance(V{0, 2, 0}, V{0, 2.2, 0}, 0., 0.11));
        assert(!within_tolerance(V{0, 2, 0}, V{0, 2.2, 0}, 0., 0.09));
        assert(within_tolerance(V{2, 0, 0}, V{2.2, 0, 0}, 0., 0.11));
        assert(!within_tolerance(V{2, 0, 0}, V{2.2, 0, 0}, 0., 0.09));
        // should support pairs
        assert(within_tolerance(std::pair<double,double>{2., 0.},
                    std::pair<double,double>{2.2, 0.}, 0., 0.11));
        assert(!within_tolerance(std::pair<double,double>{2., 0.},
                    std::pair<double,double>{2.2, 0.}, 0., 0.09));
        assert(within_tolerance(std::pair<double,double>{0., 2.},
                    std::pair<double,double>{0., 2.2}, 0., 0.11));
        assert(!within_tolerance(std::pair<double,double>{0., 2.},
                    std::pair<double,double>{0., 2.2}, 0., 0.09));
    }
    {
        std::cout << "testing find_local_maxima...\n";
        // should not find any maxima if the series is monotonic
        assert(within_tolerance(find_local_maxima(V{}, V{}), PV{{},{}},
            0., 0.));
        assert(within_tolerance(find_local_maxima(V{1.}, V{3.}), PV{{},{}},
            0., 0.));
        assert(within_tolerance(find_local_maxima(V{1., 2.}, V{3., 6.}),
            PV{{},{}}, 0., 0.));
        assert(within_tolerance(
            find_local_maxima(V{1., 2., 3.}, V{3., 6., 7.}),
            PV{{},{}}, 0., 0.));
        // should find maxima in the simple cases
        assert(within_tolerance(find_local_maxima(V{1, 2, 3}, V{3, 8, 7}),
            PV{{2},{8}}, 0., 0.));
        assert(within_tolerance(
            find_local_maxima(V{1, 2, 3, 4}, V{3, 2, 4, 1}),
            PV{{3},{4}}, 0., 0.));
        assert(within_tolerance(
            find_local_maxima(V{1, 2, 3, 4, 5, 6}, V{3, 2, 4, 1, 6, 2}),
            PV{{3, 5},{4, 6}}, 0., 0.));
        // should gracefully handle ties by picking the first one
        // (which is the specified behavior for a dataset within epsilon
        // of the data we were given)
        assert(within_tolerance(
            find_local_maxima(V{1, 2, 3, 4, 5, 6}, V{3, 2, 4, 6, 6, 2}),
            PV{{4},{6}}, 0., 0.));
        assert(within_tolerance(
            find_local_maxima(V{1, 2, 3, 4, 5, 6}, V{2, 2, 6, 6, 6, 2}),
            PV{{3},{6}}, 0., 0.));
        // should accept a compare function, which can be used to find minima
        assert(within_tolerance(
            find_local_maxima(V{1, 2, 3, 4, 5, 6}, V{3, 2, 4, 1, 6, 2},
                std::less<double>{}),
            PV{{2, 4}, {2, 1}}, 0., 0.));
    }
    {
        std::cout << "testing cubic_spline_interpolate...\n";

        // with one original data point, should assume a constant function
        assert(within_tolerance(
            cubic_spline_interpolate(V{1.1}, V{2.2}, V{0.0, 1.1, 3.3, 4.4}),
            V{2.2, 2.2, 2.2, 2.2}, 0., 0.));
        // with two original data points, should assume a linear function
        assert(within_tolerance(
            cubic_spline_interpolate(V{0.1, 1.2}, V{2.3, 3.5},
                V{0.0, 0.1, 0.3, 1.2, 1.5}),
            V{2.190909, 2.300000, 2.518182, 3.500000, 3.827273},
           1e-6, 0.));
        // should use cubic splines for 3 or more datapoints
        // (test data from R "spline" function with method="natural")
        assert(within_tolerance(
            cubic_spline_interpolate(V{0.1, 1.2, 3.3}, V{2.3, 4.5, 1.2},
                V{0.0, 0.1, 0.3, 1.2, 1.5, 3.3, 4.2}),
            V{2.038616, 2.300000, 2.818709, 4.500000, 4.588202, 1.200000,
                -1.268973},
            1e-6, 0.));
        assert(within_tolerance(
            cubic_spline_interpolate(
                V{0.1, 1.2, 3.3, 3.7, 4.3}, V{2.3, 4.5, 1.2, 2.4, 2.8},
                V{1.1, 2.1, 3.1, 4.1}
            ),
            V{4.4984771, 2.5856026, 0.8846014, 2.8224232},
            1e-7, 0.));
    }
    {
        std::cout << "testing sift...\n";

        assert(within_tolerance(
            sift(V{1.2, 2.1, 3.4, 3.7, 4.2,  5.0, 6.0,  6.5, 7.2, 8.0},
                V{1.2, 0.8, 3.4, 3.5, 2.7, -0.1, 0.3, -0.5, 2.1, 1.4}),
            V{ -4.022762880, -3.223043478, 1.108004434,  1.606285807,
                1.457152560, -0.475425331, 0.334987809, -0.660552536,
                1.390727534, -0.003844725},
            1e-9, 0.
        ));

        // should return false if monotonic
        assert(within_tolerance(
            sift(V{1.0, 2.1, 3.4, 5.5}, V{2.0, 2.1, 3.0, 3.0}),
            V{}, 0., 0.));
    }
    {
        std::cout << "testing sifting_difference...\n";

        // Note: best guess on what the author meant - see comment in function
        assert(within_tolerance(
            sifting_difference(V{0.1, 0.3, -0.2}, V{0.11, -0.1, -0.21}),
             0.7725019, 1e-7, 0.));
    }
    {
        std::cout << "testing empirical_mode_decomposition...\n";
        auto xs = V{1.2, 2.1, 3.4, 3.7, 4.2,  5.0, 6.0,  6.5, 7.2, 8.0};
        auto ys = V{1.2, 0.8, 3.4, 3.5, 2.7, -0.1, 0.3, -0.5, 2.1, 1.4};
        auto imfs = empirical_mode_decomposition(xs, ys);

        // should be at least one IMF
        assert(imfs.size() > 0);

        // sum of all IMFs should be the original signal
        std::vector<double> sum(ys.size());
        for (auto imf : imfs) {
            assert(imf.size() == ys.size());

            for (size_t i = 0; i < imf.size(); ++i) {
                sum[i] += imf[i];
            }
        }
        assert(within_tolerance(sum, ys, 1e-14, 0.));

        // all IMFs should be at the point where further sifting will not
        // improve them.
        for (auto imf : imfs) {
            auto sifted = sift(xs, imf);
            assert(sifted.size() == 0 ||
                sifting_difference(imf, sifted) <= 0.2);
        }

        // residual should be monotonic or within tolerance
        auto residual = imfs.back();
        assert(find_local_maxima(xs, residual).first.size() == 0 ||
            find_local_maxima(xs, residual, std::less<double>{}).first.size()
                == 0);
    }
    {
        std::cout << "testing reverse_64_bits...\n";
        assert(reverse_n_bits(0x8001200400106072ULL, 64) ==
                0x4E06080020048001ULL);
        assert(reverse_n_bits(0x800120040010607ULL, 60) ==
                0xE06080020048001ULL);
        assert(reverse_n_bits('\x35',6) == '\x2b');
        assert(reverse_n_bits(0xABADCAFE,32) == 0x7F53B5D5);
        assert(reverse_n_bits(0xBADCAFE,28) == 0x7F53B5D);
        assert(reverse_n_bits(0xADCAFE,24) == 0x7F53B5);
        assert(reverse_n_bits(static_cast<short>(0xBAD),12) == 0xB5D);
    }
    {
        std::cout << "testing bit_reverse_copy...\n";
        assert(within_tolerance(
            bit_reverse_copy(V{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}),
            V{0.1, 0.5, 0.3, 0.7, 0.2, 0.6, 0.4, 0.8}, 0., 0.));
        // should 0 pad as needed
        assert(within_tolerance(
            bit_reverse_copy(V{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
            V{0.1, 0.5, 0.3, 0.0, 0.2, 0.6, 0.4, 0.0}, 0., 0.));
    }
    {
        std::cout << "testing fft...\n";
        auto ys = V{1.2, 0.8, 3.4, 3.5, 2.7, -0.1, 0.3, -0.5};
        // should match results from R
        assert(within_tolerance(fft(ys), VC{
            C{11.300000, 0.0000000}, C{-3.692031,-6.5648232},
            C{ 0.200000, 2.3000000}, C{ 0.692031,-0.3648232},
            C{ 3.900000, 0.0000000}, C{ 0.692031, 0.3648232},
            C{ 0.200000,-2.3000000}, C{-3.692031, 6.5648232}},
            1e-6, 0.));
    }
    {
        std::cout << "testing ifft...\n";
        // ifft should reverse the fft
        auto ys = V{1.2, 0.8, 3.4, 3.5, 2.7, -0.1, 0.3, -0.5};
        auto cys = VC{};
        std::transform(begin(ys), end(ys), back_inserter(cys),
                [](std::complex<double> x) { return std::real(x); });
        assert(within_tolerance(ifft(fft(ys)), cys, 1e-15, 0.));
    }
    {
        std::cout << "testing analytic_representation...\n";
        V input;
        VC expected;
        const double pi = 2 * std::atan2(0.,1.);
        for (int k = 0; k < 1024; ++k) {
            input.push_back(std::cos(2*pi*k/128));
            expected.push_back(C{std::cos(2*pi*k/128), std::sin(2*pi*k/128)});
        }
        assert(within_tolerance(analytic_representation(input), expected, 1e-15, 0.));
    }

    return 0;
}
