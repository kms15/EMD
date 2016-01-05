#ifndef EMD_H
#define EMD_H

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
empirical_mode_decomposition(const Xs& xs, const Ys& ys, unsigned max_siftings = 50) {
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
        int num_siftings=0;
        Imf imf {residual};
        std::cout << "  computing IMF " << result.size() + 1 << std::flush;
        while ((max_siftings == 0 || num_siftings < max_siftings) &&
                sifted.size() > 0 && sifting_difference(imf, sifted) > 0.2) {
            ++num_siftings;
            std::cout << "." << std::flush;
            imf = std::move(sifted);
            sifted = sift(xs, imf);
        }
        std::cout << "\n";

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
template <typename Ys,
         typename T=typename make_complex<typename Ys::value_type>::type>
std::vector<T>
analytic_representation(const Ys& ys) {
    auto ws = fft(ys);

    // multiply the positive frequencies by two and zero out the negative
    // frequencies (leaving the boundary frequencies at 1).
    for (size_t i = 1; i < ws.size()/2; ++i) {
        ws[i] *= 2;
    }
    for (size_t i = ws.size()/2 + 1; i < ws.size(); ++i) {
        ws[i] = 0;
    }

    return ifft(ws);
}


//
// Calculate the derivative of a time series using finite differences.  The
// forward difference is used for the first data point, the backwards
// difference for the last data point, and the central difference for all
// other data points.  An optional difference function can be supplied to
// calculate the difference between two adjacent values; this can be
// important to specify in cases such as angles, where the difference between
// 3/4 pi radians and -3/4 pi radians may be 1/2 pi radians (and not -6/4 pi
// radians).
//
template <typename Ys, typename T=typename Ys::value_type,
         typename Diff = std::minus<T>>
std::vector<T>
derivative(const Ys& ys, Diff diff = Diff{}) {
    std::vector<T> result;
    result.reserve(ys.size());

    result.push_back(diff(ys[1], ys[0]));
    for (size_t i = 1; i < ys.size() - 1; ++ i) {
        // note: the ys[i] terms will cancel out with the standard definition
        // of diff, but are important for some definitions of diff (e.g. when
        // calculating angular velocities for a sequence like
        // (pi/3, pi, -pi/3).)
        result.push_back((diff(ys[i + 1],ys[i]) + diff(ys[i], ys[i - 1]))/2);
    }
    result.push_back(diff(ys[ys.size() - 1], ys[ys.size() - 2]));

    return std::move(result);
}


//
// calculate the closest difference between two values on a cyclic system,
// e.g. angles on a circle or modulo arithmetic.  Note that this is sometimes
// the conventional difference, e.g. the difference between 1/4 pi radians
// and -1/4 pi radians is 1/2 pi radians, and sometimes not the conventional
// difference, e.g. the difference between 3/4 pi radians and -3/4 pi radians
// is also 1/2 pi radians.
//
template <class T>
std::function<T (T, T)>
cyclic_difference(T cycle_size){
    return [cycle_size](T a, T b) {
        T t = a - b;
        if (t > cycle_size/2)
            return t - cycle_size;
        else if (t < -cycle_size/2)
            return t + cycle_size;
        else
            return t;
    };
}


//
// Calculate the instantaneous frequency and amplitude of a signal using the
// Hilbert transform.  Note that this is only well defined for signals that
// have a relatively low bandwidth at a given moment in time.
//
template <typename Ys, typename T=typename Ys::value_type>
std::pair<std::vector<T>, std::vector<T>>
instantaneous_frequency_and_amplitude(const Ys& ys) {
    const T pi = 2*std::arg(std::complex<T>(0., 1.));
    auto zs = analytic_representation(ys);

    std::vector<T> angle;
    std::transform(zs.begin(), zs.end(), std::back_inserter(angle),
        [](std::complex<double> z) { return std::arg(z); });

    std::vector<T> amplitude;
    std::transform(zs.begin(), zs.end(), std::back_inserter(amplitude),
        [](std::complex<double> z) { return std::abs(z); });

    // calculate the frequency from the angular velocity
    auto frequency = derivative(angle, cyclic_difference(2*pi));
    for (auto& f : frequency) {
        f /= 2*pi;
    }

    return std::pair<std::vector<T>, std::vector<T>>{
        std::move(frequency), std::move(amplitude)
    };
}




#endif /* EMD_H */
