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
#include <fstream>
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

#include "emd.h"


// external libraries
#include <edflib.h>

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
// Run self-tests.  These will fail with asserts (so override assert to th
//
void run_self_tests() {
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
        const double pi = 2 * std::atan2(1.,0.);
        for (int k = 0; k < 1024; ++k) {
            input.push_back(std::cos(2*pi*k/128));
            expected.push_back(
                C{std::cos(2*pi*k/128), std::sin(2*pi*k/128)});
        }
        assert(within_tolerance(analytic_representation(input), expected,
            1e-13, 0.));

        // should work for non-power of 2 sizes
        input.resize(768);
        expected.resize(768);
        assert(analytic_representation(input).size() == expected.size());
    }
    {
        std::cout << "testing cyclic difference...\n";
        // should calculate the normal difference when it's less than half
        // the cycle size...
        assert(within_tolerance(cyclic_difference(5.)(-1., 1.499999),
            -2.499999, 1e-15, 0.));
        assert(within_tolerance(cyclic_difference(5.)(1., -1.499999),
            2.499999, 1e-15, 0.));
        // should calculate the wraparound difference when it's greater than
        // half of the cycle size
        assert(within_tolerance(cyclic_difference(5.)(-1., 1.51),
            2.49, 1e-15, 0.));
        assert(within_tolerance(cyclic_difference(5.)(1., -1.51),
            -2.49, 1e-15, 0.));
    }
    {
        std::cout << "testing derivative...\n";
        // should do forward, central, and backwards differences where
        // appropriate.
        assert(within_tolerance(derivative(
                V{1.2, 0.8, 3.4, 3.5, 2.7, -0.1, 0.3, -0.5, 2.1, 1.4}
            ),
            V{-0.4, 1.1, 1.35, -0.35, -1.8, -1.2, -0.2, 0.9, 0.95, -0.7},
            1e-15, 0.));
        // should handle branch cuts if a difference function is passed to it
        assert(within_tolerance(derivative(V{0.9, 0.1, 0.2, 0.8, 0.1},
                cyclic_difference(1.)),
            V{0.2, 0.15, -0.15, -0.05, 0.3},
            1e-15, 0.));
    }
    {
        std::cout << "testing instantaneous_frequency_and_amplitude...\n";
        const double pi = 2 * std::atan2(1.,0.);
        // should correctly calculate constant frequency and amplitude for a
        // sine wave.
        {
            V input;
            V expected_frequency(1024, 1./128);
            V expected_amplitude(1024, 2.5);

            for (int k = 0; k < 1024; ++k) {
                input.push_back(2.5*std::cos(2*pi*k/128));
            }
            auto result = instantaneous_frequency_and_amplitude(input);
            assert(within_tolerance(result.first, expected_frequency,
                1e-13, 0.));
            assert(within_tolerance(result.second, expected_amplitude,
                1e-13, 0.));
        }
        // Inputs and outputs should have the same length.
        {
            V input;
            V expected_frequency(1000, 1./128);
            V expected_amplitude(1000, 2.5);

            for (int k = 0; k < 1000; ++k) {
                input.push_back(2.5*std::cos(2*pi*k/128));
            }
            auto result = instantaneous_frequency_and_amplitude(input);
            assert(result.first.size() == input.size());
            assert(result.second.size() == input.size());
        }
        // should capture an exponential decay in amplitude, at least away
        // from the edges.
        {
            V input;
            V expected_center_frequency(512, 1./128);
            V expected_center_amplitude;

            for (int k = 0; k < 1024; ++k) {
                if (k >= 256 && k < 768) {
                    expected_center_amplitude.push_back(exp(-k/1024.));
                }
                input.push_back(exp(-k/1024.)*std::cos(2*pi*k/128));
            }
            auto result = instantaneous_frequency_and_amplitude(input);

            V center_frequency;
            center_frequency.assign(result.first.begin() + 256, result.first.begin() + 768);
            assert(within_tolerance(center_frequency, expected_center_frequency,
                1e-4, 0.));

            V center_amplitude;
            center_amplitude.assign(result.second.begin() + 256, result.second.begin() + 768);
            assert(within_tolerance(center_amplitude, expected_center_amplitude,
                2.5e-3, 0.));
        }
        // should capture a gradual frequency change, at least away
        // from the edges.
        {
            V input;
            V expected_center_frequency;
            V expected_center_amplitude(512, 2.5);

            for (int k = 0; k < 1024; ++k) {
                if (k >= 256 && k < 768) {
                    expected_center_frequency.push_back((k/1024. + 1)/32. + k/1024./32);
                }
                input.push_back(2.5*std::cos(2*pi*k*(k/1024. + 1)/32.));
            }
            auto result = instantaneous_frequency_and_amplitude(input);

            V center_frequency;
            center_frequency.assign(result.first.begin() + 256, result.first.begin() + 768);
            assert(within_tolerance(center_frequency, expected_center_frequency,
                1e-4, 0.));

            V center_amplitude;
            center_amplitude.assign(result.second.begin() + 256, result.second.begin() + 768);
            assert(within_tolerance(center_amplitude, expected_center_amplitude,
                1e-3, 0.));
        }
    }
}



//
// The main entry point.  For the moment, this just runs self-tests
//

int main(int argc, char** argv) {
    std::string input_filename;
    std::string spectrum_filename;
    bool run_tests {false};

    // parse the command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == std::string("--run-tests")) {
            run_tests = true;
        } else if (argv[i] == std::string("--generate-spectrum")) {
            if ((i+1 >= argc) || (argv[i+1][0] == '-')) {
                std::cerr << "missing output filename for " <<
                    "--generate-spectrum option\n";
                return 1;
            } else if (!spectrum_filename.empty()) {
                std::cerr << "--generate-spectrum appeared more than once" <<
                    " on the command line.\n";
                return 1;
            } else {
                spectrum_filename = argv[i+1];
                ++i; // eat the filename argument
            }
        } else if (argv[i][0] == '-') {
            std::cerr << "unknown option '" << argv[i] << "' specified.\n";
            return 1;
        } else if (!input_filename.empty()) {
            std::cerr << "Multiple input files specified: '" <<
                input_filename << "' and '" << argv[i] << "'.\n";
            return 1;
        } else {
            input_filename = argv[i];
        }
    }

    if (run_tests) {
        run_self_tests();
    }

    if (!spectrum_filename.empty()) {
        // read the edf file
        edf_hdr_struct edf_header;
        if (edfopen_file_readonly(input_filename.c_str(), &edf_header,
                EDFLIB_DO_NOT_READ_ANNOTATIONS) != 0) {
            std::cerr << "Error opening input file '" << input_filename <<
                "'.\n";
            return 1;
        }

        std::vector<double> data(edf_header.signalparam[0].smp_in_file);
        edfread_physical_samples(edf_header.handle, 0, data.size(),
                data.data());
        //data.resize(100000); // uncomment to truncate data for quick tests

        std::vector<double> times;
        times.reserve(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            times.push_back(edf_header.file_duration * 1e-7 * i / (data.size() - 1));
        }
        edfclose_file(edf_header.handle);

        // take the EMD
        std::cout << "Computing empirical mode decomposition\n";
        auto emd = empirical_mode_decomposition(times, data);

        // compute the Hilbert spectrum
        std::cout << "Computing Hilbert spectrum" << std::flush;

        constexpr size_t x_bin_size = 8192; // samples per x bin
        size_t num_x_bins = (data.size() + (x_bin_size - 1)) / x_bin_size;
        constexpr size_t num_y_bins = 1024;
        const double max_frequency = 0.5; // Nyquist limit
        const double y_bin_size = max_frequency / num_y_bins;

        Binned_spectrum<> spectrum(num_x_bins, num_y_bins, x_bin_size, y_bin_size);

        for (const auto& imf : emd) {
            std::cout << "." << std::flush;
            auto freq_amp = instantaneous_frequency_and_amplitude(imf);
            spectrum.add_trace(freq_amp.first, freq_amp.second);
        }
        std::cout << "\n";

        // save the spectrum as a csv file
        std::cout << "Saving spectrum\n";
        std::ofstream spectrum_file{spectrum_filename};

        if (!spectrum_file) {
            std::cerr << "Error creating spectrum file '" <<
                spectrum_filename << "'.\n";
            return 1;
        }

        spectrum_file << spectrum;
    }

    if (argc <= 1) {
        std::cerr << "Usage:\n"
            << "  " << argv[0] << " --run-tests\n"
            << "  " << argv[0] << " edffile --generate-spectrum spectrumfile\n";
        return 1;
    }

    return 0;
}
