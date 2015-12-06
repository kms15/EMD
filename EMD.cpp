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
#include <vector>
#include <limits>
#include <cassert>
#include <cmath>
#include <sstream>
#include <functional>

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
    return (abs(ratio - 1) <= rel_tolerance);
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
template <typename XS, typename YS,
        typename Compare = std::greater<typename YS::value_type>>
std::pair<std::vector<typename XS::value_type>,
        std::vector<typename YS::value_type>>
find_local_maxima(const XS& xs, const YS& ys, Compare compare = Compare{}) {

    // confirm that the x and y lists are of the same length
    assert(xs.size() == ys.size());

    // create some temporary variables to store the results
    std::vector<typename XS::value_type> extrema_xs;
    std::vector<typename YS::value_type> extrema_ys;

    // use the iterator version to find the maxima
    find_local_maxima(begin(xs), end(xs), begin(ys), end(ys),
            back_inserter(extrema_xs), back_inserter(extrema_ys), compare);

    // return the results, using move to avoid an extra copy.
    return std::make_pair(std::move(extrema_xs), std::move(extrema_ys));
}


//
// The main entry point.  For the moment, this just runs self-tests
//

int main() {
    using V = std::vector<double>;
    using PV = std::pair<V,V>;

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

    return 0;
}
