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
#include <cassert>
#include <cmath>

using std::begin;
using std::end;
using std::abs;

// Range checking vector class, renamed from Stroustrup 2013 "The C++
// Programming Language" 4th ed. sec. 4.4.1.2.
template <typename T>
class safe_vec : public std::vector<T> {
public:
    // use the standard vector constructors
    using std::vector<T>::vector;

    // use "at" to force range checking
    T& operator [](int i) {
        return std::vector<T>::at(i);
    }
    const T& operator [](int i) const {
        return std::vector<T>::at(i);
    }
};

//
// Check if two values (or the values in two containers) are identical, at
// least within the given tolerances.  Two values are considered identical if
// their absolute difference abs(a - b) is less than abs_tolerance or if their
// relative difference (abs(a/b - 1) where abs(a) > abs(b)) is less than
// rel_tolerance.
//

// generic prototype
template <typename T1, typename T2, typename Tolerence>
bool within_tolerance(const T1& t1, const T2& t2,
        Tolerence abs_tolerance, Tolerence rel_tolerance);

// version for two sets of iterators
template <typename Iter1, typename Iter2, typename Tolerence>
bool within_tolerance(Iter1 begin1, Iter1 end1, Iter2 begin2, Iter2 end2,
    Tolerence abs_tolerance, Tolerence rel_tolerance) {

    auto i1 = begin1;
    auto i2 = begin2;
    for (; i1 != end1 && i2 != end2; ++i1, ++i2) {
        if (!within_tolerance(*i1, *i2, abs_tolerance, rel_tolerance))
            return false;
    }

    return i1 == end1 && i2 == end2;
}

// version for two containers
template <typename C1, typename C2, typename Tolerence>
bool within_tolerance(const C1& c1, const C2& c2,
        Tolerence abs_tolerance, Tolerence rel_tolerance) {
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

//
// The main entry point.  For the moment, this just runs self-tests
//

int main() {
    using V = safe_vec<double>;
    using std::cout;
    {
        cout << "testing within_tolerance...\n";
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
    }

    return 0;
}
