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

// Check if the contents of two containers are identical, at least within the
// given tolerances.  Two values are considered identical if their absolute
// difference abs(a - b) is less than abs_tolerance or if their relative
// difference (abs(a/b - 1) where abs(a) > abs(b)) is less than rel_tolerance.
template <typename C1, typename C2, typename T>
bool within_tolerance(const C1& c1, const C2& c2, const T& abs_tolerance,
        const T& rel_tolerance) {
    // To match the containers must have the same number of elements
    if (c1.size() != c2.size()) {
        return false;
    }

    // check to see if each of the elements is within tolerance
    auto i1 = begin(c1);
    auto i2 = begin(c2);
    for (; i1 != end(c1); ++i1, ++i2) {
        if (!within_tolerance(*i1, *i2, abs_tolerance, rel_tolerance))
            return false;
    }

    return true;
}

template <>
bool within_tolerance(const double& d1, const double& d2, const double& abs_tolerance,
        const double& rel_tolerance) {
    assert(abs_tolerance >= 0);
    assert(rel_tolerance >= 0);

    // passes if within absolute tolerance
    if (abs(d1 - d2) <= abs_tolerance) {
        return true;
    }

    // passes if within relative tolerance
    auto ratio = (abs(d1) > abs(d2) ? d1/d2 : d2/d1);
    return (abs(ratio - 1) <= rel_tolerance);
}

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
