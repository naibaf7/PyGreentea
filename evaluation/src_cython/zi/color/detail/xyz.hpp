//
// Copyright (C) 2010  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZI_COLOR_DETAIL_XYZ_HPP
#define ZI_COLOR_DETAIL_XYZ_HPP 1

namespace zi {
namespace color {
namespace detail {

template< class T >
class xyz {
private:
    T x_, y_, z_;

public:
    explicit xyz( const T& x = T(), const T& y = T(), const T& z = T() )
        : x_( x ), y_( y ), z_( z )
    {
    }

    T& x() { return x_; }
    T& y() { return y_; }
    T& z() { return z_; }

    const T& x() const { return x_; }
    const T& y() const { return y_; }
    const T& z() const { return z_; }

};

} // namespace detail
} // namespace zi
} // namespace color

#endif

