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

#ifndef ZI_COLOR_DETAIL_RGB_HPP
#define ZI_COLOR_DETAIL_RGB_HPP 1

namespace zi {
namespace color {
namespace detail {

template< class T >
class rgb {
private:
    T r_, g_, b_;

public:
    explicit rgb( const T& x = T(), const T& y = T(), const T& z = T() )
        : r_( x ), g_( y ), b_( z )
    {
    }

    T& r() { return r_; }
    T& g() { return g_; }
    T& b() { return b_; }

    const T& r() const { return r_; }
    const T& g() const { return g_; }
    const T& b() const { return b_; }

    const T* data() const { return reinterpret_cast< T* >( this ); }

};

} // namespace detail
} // namespace zi
} // namespace color

#endif

