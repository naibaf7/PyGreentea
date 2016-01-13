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

#ifndef ZI_COLOR_DETAIL_RGB_XYZ_CONVERSIONS_HPP
#define ZI_COLOR_DETAIL_RGB_XYZ_CONVERSIONS_HPP 1

#include <zi/color/detail/rgb_tags.hpp>

namespace zi {
namespace color {
namespace detail {

template< class Tag >
struct xyz_rgb_converter
{
//private:
    static const double matrix_[9];
};

template< class Tag >
struct rgb_xyz_converter
{
//private:
    static const double matrix_[9];
};

#define ZI_XYZ_RGB_CONVERTER( tag, a1, a2, a3, a4, a5, a6, a7, a8, a9 ) \
    template<>                                                          \
    const double xyz_rgb_converter< tag >::matrix_[] =                  \
    { a1, a2, a3, a4, a5, a6, a7, a8, a9 }

#define ZI_RGB_XYZ_CONVERTER( tag, a1, a2, a3, a4, a5, a6, a7, a8, a9 ) \
    template<>                                                          \
    const double rgb_xyz_converter< tag >::matrix_[] =                  \
    { a1, a2, a3, a4, a5, a6, a7, a8, a9 }

ZI_XYZ_RGB_CONVERTER( tag::adobe1998, 2.0414, -0.5649, -0.3447, -0.9693, 1.8760, 0.0416, 0.0134, -0.1184, 1.0154 );
ZI_RGB_XYZ_CONVERTER( tag::adobe1998, 0.5767, 0.1856, 0.1882, 0.2974, 0.6273, 0.0753, 0.0270, 0.0707, 0.9911 );


} // namespace detail
} // namespace zi
} // namespace color

#endif

