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

#ifndef ZI_COLOR_DETAIL_RGB_TAGS_HPP
#define ZI_COLOR_DETAIL_RGB_TAGS_HPP 1

namespace zi {
namespace color {
namespace detail {
namespace tag {

struct adobe1998   {};
struct apple       {};
struct cie         {};
struct color_match {};
struct hdtv        {};
struct ntsc        {};
struct pal         {};
struct sgi         {};
struct smpte_c     {};
struct smpte_240m  {};
struct srgb        {};
struct wide_gamut  {};

} // namespace tag
} // namespace detail
} // namespace zi
} // namespace color

#endif

