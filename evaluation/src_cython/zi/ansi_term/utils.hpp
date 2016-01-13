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

#ifndef ZI_ANSI_TERM_UTILS_HPP
#define ZI_ANSI_TERM_UTILS_HPP 1

#include <zi/config/config.hpp>

#if defined( ZI_OS_LINUX )
#  include <cstdio>
#  include <sys/ioctl.h>
#endif

namespace zi {
namespace tos {
namespace detail {

inline int get_term_width( int default_width = 80 )
{

    int cols = 0;

#if defined( TIOCGSIZE )

    struct ttysize ts;
    ioctl( STDOUT_FILENO, TIOCGSIZE, &ts );
    cols = static_cast< int >( ts.ts_cols );

#elif defined( TIOCGWINSZ )

    struct winsize ts;
    ioctl( STDOUT_FILENO, TIOCGWINSZ, &ts );
    cols = static_cast< int >( ts.ws_col );

#endif

    return cols ? cols : default_width;

}

} // namespace detail
} // namespace tos
} // namespace zi

#endif
