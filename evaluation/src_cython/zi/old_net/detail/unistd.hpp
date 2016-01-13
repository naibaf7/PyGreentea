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

#ifndef ZI_NET_DETAIL_UNISTD_HPP
#define ZI_NET_DETAIL_UNISTD_HPP 1

#include <zi/config/config.hpp>

#ifndef ZI_OS_LINUX
#  error "only linux is supported"
#endif

#include <unistd.h>

#include <zi/utility/assert.hpp>

namespace zi {
namespace net {
namespace unistd {

// functions

//using ::pipe;

using ::write;
using ::read;
using ::close;

struct pipe
{
    int pipe_[2];

    pipe()
    {
        ZI_VERIFY_0( ::pipe( pipe_ ) );
    }

    ~pipe()
    {
        ::close( pipe_[ 0 ] );
        ::close( pipe_[ 1 ] );
    }

    inline int in() const
    {
        return pipe_[ 0 ];
    }

    inline int out() const
    {
        return pipe_[ 1 ];
    }

};

} // namespace sock
} // namespace net
} // namespace zi

#endif

