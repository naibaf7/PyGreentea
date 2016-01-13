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

#ifndef ZI_ATOMIC_FD_SET_WIN32_HPP
#define ZI_ATOMIC_FD_SET_WIN32_HPP 1

#include <zi/net/config.hpp>
#include <zi/net/detail/socket_types.hpp>

namespace zi {
namespace net {
namespace detail {

class fd_set
{
private:
    mutable ::fd_set fd_set_;
    socket_type      fd_max_;

public:
    fd_set(): fd_max_( invalid_socket )
    {
        FD_ZERO( &fd_set_ );
    }

    bool set( socket_type fd )
    {
        if ( fd >= 0 && fd < static_cast< socket_type >( FD_SETSIZE ) )
        {
            if ( fd_max_ < fd )
            {
                fd_max_ = fd;
            }
            return true;
        }
        return false;
    }

    bool is_set( socket_type fd ) const
    {
        return FD_ISSET( fd, &fd_set_ ) != 0;
    }

    socket_type max_fd() const
    {
        return fd_max_;
    }

    operator ::fd_set*()
    {
        return &fd_set_;
    }

};

} // namespace detail
} // namespace net
} // namespace zi

#endif
