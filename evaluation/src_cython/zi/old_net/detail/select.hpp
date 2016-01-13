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

#ifndef ZI_NET_DETAIL_SELECT_HPP
#define ZI_NET_DETAIL_SELECT_HPP 1

#include <zi/config/config.hpp>

#ifndef ZI_OS_LINUX
#  error "only linux is supported"
#endif

#include <sys/select.h>
#include <cstddef>

namespace zi {
namespace net {
namespace select {

struct limits
{
    static const int fd_setsize = FD_SETSIZE;
};

// struct fd_set

struct fd_set: ::fd_set
{
    fd_set()
    {
        FD_ZERO( reinterpret_cast< ::fd_set* >( this ) );
    }
};


inline void set( int fd, const fd_set& fds )
{
    FD_SET( fd, const_cast< ::fd_set* >( reinterpret_cast< const ::fd_set* >( &fds ) ) );
}

inline bool is_set( int fd, const fd_set& fds )
{
    return FD_ISSET( fd, const_cast< ::fd_set* >( reinterpret_cast< const ::fd_set* >( &fds ) ) );
}

inline void zero( const fd_set& fds )
{
    FD_ZERO( const_cast< ::fd_set* >( reinterpret_cast< const ::fd_set* >( &fds ) ) );
}

inline void clear( int fd, const fd_set& fds )
{
    FD_CLR( fd, const_cast< ::fd_set* >( reinterpret_cast< const ::fd_set* >( &fds ) ) );
}

inline int select( int nfds,
                   fd_set* rd_fds, fd_set* wr_fds, fd_set* ex_fds,
                   ::timeval *tv )
{
    return ::select( nfds,
                     reinterpret_cast< ::fd_set* >( rd_fds ),
                     reinterpret_cast< ::fd_set* >( wr_fds ),
                     reinterpret_cast< ::fd_set* >( ex_fds ),
                     tv );
}

// functions in select.h

using ::pselect;
using ::select;




} // namespace select
} // namespace net
} // namespace zi

#endif

