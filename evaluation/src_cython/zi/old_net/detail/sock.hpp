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

#ifndef ZI_NET_DETAIL_SOCK_HPP
#define ZI_NET_DETAIL_SOCK_HPP 1

#include <zi/config/config.hpp>

#ifndef ZI_OS_LINUX
#  error "only linux is supported"
#endif

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>

#include <cstring>

namespace zi {
namespace net {
namespace sock {

// functions in socket.h

using ::accept;
using ::bind;
using ::connect;
using ::getpeername;
using ::getsockname;
using ::getsockopt;
using ::htons;
using ::listen;
using ::recv;
using ::recvfrom;
using ::recvmsg;
using ::send;
using ::sendmsg;
using ::sendto;
using ::setsockopt;
using ::shutdown;
using ::socket;
using ::socketpair;

// define structs to be initialized to 0

#define ZI_NET_DEFINE_SOCK_STRUCT( name )               \
    struct name: ::name                                 \
    {                                                   \
        name()                                          \
        {                                               \
            std::memset( this, 0, sizeof( ::name ) );   \
        }                                               \
    }

ZI_NET_DEFINE_SOCK_STRUCT( cmsghdr          );
ZI_NET_DEFINE_SOCK_STRUCT( linger           );
ZI_NET_DEFINE_SOCK_STRUCT( msghdr           );
ZI_NET_DEFINE_SOCK_STRUCT( sockaddr         );
ZI_NET_DEFINE_SOCK_STRUCT( sockaddr_in      );
ZI_NET_DEFINE_SOCK_STRUCT( sockaddr_in6     );
ZI_NET_DEFINE_SOCK_STRUCT( sockaddr_storage );

#undef ZI_NET_DEFINE_SOCK_STRUCT

// types

using ::socklen_t;

} // namespace sock
} // namespace net
} // namespace zi

#endif

