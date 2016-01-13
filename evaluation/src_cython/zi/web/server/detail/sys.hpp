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

#ifndef ZI_WEB_SERVER_DETAIL_SYS_HPP
#define ZI_WEB_SERVER_DETAIL_SYS_HPP 1

#include <zi/config/config.hpp>

#ifndef ZI_OS_LINUX
#  error "only linux is supported"
#endif

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/poll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <fcntl.h>
#include <netdb.h>

#include <cerrno>

namespace zi {
namespace web {
namespace server {
namespace detail {

using ::close;
using ::connect;
using ::htons;
using ::fcntl;
using ::freeaddrinfo;
using ::gai_strerror;
using ::getaddrinfo;
using ::gethostbyname;
using ::getnameinfo;
using ::getpeername;
using ::getsockopt;
using ::inet_addr;
using ::pipe;
using ::poll;
using ::read;
using ::recv;
using ::select;
using ::setsockopt;
using ::send;
using ::shutdown;
using ::socket;
using ::write;


using ::addrinfo;
using ::fd_set;
using ::hostent;
using ::in_addr_t;
using ::in_addr;
using ::linger;
using ::pollfd;
using ::sockaddr;
using ::sockaddr_in;
using ::sockaddr_storage;
using ::socklen_t;

} // namespace detail
} // namespace server
} // namespace web
} // namespace zi

#endif

