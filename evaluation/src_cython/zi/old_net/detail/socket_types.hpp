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

#ifndef ZI_NET_DETAIL_SOCKET_TYPES_HPP
#define ZI_NET_DETAIL_SOCKET_TYPES_HPP 1

#include <zi/net/config.hpp>

#if defined( ZI_OS_LINUX ) || defined( ZI_OS_MACOS )
#  include <sys/ioctl.h>
#  include <sys/poll.h>
#  include <sys/types.h>
#  include <sys/select.h>
#  include <sys/socket.h>
#  include <sys/uio.h>
#  include <sys/un.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <net/if.h>
#  include <limits.h>
#
#elif defined( ZI_OS_WINDOWS ) || defined( ZI_OS_CYGWIN )
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  include <mswsock.h>
#  pragma comment(lib, "ws2_32.lib")
#  pragma comment(lib, "mswsock.lib")
#
#else
#  error "os not supported"
#
#endif

namespace zi {
namespace net {
namespace detail {

#if defined( ZI_OS_LINUX )

typedef int              socket_type;
typedef sockaddr         socket_addr_type;
typedef in_addr          in_addr_type;
typedef ip_mreq          ip_mreq_type;
typedef sockaddr_in      sockaddr_type;
typedef sockaddr_storage sockaddr_storage_type;
typedef sockaddr_un      sockaddr_un_type;
typedef addrinfo         addrinfo_type;
typedef int              ioctl_arg_type;


const int invalid_socket       = -1;
const int socket_error_retval  = -1;
const int max_addr_str_len     = INET_ADDRSTRLEN;
const int shutdown_receive     = SHUT_RD;
const int shutdown_send        = SHUT_WR;
const int shutdown_both        = SHUT_RDWR;
const int message_peek         = MSG_PEEK;
const int message_out_of_band  = MSG_OOB;
const int message_do_not_route = MSG_DONTROUTE;

#elif defined( ZI_OS_WINDOWS ) || defined( ZI_OS_CYGWIN )

typedef SOCKET           socket_type;
typedef sockaddr         socket_addr_type;
typedef in_addr          in_addr_type;
typedef ip_mreq          ip_mreq_type;
typedef sockaddr_in      sockaddr_type;
typedef sockaddr_storage sockaddr_storage_type;
typedef sockaddr_un      sockaddr_un_type;
typedef addrinfo         addrinfo_type;

const int invalid_socket       = INVALID_SOCKET;
const int socket_error_retval  = SOCKET_ERROR;
const int max_addr_str_len     = 256;
const int shutdown_receive     = SD_RECEIVE;
const int shutdown_send        = SD_SEND;
const int shutdown_both        = SD_BOTH;
const int message_peek         = MSG_PEEK;
const int message_out_of_band  = MSG_OOB;
const int message_do_not_route = MSG_DONTROUTE;

#endif

} // namespace detail
} // namespace net
} // namespace zi

#endif
