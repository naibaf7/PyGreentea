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

#ifndef ZI_NET_CONNECTION_HPP
#define ZI_NET_CONNECTION_HPP 1

#include <zi/net/config.hpp>
#include <zi/net/detail/socket_types.hpp>
#include <zi/net/detail/connection.hpp>
#include <zi/net/ip_address.hpp>

#include <zi/bits/cstdint.hpp>

#include <string>

namespace zi {
namespace net {
namespace ip {

template< class Protocol >
class connection
{
private:
    ::zi::net::detail::connection conn_;

public:
    typedef Protocol                            protocol_type;
    typedef ::zi::net::detail::socket_addr_type data_type;

    connection(): conn_()
    {
    }

    connection( const Protocol& protocol, uint16_t port_no ): conn_( port_no )
    {
    }

    connection( const address& addr, uint16_t port_no ): conn_( addr, port_no )
    {
    }



};


} // namespace sock
} // namespace net
} // namespace zi

#endif

