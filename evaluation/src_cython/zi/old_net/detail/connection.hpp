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

#ifndef ZI_NET_DETAIL_CONNECTION_HPP
#define ZI_NET_DETAIL_CONNECTION_HPP 1

#include <zi/net/config.hpp>
#include <zi/net/ip_address.hpp>
#include <zi/net/detail/socket_types.hpp>

#include <zi/bits/cstdint.hpp>

#include <string>
#include <sstream>

namespace zi {
namespace net {
namespace detail {

class connection
{
private:

    union temp_union_type
    {
        socket_addr_type      addr;
        sockaddr_storage_type storage;
        sockaddr_type         v4;
    } data_;

public:

    connection(): data_()
    {
        data_.v4.sin_family      = AF_INET;
        data_.v4.sin_port        = 0;
        data_.v4.sin_addr.s_addr = INADDR_ANY;
    }

    connection( uint16_t port ): data_()
    {
        data_.v4.sin_family      = AF_INET;
        data_.v4.sin_port        = ::htons( port );
        data_.v4.sin_addr.s_addr = INADDR_ANY;
    }

    connection( const ip::address& addr, uint16_t port ): data_()
    {
        data_.v4.sin_family      = AF_INET;
        data_.v4.sin_port        = ::htons( port );
        data_.v4.sin_addr.s_addr = ::htonl( addr.to_uint32() );
    }

    connection( const connection& other ): data_( other.data_ )
    {
    }

    connection& operator=( const connection& other )
    {
        data_ = other.data_;
        return *this;
    }

    uint16_t port() const
    {
        return ::ntohs( data_.v4.sin_port );
    }

    void port( uint16_t val )
    {
        data_.v4.sin_port = ::htons( val );
    }

    ip::address address() const
    {
        return ip::address( ::ntohl( data_.v4.sin_addr.s_addr ) );
    }

    void address( const ip::address& addr )
    {
        connection tmp( addr, this->port() );
        data_ = tmp.data_;
    }

    friend bool operator==( const connection& c1, const connection& c2 )
    {
        return c1.address() == c2.address() && c1.port() == c2.port();
    }

    friend bool operator<( const connection& c1, const connection& c2 )
    {
        return ( c1.address() < c2.address() ? true :
                 ( c1.address() != c2.address() ? false :
                   ( c1.port() < c2.port() )));
    }

    std::string to_string() const
    {
        std::string addr_str = address().to_string();

        std::ostringstream oss;
        oss << addr_str << ':' << port();

        return oss.str();
    }

};


} // namespace detail
} // namespace net
} // namespace zi

#endif
