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

#ifndef ZI_NET_IP_ADDRESS_HPP
#define ZI_NET_IP_ADDRESS_HPP 1

#include <zi/net/config.hpp>
#include <zi/net/detail/socket_types.hpp>

#include <zi/bits/cstdint.hpp>

#include <string>

namespace zi {
namespace net {
namespace ip {

class address
{
private:
    ::zi::net::detail::in_addr_type addr_;

public:

    address()
    {
        addr_.s_addr = 0;
    }

    address( uint32_t addr )
    {
        addr_.s_addr = ::htonl( addr );
    }

    address( const address& other )
        : addr_( other.addr_ )
    {

    }

    address& operator=( const address& other )
    {
        addr_ = other.addr_;
        return *this;
    }

    uint32_t to_uint32() const
    {
        return ::ntohl( addr_.s_addr );
    }

    std::string to_string() const
    {
        char buffer[ ::zi::net::detail::max_addr_str_len ];
        const char* addr =
            ::inet_ntop( AF_INET, &addr_, buffer, ::zi::net::detail::max_addr_str_len );
        if ( addr == 0 )
        {
            return std::string();
        }

        return addr;
    }

    static address any()
    {
        return address( static_cast< uint32_t >( INADDR_ANY ) );
    }

    static address loopback()
    {
        return address( static_cast< uint32_t >( INADDR_LOOPBACK ) );
    }

    static address broadcast()
    {
        return address( static_cast< uint32_t >( INADDR_BROADCAST ) );
    }

    friend bool operator==( const address& a1, const address& a2 )
    {
        return a1.addr_.s_addr == a2.addr_.s_addr;
    }

    friend bool operator!=( const address& a1, const address& a2 )
    {
        return a1.addr_.s_addr != a2.addr_.s_addr;
    }

    friend bool operator<( const address& a1, const address& a2 )
    {
        return a1.addr_.s_addr < a2.addr_.s_addr;
    }

};


} // namespace ip
} // namespace net
} // namespace zi

#endif

