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

#ifndef ZI_NET_UDP_HPP
#define ZI_NET_UDP_HPP 1

#include <zi/net/config.hpp>
#include <zi/net/detail/socket_types.hpp>

namespace zi {
namespace net {
namespace ip {

class udp
{
private:
    int family_;

    explicit udp( int family ): family_( family )
    {
    }

public:

    static udp v4()
    {
        return udp( PF_INET );
    }

    static udp v6()
    {
        return udp( PF_INET6 );
    }

    int type() const
    {
        return SOCK_STREAM;
    }

    int protocol() const
    {
        return IPPROTO_UDP;
    }

    int family() const
    {
        return family_;
    }

    friend bool operator==( const udp& udp1, const udp& udp2 )
    {
        return udp1.family_ == udp2.family_;
    }

    friend bool operator!=( const udp& udp1, const udp& udp2 )
    {
        return udp1.family_ != udp2.family_;
    }

};


} // namespace ip
} // namespace net
} // namespace zi

#endif
