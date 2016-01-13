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

#ifndef ZI_NET_TCP_HPP
#define ZI_NET_TCP_HPP 1

#include <zi/net/config.hpp>
#include <zi/net/detail/socket_types.hpp>

namespace zi {
namespace net {
namespace ip {

class tcp
{
private:
    int family_;

    explicit tcp( int family ): family_( family )
    {
    }

public:

    static tcp v4()
    {
        return tcp( PF_INET );
    }

    static tcp v6()
    {
        return tcp( PF_INET6 );
    }

    int type() const
    {
        return SOCK_STREAM;
    }

    int protocol() const
    {
        return IPPROTO_TCP;
    }

    int family() const
    {
        return family_;
    }

    friend bool operator==( const tcp& tcp1, const tcp& tcp2 )
    {
        return tcp1.family_ == tcp2.family_;
    }

    friend bool operator!=( const tcp& tcp1, const tcp& tcp2 )
    {
        return tcp1.family_ != tcp2.family_;
    }

};


} // namespace ip
} // namespace net
} // namespace zi

#endif
