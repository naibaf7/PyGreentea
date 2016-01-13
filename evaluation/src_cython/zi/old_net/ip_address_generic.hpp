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

namespace zi {
namespace net {
namespace ip {

class address
{
private:

    enum address_type
    {
        v4 = 4,
        v6 = 6
    };

    address_type type_;

    v4_address v4_address_;
    v6_address v6_address_;

public:

    address()
        : type_( v4 ), v4_address_(), v6_address_()
    {
    }

    address( const v4_address& addr )
        : type_( v4 ), v4_address_( addr ), v6_address_()
    {
    }

    address( const v6_address& addr )
        : type_( v6 ), v4_address_(), v6_address_( addr )
    {
    }

    address( const address& addr )
        : type_( addr.type_ ),
          v4_address_( addr.v4_address_ ),
          v6_address_( addr.v6_address_ )
    {
    }

    address& operator=( const v4_address& addr )
    {
        type_ = v4;
        v4_address_ = addr;
        v6_address_ = v6_address();
        return *this;
    }

    address& operator=( const v6_address& addr )
    {
        type_ = v6;
        v4_address_ = v4_address();
        v6_address_ = addr;
        return *this;
    }

    address& operator=( const address& addr )
    {
        type_       = addr.type_;
        v4_address_ = addr.v4_address_;
        v6_address_ = addr.v6_address_;
        return *this;
    }

}


} // namespace sock
} // namespace net
} // namespace zi

#endif

