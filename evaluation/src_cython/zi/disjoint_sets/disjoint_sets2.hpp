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

#ifndef ZI_DISJOINT_SETS_DISJOINT_SETS2_HPP
#define ZI_DISJOINT_SETS_DISJOINT_SETS2_HPP 1

#include <zi/config/config.hpp>
#include <zi/bits/type_traits.hpp>
#include <zi/utility/enable_if.hpp>
#include <zi/utility/assert.hpp>
#include <zi/utility/detail/empty_type.hpp>

#include <cstddef>
#include <cassert>

namespace zi {

template< class T >
class disjoint_sets2:
        enable_if< is_integral< T >::value, detail::empty_type >::type
{

private:
    struct node
    {
        T r, p, v;

        node(): r(0), p(0)
        {
        }
    };

    node  *x_;
    T     size_ ;
    T     sets_ ;

    inline T find_set_( T id ) const
    {
        ZI_ASSERT( id < size_ );
        T n( id ), x;

        while ( n != x_[ n ].p )
        {
            n = x_[ n ].p;
        }

        while ( n != id )
        {
            x = x_[ id ].p;
            x_[ id ].p = n;
            id = x;
        }

        return n;
    }


public:

    disjoint_sets2( const T& s ): size_( s ), sets_( s )
    {
        ZI_ASSERT( s >= 0 );
        x_ = new node[ s ];
        for ( T i = 0; i < s; ++i )
        {
            x_[ i ].p = i;
            x_[ i ].v = i;
            x_[ i ].r = 0;
        }
    }

    ~disjoint_sets2()
    {
        delete [] x_;
    }

    inline T find_set( T id ) const
    {
        return x_[ find_set_( id ) ].v;
    }

    inline T join( T ox, T oy )
    {
        ZI_ASSERT( ox < size_ && ox >= 0 );
        ZI_ASSERT( oy < size_ && oy >= 0 );

        T x = find_set_( ox );
        T y = find_set_( oy );

        if ( x == y )
        {
            return x_[ x ].v;
        }

        --sets_;

        if ( x_[ x ].r >= x_[ y ].r )
        {
            x_[ y ].p = x;
            if ( x_[ x ].r == x_[ y ].r )
            {
                ++x_[x].r;
            }
            x_[ x ].v = oy;
            return oy;
        }

        x_[ x ].p = y;
        x_[ y ].v = oy;
        return oy;
    }

    inline void clear()
    {
        for ( T i = 0; i < size_; ++i )
        {
            x_[ i ].p = i;
            x_[ i ].v = i;
            x_[ i ].r = 0;
        }
        sets_ = size_;
    }

    T size() const
    {
        return size_;
    }

    T set_count() const
    {
        return sets_;
    }

};

} // namespace zi

#endif
