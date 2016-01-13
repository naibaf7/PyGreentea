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

#ifndef ZI_ALGORITHM_ALGO_HPP
#define ZI_ALGORITHM_ALGO_HPP 1

#include <zi/algorithm/algorithm_fwd.hpp>
#include <algorithm>

namespace zi {

template< class T >
::std::pair< const T&, const T& >
minmax( const T& e1, const T& e2 )
{
    if ( e1 < e2 )
    {
        return ::std::pair< const T&, const T& >( e1, e2 );
    }
    else
    {
        return ::std::pair< const T&, const T& >( e2, e1 );
    }
}

template< class T, class Compare >
::std::pair< const T&, const T& >
minmax( const T& e1, const T& e2, Compare cmp )
{
    if ( cmp( e1, e2 ) )
    {
        return ::std::pair< const T&, const T& >( e1, e2 );
    }
    else
    {
        return ::std::pair< const T&, const T& >( e2, e1 );
    }
}

template< class Iter >
::std::pair< Iter, Iter >
minmax_element( Iter i1, Iter i2 )
{
    return ::std::pair< Iter, Iter >( ::std::min_element( i1, i2 ),
                                      ::std::max_element( i1, i2 ) );
}

template< class Iter, class Compare >
::std::pair< Iter, Iter >
minmax_element( Iter i1, Iter i2, Compare cmp )
{
    return ::std::pair< Iter, Iter >( ::std::min_element( i1, i2, cmp ),
                                      ::std::max_element( i1, i2, cmp ) );
}



template< class T >
::std::pair< const T&, const T& >
absminmax( const T& e1, const T& e2)
{
    if ( e1 < 0 )
    {

    }
}

template< class T, class Compare >
::std::pair< const T&, const T& >
absminmax( const T&, const T&, Compare );



template< class T >
const T&
absmin( const T& e1, const T& e2)
{
    if ( e1 < 0 )
    {
        if ( e2 < 0 )
        {
            return std::max( e1, e2 );
        }
        return e1;
    }

    if ( e2 < 0 )
    {
        return e2;
    }

    return std::min( e1, e2 );
}

template< class T, class Compare >
const T&
absmin( const T&, const T&, Compare );

template< class T >
const T&
absmax( const T&, const T& );

template< class T, class Compare >
const T&
absmax( const T&, const T&, Compare );

template< class Iter >
Iter
absmin_element( Iter, Iter );

template< class Iter, class Compare >
Iter
absmin_element( Iter, Iter, Compare );

template< class Iter >
Iter
absmax_element( Iter, Iter );

template< class Iter, class Compare >
Iter
absmax_element( Iter, Iter, Compare );

template< class Iter >
::std::pair< Iter, Iter >
absminmax_element( Iter, Iter );

template< class Iter, class Compare >
::std::pair< Iter, Iter >
absminmax_element( Iter, Iter, Compare );

} // namespace zi

#endif
