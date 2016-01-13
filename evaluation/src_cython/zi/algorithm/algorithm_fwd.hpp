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

#ifndef ZI_ALGORITHM_ALGORITHM_FWD_HPP
#define ZI_ALGORITHM_ALGORITHM_FWD_HPP 1

#include <utility>

namespace zi {

template< class T >
const T&
absmin( const T&, const T& );

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

template< class T >
::std::pair< const T&, const T& >
minmax( const T&, const T& );

template< class T, class Compare >
::std::pair< const T&, const T& >
minmax( const T&, const T&, Compare );

template< class Iter >
::std::pair< Iter, Iter >
minmax_element( Iter, Iter );

template< class Iter, class Compare >
::std::pair< Iter, Iter >
minmax_element( Iter, Iter, Compare );

template< class T >
::std::pair< const T&, const T& >
absminmax( const T&, const T& );

template< class T, class Compare >
::std::pair< const T&, const T& >
absminmax( const T&, const T&, Compare );

template< class Iter >
::std::pair< Iter, Iter >
absminmax_element( Iter, Iter );

template< class Iter, class Compare >
::std::pair< Iter, Iter >
absminmax_element( Iter, Iter, Compare );

} // namespace zi

#endif
