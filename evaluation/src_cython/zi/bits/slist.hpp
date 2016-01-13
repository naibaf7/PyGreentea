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

#ifndef ZI_BITS_SLIST_HPP
#define ZI_BITS_SLIST_HPP 1

#include <zi/config/config.hpp>

#if defined( ZI_NO_BOOST )
#  error "Can't use bits/slist.hpp with ZI_NO_BOOST defined"
#endif

#include <boost/config.hpp>

#if defined( BOOST_NO_SLIST )
#  error "slist implementation not found"
#else
#  if defined( BOOST_SLIST_HEADER )
#    include BOOST_SLIST_HEADER
#  else
#    include <slist>
#  endif
#endif

namespace zi {

using BOOST_STD_EXTENSION_NAMESPACE::slist;

} // namespace zi

#endif
