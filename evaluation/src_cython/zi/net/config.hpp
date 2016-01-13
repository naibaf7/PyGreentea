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

#ifndef ZI_NET_CONFIG_HPP
#define ZI_NET_CONFIG_HPP 1
#
#include <zi/config/config.hpp>
#
# // Windows specific stuff
#if defined( ZI_OS_WINDOWS ) || defined( ZI_OS_CYGWIN )
#
#  // winnt version
#  if !defined( _WIN32_WINNT )
#    if defined( ZI_CXX_MSVC )
#      pragma message( "Using default _WIN32_WINNT=0x501 ( Win XP )" )
#    else
#      warning Using default _WIN32_WINNT=0x501 ( Win XP )
#    endif
#    define _WIN32_WINNT 0x501
#  endif
#
#  // for winsock2.h
#  if defined( ZI_CXX_MSVC ) && defined( _WIN32 ) && !defined( WIN32 )
#    if defined( _WINSOCK2API_ )
#      error Define WIN32 before including winsock2.h
#    else
#      define WIN32
#    endif
#  endif
#
#  // IO completion ports
#  if ( _WIN32_WINNT >= 0x0400 ) && !defined( UNDER_CE )
#    define ZI_NET_HAS_WIN_IOCP
#  endif
#
#  // minimal windows inculdes
#  if !defined( WIN32_LEAN_AND_MEAN )
#    define WIN32_LEAN_AND_MEAN
#  endif
#
#  // no min and max macros
#  if !defined( NOMINMAX )
#    define NOMINMAX 1
#  endif
#
#
# // Linux specific stuff
#elif defined( ZI_OS_LINUX )
#  include <linux/version.h>
#
#  // epoll
#  if LINUX_VERSION_CODE >= KERNEL_VERSION( 2,5,45 )
#    define ZI_NET_HAS_LINUX_EPOLL
#  endif
#
#
# // MacOS specific stuff
#elif defined( ZI_OS_MACOS )
#
#  // kqueue
#  define ZI_NET_HAS_BSD_KQUEUE
#
#endif
#
#endif
