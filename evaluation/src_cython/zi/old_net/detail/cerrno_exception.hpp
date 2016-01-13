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

#ifndef ZI_NET_DETAIL_CERRNO_EXCEPTION_HPP
#define ZI_NET_DETAIL_CERRNO_EXCEPTION_HPP 1

#include <cerrno>
#include <cstring>
#include <exception>
#include <string>
#include <sstream>

namespace zi {
namespace net {

class cerrno_exception: public std::exception
{
protected:
    const int         cerrno_ ;
    const std::string message_;

public:
    cerrno_exception( const int err ):
        cerrno_( err ),
        message_( error_string() )
    {
    }

    cerrno_exception( const int err, const std::string& message ):
        cerrno_( err ),
        message_( error_string() + message )
    {
    }

    virtual ~cerrno_exception() throw()
    {
    }

    virtual const char* what() const throw()
    {
        if ( message_.empty() )
        {
            return "default cerrno_exception";
        }
        else
        {
            return message_.c_str();
        }
    }

    int cerrno() const
    {
        return cerrno_;
    }

    std::string error_string() const
    {
        std::ostringstream oss;
        oss << "cerrno: " << cerrno_ << " ( " << strerror( cerrno_ ) << " ) ";
        return oss.str();
    }

};

} // namespace net
} // namespace zi

#define ZI_NET_CERRNO_EXCEPTION_STRINIGIFY_H( what ) #what
#define ZI_NET_CERRNO_EXCEPTION_STRINIGIFY( what )      \
    ZI_NET_CERRNO_EXCEPTION_STRINIGIFY_H( what )

#define ZI_NET_THROW_CERRNO()                                   \
    throw ::zi::net::cerrno_exception                           \
    ( errno, std::string( " [" ) + __FILE__ + ": " +            \
      ZI_NET_CERRNO_EXCEPTION_STRINIGIFY( __LINE__ ) + "]" )

#define ZI_NET_THROW_ERR( err )                                 \
    throw ::zi::net::cerrno_exception                           \
    ( err, std::String( " [" ) + __FILE__ + ": " +              \
      ZI_NET_CERRNO_EXCEPTION_STRINIGIFY( __LINE__ ) + "]" )

#define ZI_NET_THROW_CERRNO_MSG( message )                      \
    throw ::zi::net::cerrno_exception                           \
    ( errno, std::string( message ) +                           \
      " [" + __FILE__ + ": " +                                  \
      ZI_NET_CERRNO_EXCEPTION_STRINIGIFY( __LINE__ ) + "]" )

#define ZI_NET_THROW_ERR_MSG( err, message )                    \
    throw ::zi::net::cerrno_exception                           \
    ( err, std::string( message ) +                             \
      " [" + __FILE__ + ": " +                                  \
      ZI_NET_CERRNO_EXCEPTION_STRINIGIFY( __LINE__ ) + "]" )

#endif
