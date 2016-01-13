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

#ifndef ZI_WEB_SERVER_SOCKET_HPP
#define ZI_WEB_SERVER_SOCKET_HPP 1

#include <zi/web/server/detail/sys.hpp>
#include <zi/web/server/cerrno_exception.hpp>

#include <zi/bits/cstdint.hpp>
#include <zi/time/interval.hpp>
#include <zi/time/now.hpp>
#include <zi/time/time_utils.hpp>
#include <zi/utility/non_copyable.hpp>

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

namespace zi {
namespace web {
namespace server {

class socket: non_copyable
{
private:

    int fd_ ;

    mutable std::string host_;
    mutable std::string addr_;
    mutable int port_        ;

    socket( int fd )
        : fd_( fd ),
          host_( "" ),
          addr_( "" ),
          port_( -1 )
    {
    }

    ~socket()
    {
        close();
    }

    void close()
    {
        if ( is_open() )
        {
            detail::shutdown( fd_, SHUT_RDWR );
            detail::close( fd_ );
        }
        fd_ = -1;
    }

    bool is_open() const
    {
        return fd_ >= 0;
    }

    int fd() const
    {
        return fd_;
    }

    std::size_t read( uint8_t* buf, std::size_t len )
    {
        if ( !is_open() )
        {
            ZI_THROW_ERRNO_MSG( "read non open socket" );
        }

        int ret = detail::read( fd_, reinterpret_cast< char* >( buf ), len );

        if ( ret < 0 )
        {
            if ( errno != EAGAIN )
            {
                close();
            }
            ZI_THROW_ERRNO_MSG( "couldn't read from the socket" );
        }

        return static_cast< std::size_t >( ret );
    }

    void write( const uint8_t* buffer, std::size_t len )
    {

        ZI_VERIFY( len > 0 );

        if ( !is_open() )
        {
            ZI_THROW_ERRNO_MSG( "read non open socket" );
        }

        std::size_t len_done = 0;

        while ( len_done < len )
        {
            int ret = detail::write( fd_, buffer + len_done, len - len_done );

            if ( ret < 0 )
            {
                close();
                ZI_THROW_ERRNO_MSG( "writting to socket failed" );
            }

            if ( ret == 0 )
            {
                ZI_THROW_ERRNO_MSG( "write returned 0" );
            }

            len_done += static_cast< std::size_t >( ret );
        }
    }

    std::string addr() const
    {
        if ( addr_.empty() )
        {

            detail::sockaddr_storage stor;
            detail::socklen_t len = sizeof( detail::sockaddr_storage );

            if ( !is_open() )
            {
                return addr_;
            }

            if ( detail::getpeername( fd_, reinterpret_cast< detail::sockaddr* >( &stor ), &len) )
            {
                return addr_;
            }

            char rhost[ NI_MAXHOST ];
            char rport[ NI_MAXSERV ];

            detail::getnameinfo( reinterpret_cast< detail::sockaddr* >( &stor ), len,
                                 rhost, NI_MAXHOST, rport, NI_MAXSERV,
                                 NI_NUMERICHOST | NI_NUMERICSERV );

            addr_ = rhost;
            port_ = std::atoi( rport );
        }

        return addr_;
    }

    int port() const
    {
        (void) addr();
        return port_;
    }

    std::string host() const
    {
        if ( host_.empty() )
        {

            detail::sockaddr_storage stor;
            detail::socklen_t len = sizeof( detail::sockaddr_storage );

            if ( !is_open() )
            {
                return host_;
            }

            if ( detail::getpeername( fd_, reinterpret_cast< detail::sockaddr* >( &stor ), &len) )
            {
                return host_;
            }

            char rhost[ NI_MAXHOST ];
            char rport[ NI_MAXSERV ];

            detail::getnameinfo( reinterpret_cast< detail::sockaddr* >( &stor ), len,
                                 rhost, NI_MAXHOST, rport, NI_MAXSERV, 0 );

            host_ = rhost;
        }

        return host_;
    }

public:

    template< int64_t I, int64_t J >
    static shared_ptr< socket > connect( const std::string& host, const int port,
                                         const interval::detail::interval_tpl< I > &send_to,
                                         const interval::detail::interval_tpl< J > &recv_to )
    {

        if ( port < 0 || port > 0xFFFF )
        {
            ZI_THROW_ERR_MSG( EINVAL, "bad port" );
        }

        int fd = detail::socket( AF_INET, SOCK_STREAM, 0 );

        if ( fd == -1 )
        {
            ZI_THROW_ERRNO_MSG( "unable to create socket handle" );
        }

        int tmp = 1;
        detail::setsockopt( fd, IPPROTO_TCP, TCP_NODELAY, &tmp, sizeof(int) );

        detail::sockaddr_in sock;
        std::memset( &sock, 0, sizeof( detail::sockaddr_in ) );

        detail::in_addr_t addr = detail::inet_addr( host.c_str() );
        char port_c_str[ 10 ];
        sprintf( port_c_str, "%d", port );

        if ( addr != INADDR_NONE )
        {
            sock.sin_addr.s_addr = addr;
        }
        else
        {
            detail::hostent *h = detail::gethostbyname( host.c_str() );
            ZI_VERIFY( h && h->h_length == 4 );
            sock.sin_addr.s_addr = ( reinterpret_cast< detail::in_addr* >( h->h_addr ) )->s_addr;
        }

        sock.sin_port = detail::htons( port );

        if ( detail::connect( fd, reinterpret_cast< detail::sockaddr* >( &sock ),
                              sizeof( detail::sockaddr_in ) ) )
        {
            detail::close( fd );
            ZI_THROW_ERRNO_MSG( "couldn't connect to dest" );
        }

        if ( send_to.msecs() > 0 )
        {
            timeval tv;
            time_utils::msec_to_tv( &tv, send_to.msecs() );
            if ( detail::setsockopt( fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof( tv ) ) )
            {
                detail::close( fd );
                ZI_THROW_ERRNO_MSG( "couldn't set send timeout" );
            }
        }

        if ( recv_to.msecs() > 0 )
        {
            timeval tv;
            time_utils::msec_to_tv( &tv, send_to.msecs() );
            if ( detail::setsockopt( fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof( tv ) ) )
            {
                detail::close( fd );
                ZI_THROW_ERRNO_MSG( "couldn't set send timeout" );
            }
        }

        return shared_ptr< socket >( new socket( fd, recv_to.msecs() ) );

    }

};

} // namespace server
} // namespace web
} // namespace zi

#endif

