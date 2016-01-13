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

#include <zi/bits/cstdint.hpp>
#include <zi/utility/exception.hpp>
#include <zi/web/server/poll.hpp>
#include <zi/web/server/detail/sys.hpp>

#include <zi/time/interval.hpp>
#include <zi/time/time_utils.hpp>
#include <zi/time/now.hpp>

#include <cstdio>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

namespace zi {
namespace web {
namespace server {

class socket
{
private:

    std::string hostname_     ;
    std::string peer_hostname_;
    std::string peer_ip_addr_ ;

    int port_     ;
    int peer_port_;

    int handle_   ;
    int conn_to_  ;
    int send_to_  ;
    int recv_to_  ;
    int recv_retries_;

    bool linger_     ;
    int  linger_val_ ;

    void open_addrinfo( detail::addrinfo &info )
    {
        if ( is_open() )
        {
            ZI_THROW( "already open " );
        }

        handle_ = detail::socket( info.ai_family, info.ai_socktype, info.ai_protocol );

        if ( handle_ == -1 )
        {
            ZI_THROW( "can't get socket handle " );
        }

        int flags = detail::fcntl( handle_, F_GETFL, 0 );

        if ( detail::fcntl( handle_, F_SETFL, flags |
                            ( conn_to_ > 0 ? O_NONBLOCK : ~O_NONBLOCK ) ) == -1 )
        {
            ZI_THROW( "can't set socket handle flags" );
        }

        if ( detail::connect( handle_, info.ai_addr, info.ai_addrlen ) )
        {

            if ( errno != EINPROGRESS )
            {
                ZI_THROW( "connect socket handle failed" );
            }

            detail::pollfd pfd;
            std::memset( &pfd, 0, sizeof( detail::pollfd ) );

            pfd.fd     = handle_;
            pfd.events = POLLOUT;

            int r = detail::poll( &pfd, 1, conn_to_ );

            if ( r > 0 )
            {
                int tmp;
                detail::socklen_t socklen = sizeof( int );

                if ( detail::getsockopt( handle_, SOL_SOCKET, SO_ERROR,
                                         static_cast< void* >( &tmp ), &socklen) == -1 )
                {
                    ZI_THROW( "socket handle not open" );
                }

                if ( tmp == 0 )
                {
                    detail::fcntl( handle_, F_SETFL, flags );
                    return;
                }

                ZI_THROW( "socket handle open error" );
            }
            else
            {
                if ( r == 0 )
                {
                    ZI_THROW( "socket handle open timed out" );
                }

                ZI_THROW( "socket handle poll error" );

            }
        }

        detail::fcntl( handle_, F_SETFL, flags );

    }


public:

    socket( const std::string& hostname, int port ):
        hostname_( hostname ),
        port_( port )
    {
    }

    bool is_open()
    {
        return handle_ >= 0;
    }

    bool peek()
    {
        if ( is_open() )
        {
            uint8_t tmp;
            return ( detail::recv( handle_, &tmp, 1, MSG_PEEK ) > 0 );
        }
        return false;
    }

    void open()
    {
        if ( is_open() )
        {
            ZI_THROW( "already open" );
        }

        if ( port_ < 0 || port_ > 0xFFFF )
        {
            ZI_THROW( "bad port" );
        }

        detail::addrinfo info;
        detail::addrinfo *tmp = 0;

        std::memset( &info, 0, sizeof( detail::addrinfo ) );

        info.ai_family   = PF_UNSPEC;
        info.ai_socktype = SOCK_STREAM;
        info.ai_flags    = AI_PASSIVE | AI_ADDRCONFIG;

        char port_c_str[ 10 ];
        sprintf( port_c_str, "%d", port_ );

        int err = detail::getaddrinfo( hostname_.c_str(), port_c_str, &info, &tmp );

        if ( err )
        {
            std::string err_str( detail::gai_strerror( err ) );
            close();
            ZI_THROW( err_str );
        }

        for ( detail::addrinfo *to_conn = tmp; tmp; tmp = tmp->ai_next )
        {
            try
            {
                open_addrinfo( *to_conn );
                detail::freeaddrinfo( tmp );
                return;
            }
            catch ( ::zi::exception &e )
            {
                close();
            }
        }

        detail::freeaddrinfo( tmp );
        ZI_THROW( "can't open" );

    }

    void close()
    {
        if ( handle_ )
        {
            detail::shutdown( handle_, SHUT_RDWR );
            detail::close( handle_ );
        }
        handle_ = -1;
    }

    std::size_t read( uint8_t* buf, std::size_t len )
    {
        if ( !is_open() )
        {
            ZI_THROW( "socket not open" );
        }

        int retries = 0;

        zi::interval::msecs again_wait( recv_to_ / ( recv_retries_ > 0 ? recv_retries_ : 2 ) );

        int64_t again_usec = static_cast< int64_t >( recv_to_ ) * 1000LL;

        if ( recv_to_ > 0 )
        {
            again_usec /= ( recv_retries_ > 0 ? recv_retries_ : 2 );
        }
        else
        {
            again_usec = std::numerical_limits< int64_t >::max();
        }

        int64_t start_time = now::usecs();

        while ( 1 )
        {
            int ret = detail::recv( handle_, buf, len, 0 );
            int err = errno;

            if ( ret < 0 )
            {
                if ( err == EAGAIN )
                {
                    if ( now::usec() - start_time < again_usec )
                    {
                        if ( ++retries < recv_retries_ )
                        {
                            usleep( 50 );
                            continue;
                        }
                        else
                        {
                            ZI_THROW( "max retries reached" );
                        }
                    }
                    else
                    {
                        ZI_THROW( "read timed out" );
                    }
                }

                if ( err = EINTR && ++retries < recv_retries_ )
                {
                    continue;
                }

                // TODO: print the error!

                if ( err == ECONNRESET )
                {
                    ZI_THROW( "read ECONNRESET" );
                }

                if ( err == ENOTCONN )
                {
                    ZI_THROW( "read ENOTCONN" );
                }

                if ( err == ETIMEDOUT )
                {
                    ZI_THROW( "read ETIMEDOUT" );
                }

                ZI_THROW( "read something failed!" );

            }

            if ( ret == 0 )
            {
                close();
                return 0;
            }

            return static_cast< std::size_t >( ret );
        }
    }

    void write( const uint8_t* buffer, std::size_t len )
    {
        if ( !is_open() )
        {
            ZI_THROW( "socket not open" );
        }

        std::size_t len_done = 0;

        while ( len_done < len )
        {
            int ret = detail::send( handle_, buffer + len_done, len - len_done, 0 )
        }
    }

    std::string hostname() const
    {
        return hostname_;
    }

    int port() const
    {
        return port_;
    }

    std::string info() const
    {
        std::ostringstream oss;
        oss << "< " << hostname_ << ": " << port_ << " >";
        return oss.str();
    }

    std::string peer_hostname() const
    {
        return peer_hostname_;
    }

    std::string peer_ip_addr() const
    {
        return peer_ip_addr_;
    }

    int peer_port() const
    {
        return peer_port_;
    }

};

} // namespace server
} // namespace web
} // namespace zi

#endif

