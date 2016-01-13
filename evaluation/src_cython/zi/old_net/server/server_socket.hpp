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

#ifndef ZI_NET_SERVER_SERVER_SOCKET_HPP
#define ZI_NET_SERVER_SERVER_SOCKET_HPP 1

#include <zi/net/detail/sock.hpp>
#include <zi/net/detail/unistd.hpp>
#include <zi/net/detail/fcntl.hpp>
#include <zi/net/detail/aio_manager.hpp>
#include <zi/net/detail/cerrno_exception.hpp>

#include <zi/net/server/connection.hpp>

#include <zi/utility/assert.hpp>
#include <zi/utility/non_copyable.hpp>

#include <zi/concurrency/mutex.hpp>
#include <zi/concurrency/monitor.hpp>
#include <zi/concurrency/condition_variable.hpp>

namespace zi {
namespace net {
namespace server {

void couter( int fd )
{
    std::cout << "CAN READ!\n";
    char buff[ 1024 ];
    int n = unistd::read( fd, buff, 2 );
    if ( n <= 0 )
    {
        std::cout << "DAMN!!!\n";
        return;
    }
    std::string xxx( buff, n );
    std::cout << xxx;
}

struct char_buffer
{
    int len;
    char* buff;
};

class server_socket: non_copyable
{
private:

    static const int accept_back_log = 1024;

    int                fd_     ;
    int                port_   ;
    aio_manager        poll_   ;

    mutex              mutex_     ;
    condition_variable waiting_cv_;
    condition_variable sending_cv_;

    void accept()
    {

        sock::sockaddr_in sin;
        sock::socklen_t   slen = sizeof( sin );

        int fd = sock::accept( fd_, reinterpret_cast< sock::sockaddr* >( &sin ), &slen );
        if ( fd < 0 )
        {
            ZI_NET_THROW_CERRNO();
        }

        new connection( fd, bind( &couter, fd ), bind( &couter, fd ) );

    }

public:

    server_socket( int port )
        : fd_( -1 ),
          port_( port ),
          poll_(),
          mutex_(),
          waiting_cv_(),
          sending_cv_()
    {
    }

    ~server_socket()
    {
        close();
    }

    bool send( char* b, int sz )
    {
        mutex::guard g( mutex_ );

        return 0;
    }

    bool is_open()
    {
        return fd_ >= 0;
    }

    void close()
    {
        if ( fd_ >= 0 )
        {
            unistd::close( fd_ );
        }
    }

    void stop()
    {
        poll_.stop();
    }

    void listen()
    {
        ZI_VERIFY( port_ > 0 && port_ < 0xFFFF );

        sock::sockaddr_in sin;
        sin.sin_family = AF_INET;
        sin.sin_port   = sock::htons( port_ );

        fd_ = sock::socket( AF_INET, SOCK_STREAM, 0 );
        if ( fd_ < 0 )
        {
            ZI_NET_THROW_CERRNO();
        }

        int tmp = 1;
        sock::setsockopt( fd_, SOL_SOCKET, SO_REUSEADDR, &tmp, sizeof( int ) );
        sock::setsockopt( fd_, IPPROTO_TCP, TCP_NODELAY, &tmp, sizeof( int ) );

        sock::linger l;
        sock::setsockopt( fd_, SOL_SOCKET, SO_LINGER, &l, sizeof( l ) );

        if ( sock::bind( fd_, reinterpret_cast< sock::sockaddr* >( &sin ),
                         sizeof( sin ) ) < 0 )
        {
            ZI_NET_THROW_CERRNO();
        }

        if ( sock::listen( fd_, accept_back_log ) < 0 )
        {
            ZI_NET_THROW_CERRNO();
        }

        poll_.add< aio::READ >( fd_, bind( &server_socket::accept, this ) );
        poll_.run();
    }

};

} // namespace server
} // namespace net
} // namespace zi

#endif

