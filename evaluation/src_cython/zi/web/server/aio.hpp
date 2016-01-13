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

#ifndef ZI_WEB_SERVER_AIO_HPP
#define ZI_WEB_SERVER_AIO_HPP 1

#include <zi/web/server/detail/sys.hpp>
#include <zi/concurrency/spinlock.hpp>
#include <zi/utility/assert.hpp>
#include <zi/utility/non_copyable.hpp>
#include <zi/utility/enable_if.hpp>
#include <zi/bits/function.hpp>

#include <list>
#include <set>

namespace zi {
namespace web {
namespace server {

class aio: non_copyable
{
public:

    static const int NONE   =  00;
    static const int RDONLY =  01;
    static const int WRONLY =  02;
    static const int RDWR   =  03;
    static const int MASK   = ~03;

private:

    spinlock       lock_     ;
    int            high_fd_  ;
    std::set<int>  fds_      ;
    int            pipe_[2]  ;
    detail::fd_set read_fds_ ;
    detail::fd_set write_fds_;

public:

    aio(): lock_(), high_fd_( 0 ), fds_()
    {
        spinlock::guard g( lock_ );

        FD_ZERO( &read_fds_  );
        FD_ZERO( &write_fds_ );

        ZI_VERIFY_0( detail::pipe( pipe_ ) );

        fds_.insert( pipe_[ 0 ] );
        FD_SET( pipe_[ 0 ], &read_fds_ );
        high_fd_ = pipe_[ 0 ];

        int flags = detail::fcntl( high_fd_, F_GETFL, 0 ) | O_NONBLOCK;
        detail::fcntl( high_fd_, F_SETFL, flags );
    }


    template< int Flag >
    void add( int fd, typename enable_if< Flag ? true : false >::type* = 0 )
    {
        spinlock::guard g( lock_ );

        if ( high_fd_ < fd )
        {
            high_fd_ = fd;
        }

        if ( Flag & RDONLY )
        {
            FD_SET( fd, &read_fds_ );
        }

        if ( Flag & WRONLY )
        {
            FD_SET( fd, &write_fds_ );
        }

        fds_.insert( fd );
        char tmp = 1;
        ZI_VERIFY( detail::write( pipe_[ 1 ], &tmp, 1 ) == 1 );

    }

    template< int Flag >
    bool contains( int fd, typename enable_if< Flag ? true : false >::type* = 0 ) const
    {
        spinlock::guard g( lock_ );

        bool ok = false;

        if ( Flag & RDONLY )
        {
            ok |= FD_ISSET( fd, &read_fds_ );
        }


        if ( Flag & WRONLY )
        {
            ok |= FD_ISSET( fd, &write_fds_ );
        }

        return ok;
    }

    template< int Flag >
    bool remove( int fd )
    {
        spinlock::guard g( lock_ );

        if ( Flag & RDONLY )
        {
            FD_CLR( fd, &read_fds_ );
        }

        if ( Flag & WRONLY )
        {
            FD_CLR( fd, &write_fds_ );
        }

        if ( !contains< RDWR >( fd ) && fd == high_fd_ )
        {
            fds_.erase( fd );
            high_fd_ = *fds_.rbegin();
        }

        char tmp = 1;
        ZI_VERIFY( detail::write( pipe_[ 1 ], &tmp, 1 ) == 1 );

        return ( !this->template contains< Flag >( fd ) );
    }

    bool wait( std::list< int > &readable, std::list< int > &writable )
    {
        detail::fd_set read_fds, write_fds;
        int high_fd;
        bool ok = false;

        {
            spinlock::guard g( lock_ );
            read_fds  = read_fds_ ;
            write_fds = write_fds_;
            high_fd   = high_fd_  ;
        }

        if ( detail::select( high_fd + 1, &read_fds, &write_fds, 0, 0 ) < 0 )
        {
            ZI_VERIFY( errno == EINTR );
            return false;
        }

        for ( int fd = 0; fd <= high_fd; ++fd )
        {

            if ( FD_ISSET( fd, &read_fds ) )
            {
                if ( fd == pipe_[ 0 ] )
                {
                    char tmp;
                    ZI_VERIFY( detail::read( fd, &tmp, 1 ) == 1 );
                    ZI_VERIFY( tmp == 1 );
                }
                else
                {
                    readable.push_back( fd );
                    ok = true;
                }
            }

            if ( FD_ISSET( fd, &write_fds ) )
            {
                writable.push_back( fd );
                ok = true;
            }
        }

        return ok;
    }

    void interrupt() const
    {
        spinlock::guard g( lock_ );
        char tmp = 1;
        ZI_VERIFY( detail::write( pipe_[ 1 ], &tmp, 1 ) == 1 );
    }

};


} // namespace server
} // namespace web
} // namespace zi

#endif

