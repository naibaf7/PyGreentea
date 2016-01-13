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

#ifndef ZI_NET_DETAIL_AIO_HPP
#define ZI_NET_DETAIL_AIO_HPP 1

#include <cerrno>

#include <zi/net/detail/select.hpp>
#include <zi/net/detail/unistd.hpp>
#include <zi/net/detail/fcntl.hpp>

#include <zi/utility/assert.hpp>
#include <zi/utility/non_copyable.hpp>
#include <zi/utility/enable_if.hpp>

#include <list>
#include <set>

namespace zi {
namespace net {

class aio: non_copyable
{
public:

    static const int READ  =  01;
    static const int WRITE =  02;
    static const int RDWR  =  03;

private:

    int            high_fd_  ;
    std::set<int>  fds_      ;
    select::fd_set rd_fds_   ;
    select::fd_set wr_fds_   ;
    unistd::pipe   pipe_     ;

public:

    aio(): high_fd_( 0 ), fds_(), rd_fds_(), wr_fds_(), pipe_()
    {
        fds_.insert( pipe_.in() );
        select::set( pipe_.in(), rd_fds_ );
        high_fd_ = pipe_.in();

        int flags = fcntl::fcntl( high_fd_, F_GETFL, 0 ) | O_NONBLOCK;
        fcntl::fcntl( high_fd_, F_SETFL, flags );
    }

    ~aio()
    {
    }

    template< int Flag >
    void add( int fd, typename enable_if< Flag ? true : false >::type* = 0 )
    {
        if ( high_fd_ < fd )
        {
            high_fd_ = fd;
        }

        if ( Flag & READ )
        {
            select::set( fd, rd_fds_ );
        }

        if ( Flag & WRITE )
        {
            select::set( fd, wr_fds_ );
        }

        fds_.insert( fd );
        char tmp = 1;
        ZI_VERIFY( unistd::write( pipe_.out(), &tmp, 1 ) == 1 );
    }

    template< int Flag >
    bool contains( int fd, typename enable_if< Flag ? true : false >::type* = 0 ) const
    {
        bool ok = false;

        if ( Flag & READ )
        {
            ok |= select::is_set( fd, rd_fds_ );
        }


        if ( Flag & WRITE )
        {
            ok |= select::is_set( fd, wr_fds_ );
        }

        return ok;
    }

    template< int Flag >
    bool remove( int fd )
    {
        if ( Flag & READ )
        {
            select::clear( fd, rd_fds_ );
        }

        if ( Flag & WRITE )
        {
            select::clear( fd, wr_fds_ );
        }

        if ( !contains< RDWR >( fd ) && fd == high_fd_ )
        {
            fds_.erase( fd );
            high_fd_ = *fds_.rbegin();
        }

        char tmp = 1;
        ZI_VERIFY( unistd::write( pipe_.out(), &tmp, 1 ) == 1 );

        return ( !this->template contains< Flag >( fd ) );
    }

    bool wait( std::list< int > &readable, std::list< int > &writable )
    {
        select::fd_set rd_fds, wr_fds;
        int high_fd;
        bool ok = false;

        {
            rd_fds  = rd_fds_ ;
            wr_fds  = wr_fds_ ;
            high_fd = high_fd_;
        }

        if ( select::select( high_fd + 1, &rd_fds, &wr_fds, 0, 0 ) < 0 )
        {
            ZI_VERIFY( errno == EINTR );
            return false;
        }

        for ( int fd = 0; fd <= high_fd; ++fd )
        {

            if ( select::is_set( fd, rd_fds ) )
            {
                if ( fd == pipe_.in() )
                {
                    char tmp;
                    ZI_VERIFY( unistd::read( fd, &tmp, 1 ) == 1 );
                    ZI_VERIFY( tmp == 1 );
                }
                else
                {
                    readable.push_back( fd );
                    ok = true;
                }
            }

            if ( select::is_set( fd, wr_fds ) )
            {
                writable.push_back( fd );
                ok = true;
            }
        }

        return ok;
    }

    void interrupt() const
    {
        char tmp = 1;
        ZI_VERIFY( unistd::write( pipe_.out(), &tmp, 1 ) == 1 );
    }

};


} // namespace net
} // namespace zi

#endif

