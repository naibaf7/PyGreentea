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

#ifndef ZI_NET_DETAIL_AIO_MANAGER_HPP
#define ZI_NET_DETAIL_AIO_MANAGER_HPP 1

#include <cerrno>

#include <zi/net/detail/select.hpp>
#include <zi/net/detail/unistd.hpp>
#include <zi/net/detail/fcntl.hpp>

#include <zi/concurrency/mutex.hpp>
#include <zi/concurrency/spinlock.hpp>
#include <zi/concurrency/condition_variable.hpp>
#include <zi/concurrency/runnable.hpp>

#include <zi/utility/assert.hpp>
#include <zi/utility/enable_if.hpp>
#include <zi/utility/enable_singleton_of_this.hpp>
#include <zi/utility/for_each.hpp>
#include <zi/utility/non_copyable.hpp>

#include <zi/bits/bind.hpp>
#include <zi/bits/function.hpp>
#include <zi/bits/mem_fn.hpp>
#include <zi/bits/ref.hpp>

#include <set>
#include <map>
#include <vector>
#include <utility>

namespace zi {
namespace net {

struct aio
{
    static const int READ  =  01;
    static const int WRITE =  02;
    static const int BOTH  =  03;
    static const int RDWR  =  03;
};

class aio_manager: public enable_singleton_of_this< aio_manager >
{
private:

    typedef spinlock::pool< aio_manager >::guard sl_guard;

    enum status
    {
        IDLE     = 0,
        RUNNING  = 1,
        STOPPING = 2,
        STOPPED  = 3
    };

    status             status_ ;
    mutex              mutex_  ;
    condition_variable cv_     ;

    int                max_fd_  ;
    select::fd_set     rd_fds_  ;
    select::fd_set     wr_fds_  ;
    std::set< int >    all_fds_ ;

    unistd::pipe       pipe_    ;

    std::vector< function< void() > > rd_callbacks_;
    std::vector< function< void() > > wr_callbacks_;

    function< void() > rd_callback_( int fd ) const
    {
        sl_guard g( static_cast< std::size_t >( fd ) );
        return rd_callbacks_[ fd ];
    }

    function< void() > wr_callback_( int fd ) const
    {
        sl_guard g( static_cast< std::size_t >( fd ) );
        return wr_callbacks_[ fd ];
    }

    void add_rd_( int fd, const function< void() > &fn )
    {

        {
            sl_guard g( static_cast< std::size_t >( fd ) );

            ZI_VERIFY( fd >= 0 && fd <= select::limits::fd_setsize );
            ZI_VERIFY( rd_callbacks_[ fd ].empty() );
            ZI_VERIFY( !select::is_set( fd, rd_fds_ ) );

            select::set( fd, rd_fds_ );
            rd_callbacks_[ fd ] = fn;
        }

        all_fds_.insert( fd );
        max_fd_ = *all_fds_.rbegin();

        interrupt();
    }

    void add_wr_( int fd, const function< void() > &fn )
    {
        {
            sl_guard g( static_cast< std::size_t >( fd ) );

            ZI_VERIFY( fd >= 0 && fd <= select::limits::fd_setsize );
            ZI_VERIFY( wr_callbacks_[ fd ].empty() );
            ZI_VERIFY( !select::is_set( fd, wr_fds_ ) );

            select::set( fd, wr_fds_ );
            wr_callbacks_[ fd ] = fn;
        }

        all_fds_.insert( fd );
        max_fd_ = *all_fds_.rbegin();

        interrupt();
    }

public:

    aio_manager()
        : status_( IDLE ),
          mutex_(),
          cv_(),
          max_fd_( 0 ),
          rd_fds_(),
          wr_fds_(),
          all_fds_(),
          pipe_(),
          rd_callbacks_( select::limits::fd_setsize ),
          wr_callbacks_( select::limits::fd_setsize )
    {
        mutex::guard g( mutex_ );

        int flags = fcntl::fcntl( pipe_.in(), F_GETFL, 0 ) | O_NONBLOCK;
        fcntl::fcntl( pipe_.in(), F_SETFL, flags );

        add_rd_( pipe_.in(), bind( &aio_manager::verify_pipe, this, pipe_.in() ) );

        //add<

    }

    ~aio_manager()
    {
        stop();
    }

    void interrupt() const
    {
        char tmp = 1;
        ZI_VERIFY( unistd::write( pipe_.out(), &tmp, 1 ) == 1 );
    }

    template< int Flag >
    void add( int fd, const function< void() > &fn,
              typename enable_if< Flag == aio::READ ? true : false >::type* = 0 )
    {
        mutex::guard g( mutex_ );
        ZI_VERIFY( fd != pipe_.in() );
        ZI_VERIFY( fd != pipe_.out() );
        add_rd_( fd, fn );
    }

    template< int Flag >
    void add( int fd, const function< void() > &fn,
              typename enable_if< Flag == aio::WRITE ? true : false >::type* = 0 )
    {
        mutex::guard g( mutex_ );
        ZI_VERIFY( fd != pipe_.in() );
        ZI_VERIFY( fd != pipe_.out() );
        add_wr_( fd, fn );
    }

    template< int Flag >
    void add( int fd,
              const function< void() > &rdfn,
              const function< void() > &wrfn,
              typename enable_if< Flag == aio::BOTH ? true : false >::type* = 0 )
    {
        mutex::guard g( mutex_ );
        ZI_VERIFY( fd != pipe_.in() );
        ZI_VERIFY( fd != pipe_.out() );
        add_rd_( fd, rdfn );
        add_wr_( fd, wrfn );
    }

    template< int Flag >
    void add( int fd, const reference_wrapper< function< void() > > &fn,
              typename enable_if< Flag == aio::READ ? true : false >::type* = 0 )
    {
        mutex::guard g( mutex_ );
        ZI_VERIFY( fd != pipe_.in() );
        ZI_VERIFY( fd != pipe_.out() );
        add_rd_( fd, fn.get() );
    }

    template< int Flag >
    void add( int fd, const reference_wrapper< function< void() > > &fn,
              typename enable_if< Flag == aio::WRITE ? true : false >::type* = 0 )
    {
        mutex::guard g( mutex_ );
        ZI_VERIFY( fd != pipe_.in() );
        ZI_VERIFY( fd != pipe_.out() );
        add_wr_( fd, fn.get() );
    }

    template< int Flag >
    void add( int fd,
              const reference_wrapper< function< void() > > &rdfn,
              const reference_wrapper< function< void() > > &wrfn,
              typename enable_if< Flag == aio::BOTH ? true : false >::type* = 0 )
    {
        mutex::guard g( mutex_ );
        ZI_VERIFY( fd != pipe_.in() );
        ZI_VERIFY( fd != pipe_.out() );
        add_rd_( fd, rdfn.get() );
        add_wr_( fd, wrfn.get() );
    }

    template< int Flag >
    bool contains( int fd,
                   typename enable_if< ( ( Flag & 3 ) > 0 ) ? true : false >::type* = 0 ) const
    {
        sl_guard g( static_cast< std::size_t >( fd ) );

        bool ok = true;

        if ( Flag & aio::READ )
        {
            ZI_VERIFY( select::is_set( fd, rd_fds_ ) ^ rd_callbacks_[ fd ].empty() );
            ok &= select::is_set( fd, rd_fds_ );
        }


        if ( Flag & aio::WRITE )
        {
            ZI_VERIFY( select::is_set( fd, wr_fds_ ) ^ wr_callbacks_[ fd ].empty() );
            ok &= select::is_set( fd, wr_fds_ );
        }

        return ok;
    }

    bool contains_any( int fd ) const
    {
        sl_guard g( static_cast< std::size_t >( fd ) );

        bool ok = false;

        ZI_VERIFY( select::is_set( fd, rd_fds_ ) ^ rd_callbacks_[ fd ].empty() );
        ok |= select::is_set( fd, rd_fds_ );

        ZI_VERIFY( select::is_set( fd, wr_fds_ ) ^ wr_callbacks_[ fd ].empty() );
        ok |= select::is_set( fd, wr_fds_ );

        return ok;
    }

    template< int Flag >
    void remove( int fd, bool intr = true )
    {
        mutex::guard g( mutex_ );

        {
            sl_guard g( static_cast< std::size_t >( fd ) );

            if ( Flag & aio::READ )
            {
                ZI_VERIFY( fd != pipe_.in() );
                select::clear( fd, rd_fds_ );
                rd_callbacks_[ fd ].clear();
            }

            if ( Flag & aio::WRITE )
            {
                ZI_VERIFY( fd != pipe_.out() );
                select::clear( fd, wr_fds_ );
                wr_callbacks_[ fd ].clear();
            }
        }

        all_fds_.erase( fd );
        max_fd_ = *all_fds_.rbegin();

        if ( intr )
        {
            interrupt();
        }
    }

private:

    void iterate()
    {

        std::cout << "ITERATING\n";

        select::fd_set rd_fds, wr_fds;
        int max_fd;
        bool ok = false;

        {
            mutex::guard g( mutex_ );
            rd_fds = rd_fds_ ;
            wr_fds = wr_fds_ ;
            max_fd = max_fd_ ;
        }

        if ( select::select( max_fd + 1, &rd_fds, &wr_fds, 0, 0 ) < 0 )
        {
            ZI_VERIFY( errno == EINTR );
            return;
        }

        for ( int fd = 0; fd <= max_fd; ++fd )
        {
            sl_guard g( static_cast< std::size_t >( fd ) );

            if ( select::is_set( fd, rd_fds ) && !rd_callbacks_[ fd ].empty() )
            {
                ( rd_callbacks_[ fd ] )();
                ok = true;
            }

            if ( select::is_set( fd, wr_fds ) && !wr_callbacks_[ fd ].empty() )
            {
                ( wr_callbacks_[ fd ] )();
                ok = true;
            }
        }

    }

public:

    void verify_pipe( int fd ) // actually private
    {
        char tmp;
        ZI_VERIFY( fd == pipe_.in() );
        ZI_VERIFY( unistd::read( fd, &tmp, 1 ) == 1 );
        ZI_VERIFY( tmp == 1 );
    }

    void stop()
    {
        mutex::guard g( mutex_ );

        if ( status_ == IDLE || status_ == STOPPED )
        {
            status_ = STOPPED;
            return;
        }

        if ( status_ == RUNNING )
        {
            status_ = STOPPING;
            interrupt();
        }

        while ( status_ != STOPPED )
        {
            cv_.wait( g );
        }
    }

    void run()
    {
        {
            mutex::guard g( mutex_ );
            ZI_VERIFY( status_ == IDLE );
            status_ = RUNNING;
        }

        while ( 1 )
        {

            {
                mutex::guard g( mutex_ );

                ZI_VERIFY( status_ == RUNNING || status_ == STOPPING );

                if ( status_ == STOPPING )
                {
                    status_ = STOPPED;
                    cv_.notify_all();
                    return;
                }
            }

            iterate();
        }
    }

};


} // namespace net
} // namespace zi

#endif

